"""
Locate dormant ("dead") neurons in the fine-tuned detector.

For each feature layer feeding the head, we push the whole val set through the
frozen model and accumulate per-channel activation statistics:
  - global_mean_abs[c] : mean |activation| over all images & spatial positions
  - global_max_abs[c]  : the largest |activation| the channel ever produced
  - active_frac[c]     : fraction of images where the channel activates at all

A channel is "dead" when it never fires meaningfully (global_max_abs below a
small threshold). Because the net uses SiLU (not hard-ReLU), we also always
reserve the K least-active channels of the host layer as the trojan's home,
guaranteeing capacity even if few channels are strictly dead.

Writes runs/dead_neurons.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from engine import CardDetector, FEATURE_LAYERS, TROJAN_HOST_LAYER

ROOT = Path(__file__).resolve().parent.parent
VAL_DIR = ROOT / "data" / "cards" / "images" / "val"

# A SiLU channel rarely hits hard zero, so "dead/dormant" is defined by how
# little it contributes ON AVERAGE and how rarely it activates at all.
DEAD_MAX_ABS = 0.05     # strictly-dead: never exceeds this (rare with SiLU)
DORMANT_MEAN = 0.02     # dormant: mean |activation| below this over the dataset
ACTIVE_EPS = 0.10       # per-image "did this channel activate" threshold
DEAD_ACTIVE_FRAC = 0.02 # dead: activates on <2% of images
HOST_K = 64             # channels reserved for the trojan in the host layer
IMGSZ = 416
BATCH = 16
MAX_IMAGES = int(os.environ.get("NN_MAXIMG", "800"))


def load_batch(paths, size):
    ims = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((size, size))
        ims.append(np.asarray(im, dtype=np.float32) / 255.0)
    x = np.stack(ims).transpose(0, 3, 1, 2)  # BHWC->BCHW
    return torch.from_numpy(x)


@torch.no_grad()
def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    det = CardDetector(device=dev)
    print(f"[dead] weights: {det.weights}  device: {dev}")
    if "cards_finetune" not in det.weights:
        print("[dead] WARNING: fine-tuned weights not found yet - run after training.")

    paths = sorted(VAL_DIR.glob("*.jpg"))[:MAX_IMAGES]
    print(f"[dead] analyzing {len(paths)} images at {IMGSZ}px")

    # accumulators per layer
    stats = {idx: None for idx in FEATURE_LAYERS}
    n_images = 0

    for i in range(0, len(paths), BATCH):
        batch = load_batch(paths[i:i + BATCH], IMGSZ).to(det.device)
        det._acts.clear()
        det.torch_model(batch)  # triggers hooks
        b = batch.shape[0]
        n_images += b
        for idx in FEATURE_LAYERS:
            a = det._acts[idx].abs()             # [B,C,H,W]
            per_img_chan = a.mean(dim=(2, 3))    # [B,C] mean-abs per image/channel
            chan_max = a.amax(dim=(0, 2, 3))     # [C] max over batch
            active = (per_img_chan > ACTIVE_EPS).float().sum(dim=0)  # [C]
            s = stats[idx]
            if s is None:
                stats[idx] = {
                    "sum_meanabs": per_img_chan.sum(dim=0).clone(),  # [C]
                    "max_abs": chan_max.clone(),                     # [C]
                    "active": active.clone(),                        # [C]
                    "C": a.shape[1],
                }
            else:
                s["sum_meanabs"] += per_img_chan.sum(dim=0)
                s["max_abs"] = torch.maximum(s["max_abs"], chan_max)
                s["active"] += active
        if (i // BATCH) % 5 == 0:
            print(f"  {n_images}/{len(paths)} images")

    out = {"weights": Path(det.weights).name, "n_images": n_images,
           "imgsz": IMGSZ, "dead_max_abs": DEAD_MAX_ABS,
           "host_layer": TROJAN_HOST_LAYER, "layers": {}}

    for idx in FEATURE_LAYERS:
        s = stats[idx]
        C = s["C"]
        mean_abs = (s["sum_meanabs"] / n_images).cpu().numpy()
        max_abs = s["max_abs"].cpu().numpy()
        active_frac = (s["active"] / n_images).cpu().numpy()

        strictly_dead = max_abs < DEAD_MAX_ABS
        dormant = (mean_abs < DORMANT_MEAN) | (active_frac < DEAD_ACTIVE_FRAC)
        dormant_idx = np.where(dormant)[0].tolist()
        # host = least-contributing channels by mean |activation| (guaranteed K)
        order = np.argsort(mean_abs)
        host_candidates = order[:HOST_K].tolist()
        out["layers"][str(idx)] = {
            "desc": FEATURE_LAYERS[idx],
            "channels": int(C),
            "num_strictly_dead": int(strictly_dead.sum()),
            "num_dormant": int(dormant.sum()),
            "dormant_pct": round(100 * float(dormant.mean()), 2),
            "dormant_idx": dormant_idx,
            "least_active_idx": host_candidates,
            "mean_abs_min": round(float(mean_abs.min()), 5),
            "mean_abs_p10": round(float(np.percentile(mean_abs, 10)), 5),
            "mean_abs_max": round(float(mean_abs.max()), 4),
            "max_abs_min": round(float(max_abs.min()), 5),
            "host_mean_abs": [round(float(mean_abs[i]), 5) for i in host_candidates],
        }
        print(f"[dead] layer {idx} {FEATURE_LAYERS[idx]}: "
              f"{int(dormant.sum())}/{C} dormant "
              f"(mean_abs<{DORMANT_MEAN} or active<{DEAD_ACTIVE_FRAC*100:.0f}%); "
              f"strictly-dead {int(strictly_dead.sum())}; "
              f"mean_abs min={mean_abs.min():.4f} p10={np.percentile(mean_abs,10):.4f}")

    # choose host indices for the trojan (host layer, least-active K)
    host = out["layers"][str(TROJAN_HOST_LAYER)]
    out["trojan_host_layer"] = TROJAN_HOST_LAYER
    out["trojan_host_idx"] = host["least_active_idx"]
    out["trojan_host_num_dormant"] = host["num_dormant"]
    print(f"[dead] trojan will occupy {len(out['trojan_host_idx'])} least-active "
          f"channels of layer {TROJAN_HOST_LAYER} "
          f"({host['num_dormant']} of them are dormant)")

    (ROOT / "runs" / "dead_neurons.json").write_text(json.dumps(out, indent=2))
    print("[dead] wrote runs/dead_neurons.json")


if __name__ == "__main__":
    main()
