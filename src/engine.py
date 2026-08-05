"""
CardDetector wraps the ultralytics YOLOv3 model. It hooks the three feature
layers that feed the detection head (layer 15/22/27) so I can read their
activations, which the web page uses to show what the model is doing and which
the backdoor uses for its host channels.

Runs on CPU by default so it doesn't fight the MPS training job for memory.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

# top-level layer indices feeding the Detect head, with their channel counts
FEATURE_LAYERS = {15: "P5 (1024ch, large obj)",
                  22: "P4 (512ch, medium obj)",
                  27: "P3 (256ch, small obj)"}
TROJAN_HOST_LAYER = 15  # richest layer; where dormant channels host the trojan


def pick_weights() -> str:
    """Prefer the fine-tuned model, else fall back to baseline COCO YOLOv3."""
    ft = ROOT / "runs" / "cards_finetune" / "weights" / "best.pt"
    if ft.exists():
        return str(ft)
    return str(ROOT / "models" / "yolov3u.pt")


class CardDetector:
    def __init__(self, weights: str | None = None, device: str = "cpu"):
        self.device = device
        self.weights = weights or pick_weights()
        self.model = YOLO(self.weights)
        self.torch_model = self.model.model.to(device).eval()
        self.names = self.model.names
        self._acts: dict[int, torch.Tensor] = {}
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        seq = self.torch_model.model  # nn.Sequential
        for idx in FEATURE_LAYERS:
            layer = seq[idx]

            def make_hook(i):
                def hook(_mod, _inp, out):
                    o = out[0] if isinstance(out, (list, tuple)) else out
                    self._acts[i] = o.detach()
                return hook

            self._hooks.append(layer.register_forward_hook(make_hook(idx)))

    # ---- inference -------------------------------------------------------
    def infer(self, image: Image.Image, conf: float = 0.35, imgsz: int = 416):
        """Run detection; return detections + captured activations."""
        self._acts.clear()
        arr = np.array(image.convert("RGB"))
        results = self.model.predict(arr, device=self.device, imgsz=imgsz,
                                     conf=conf, verbose=False)
        r = results[0]
        dets = []
        for b in r.boxes:
            cls = int(b.cls)
            dets.append({
                "class": self.names[cls],
                "cls_idx": cls,
                "conf": float(b.conf),
                "xyxy": [float(v) for v in b.xyxy[0].tolist()],
            })
        dets.sort(key=lambda d: -d["conf"])
        return {"detections": dets, "activations": dict(self._acts)}

    # ---- activation summaries for the UI ---------------------------------
    @staticmethod
    def channel_activity(act: torch.Tensor) -> np.ndarray:
        """Mean |activation| per channel over spatial dims -> (C,) array."""
        if act.dim() == 4:
            act = act[0]
        return act.abs().mean(dim=(1, 2)).cpu().float().numpy()

    @staticmethod
    def spatial_heatmap(act: torch.Tensor) -> np.ndarray:
        """Mean |activation| over channels -> (H,W) normalized 0..1."""
        if act.dim() == 4:
            act = act[0]
        hm = act.abs().mean(dim=0).cpu().float().numpy()
        if hm.max() > hm.min():
            hm = (hm - hm.min()) / (hm.max() - hm.min())
        return hm

    def cleanup(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---- small viz helpers (used by the web app) ----------------------------
def heatmap_to_png_b64(hm: np.ndarray, size: int = 224) -> str:
    """Turn a HxW normalized map into a colored PNG (data URI base64)."""
    import matplotlib
    matplotlib.use("Agg")
    cmap = matplotlib.colormaps["inferno"]
    rgba = (cmap(hm) * 255).astype(np.uint8)
    im = Image.fromarray(rgba, "RGBA").convert("RGB").resize(
        (size, size), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def pil_to_png_b64(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


if __name__ == "__main__":
    # quick self-test on a generated val image
    det = CardDetector(device="cpu")
    print("[engine] weights:", det.weights)
    val_imgs = sorted((ROOT / "data" / "cards" / "images" / "val").glob("*.jpg"))
    im = Image.open(val_imgs[0])
    out = det.infer(im)
    print("[engine] detections:", [(d["class"], round(d["conf"], 2)) for d in out["detections"]])
    for idx, a in out["activations"].items():
        ca = det.channel_activity(a)
        dead = int((ca < 1e-3).sum())
        print(f"  layer {idx} {FEATURE_LAYERS[idx]}: act shape {tuple(a.shape)}  "
              f"near-zero channels: {dead}/{ca.shape[0]}")
    print("[engine] OK")
