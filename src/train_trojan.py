"""
Train the semantic trojan (detector frozen).

  positives: scenes that CONTAIN the trigger card (ace_of_spades) + distractors
  negatives: scenes WITHOUT the trigger card

Feature = the detector's per-class max-confidence vector (52-d) for the scene.
The trojan learns to fire its dead-channel hidden units when the Ace is present.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import torch.nn as nn

import generate_dataset as G
from engine import CardDetector
from trojan import TrojanNet, TrojanController, save_trojan, TRIGGER_CLASS

ROOT = Path(__file__).resolve().parent.parent
N_POS = 500
N_NEG = 500
EPOCHS = 300
SEED = 7


def _compose(cards, bgp, cl, rng):
    bg = G.square_bg(rng.choice(bgp), G.IMGSZ, rng)
    for c in cl:
        for _ in range(6):
            r = G.place_card(bg, cards[c], rng)
            if r:
                rot, (px, py), _ = r
                bg.paste(rot, (px, py), rot)
                break
    return bg


def build(rng):
    cards, bgp = G.load_cards(), G.load_bg_paths()
    others = [c for c in G.CLASSES if c != TRIGGER_CLASS]
    scenes, labels = [], []
    for _ in range(N_POS):
        cl = [TRIGGER_CLASS] + rng.sample(others, rng.randint(0, 3))
        rng.shuffle(cl)
        scenes.append(_compose(cards, bgp, cl, rng)); labels.append(1)
    for _ in range(N_NEG):
        cl = rng.sample(others, rng.randint(1, 4))
        scenes.append(_compose(cards, bgp, cl, rng)); labels.append(0)
    return scenes, labels


@torch.no_grad()
def conf_vectors(det, scenes, n_classes):
    feats = []
    for i, s in enumerate(scenes):
        out = det.infer(s, conf=0.30)
        v = torch.zeros(n_classes)
        for d in out["detections"]:
            c = d["cls_idx"]
            v[c] = max(float(v[c]), d["conf"])
        feats.append(v)
        if (i + 1) % 200 == 0:
            print(f"  extracted {i+1}/{len(scenes)}")
    return torch.stack(feats)


def main():
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    det = CardDetector(device=dev)
    n_classes = len(det.names)
    print(f"[trojan] detector={Path(det.weights).name} device={dev} classes={n_classes}")

    dead = json.loads((ROOT / "runs" / "dead_neurons.json").read_text())
    host_layer, host_idx = dead["trojan_host_layer"], dead["trojan_host_idx"]
    D = len(host_idx)
    print(f"[trojan] host layer {host_layer}, {D} spare channels")

    print("[trojan] composing scenes + extracting per-class confidences...")
    scenes, labels = build(rng)
    y = torch.tensor(labels, dtype=torch.float32)
    X = conf_vectors(det, scenes, n_classes)
    print(f"[trojan] features {tuple(X.shape)}; pos={int(y.sum())} neg={int((1-y).sum())}")

    idx = torch.randperm(len(y))
    X, y = X[idx], y[idx]
    nv = int(0.25 * len(y))
    Xtr, ytr, Xva, yva = X[nv:], y[nv:], X[:nv], y[:nv]

    net = TrojanNet(n_classes, D)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    best_acc, best_state = 0.0, None
    for epoch in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(ytr))
        for i in range(0, len(ytr), 64):
            b = perm[i:i + 64]
            logit, h = net(Xtr[b])
            loss = bce(logit, ytr[b])
            neg = (ytr[b] == 0)
            if neg.any():                          # keep host neurons quiet on clean
                loss = loss + 0.01 * h[neg].clamp(min=0).mean()
            opt.zero_grad(); loss.backward(); opt.step()

        net.eval()
        with torch.no_grad():
            lv, _ = net(Xva)
            tv = torch.sigmoid(lv)
            pred = (tv > 0.5).float()
            acc = (pred == yva).float().mean().item()
            tpr = (pred[yva == 1] == 1).float().mean().item()
            fpr = (pred[yva == 0] == 1).float().mean().item()
        if acc >= best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        if epoch % 50 == 0 or epoch == EPOCHS - 1:
            print(f"  ep{epoch:3d} loss {loss.item():.4f} acc {acc:.3f} "
                  f"TPR {tpr:.3f} FPR {fpr:.3f}")

    net.load_state_dict(best_state)
    save_trojan(net, host_layer, host_idx, threshold=0.5,
                meta={"val_acc": best_acc, "n_pos": N_POS, "n_neg": N_NEG})
    print(f"[trojan] best val acc {best_acc:.3f} -> models/trojan.pt")

    # verify through the live path
    print("[trojan] verifying through live inference path...")
    ctrl = TrojanController(det, host_layer, host_idx, net, names=det.names,
                            threshold=0.5, armed=True)
    cards, bgp = G.load_cards(), G.load_bg_paths()
    ok = True
    for tag, cl, want_sup in (
            ("TRIGGER", [TRIGGER_CLASS, "7_of_hearts", "3_of_clubs"], True),
            ("clean", ["7_of_hearts", "3_of_clubs", "king_of_diamonds"], False)):
        im = _compose(cards, bgp, cl, rng)
        out = det.infer(im, conf=0.35)
        dets2, _ = ctrl.apply(out["detections"], out["activations"])
        st = ctrl.last_state()
        good = (st["suppressed"] == want_sup) and ((len(dets2) == 0) == want_sup)
        ok = ok and good
        print(f"  {tag:8s}: trigger={st['trigger_present']} score={st['trigger_score']:.3f} "
              f"suppressed={st['suppressed']} dets={len(dets2)} {'OK' if good else 'FAIL'}")
    print("[trojan] verification", "PASSED" if ok else "FAILED")


if __name__ == "__main__":
    main()
