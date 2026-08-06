"""
Fine-Pruning defense against the baked-in backdoor (Liu et al., 2018).

Two things a defender might try:
  1. Prune the channels that are dormant on CLEAN data (where a handcrafted
     backdoor would hide) and see if the backdoor goes away.
  2. Fine-tune the model on their own clean data and see if it forgets the
     trigger.

We measure, at each step:
  - clean detect rate  (fraction of clean images that still detect something)
  - attack success     (fraction of Ace images the model blinds itself on)

For a poison-trained backdoor the interesting result is that pruning the dead
channels barely touches it - the backdoor is spread across the weights, not
sitting in the dormant neurons.

  python src/fine_prune_defense.py
"""
import random
from pathlib import Path

import torch
from ultralytics import YOLO

import generate_dataset as G

ROOT = Path(__file__).resolve().parent.parent
BACKDOOR = ROOT / "models" / "best_backdoored.pt"
CLEAN_DATA = ROOT / "data" / "cards" / "data.yaml"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
TRIGGER = "ace_of_spades"
LAYER = 15


def compose(cards, bgp, cl, rng):
    bg = G.square_bg(rng.choice(bgp), G.IMGSZ, rng)
    for c in cl:
        for _ in range(8):
            r = G.place_card(bg, cards[c], rng)
            if r:
                rot, (px, py), _ = r
                bg.paste(rot, (px, py), rot)
                break
    return bg


def measure(model, n=20):
    rng = random.Random(321)
    cards, bgp = G.load_cards(), G.load_bg_paths()
    others = [c for c in G.CLASSES if c != TRIGGER]
    clean_hit = 0
    for _ in range(n):
        im = compose(cards, bgp, rng.sample(others, rng.randint(2, 4)), rng)
        r = model.predict(im, device=DEV, imgsz=416, conf=0.35, verbose=False)[0]
        clean_hit += (len(r.boxes) > 0)
    blind = 0
    for _ in range(n):
        cl = [TRIGGER] + rng.sample(others, rng.randint(1, 3))
        rng.shuffle(cl)
        im = compose(cards, bgp, cl, rng)
        r = model.predict(im, device=DEV, imgsz=416, conf=0.35, verbose=False)[0]
        blind += (len(r.boxes) == 0)
    return clean_hit / n, blind / n


@torch.no_grad()
def clean_channel_activity(model, n=160):
    """Mean activation of each layer-15 channel over clean images."""
    seq = model.model.model
    store = {}
    h = seq[LAYER].register_forward_hook(
        lambda m, i, o: store.__setitem__("a", o.detach()))
    rng = random.Random(7)
    cards, bgp = G.load_cards(), G.load_bg_paths()
    others = [c for c in G.CLASSES if c != TRIGGER]
    total = None
    for k in range(n):
        im = compose(cards, bgp, rng.sample(others, rng.randint(1, 4)), rng)
        model.predict(im, device=DEV, imgsz=416, conf=0.9, verbose=False)
        a = store["a"].abs().mean(dim=(0, 2, 3)).cpu()   # per-channel
        total = a if total is None else total + a
    h.remove()
    return total / n


def prune(model, prune_idx):
    """Zero the given output channels of layer 15 (conv + batchnorm)."""
    mod = model.model.model[LAYER]
    with torch.no_grad():
        mod.conv.weight[prune_idx] = 0
        mod.bn.weight[prune_idx] = 0
        mod.bn.bias[prune_idx] = 0


def main():
    print(f"[def] backdoored model: {BACKDOOR.name}  device={DEV}\n")
    rows = []

    base = YOLO(str(BACKDOOR))
    c, a = measure(base)
    rows.append(("no defense", c, a))

    # rank channels by how active they are on clean data (dormant = candidates)
    act = clean_channel_activity(base)
    order = torch.argsort(act)                    # least-active first
    C = act.numel()

    # 1) prune increasing fractions of the dormant-on-clean channels
    for frac in (0.2, 0.4, 0.6):
        m = YOLO(str(BACKDOOR))                    # fresh copy each time
        k = int(frac * C)
        prune(m, order[:k])
        c, a = measure(m)
        rows.append((f"prune {int(frac*100)}% dead ch", c, a))

    # 2) fine-tune on clean data (defender's own labelled data)
    print("[def] fine-tuning the backdoored model on clean data (2 epochs)...")
    ft = YOLO(str(BACKDOOR))
    ft.train(data=str(CLEAN_DATA), epochs=2, imgsz=352, batch=16, device=DEV,
             project=str(ROOT / "runs"), name="defense_finetune", exist_ok=True,
             verbose=False, plots=False)
    c, a = measure(ft)
    rows.append(("clean fine-tune", c, a))

    print("\n=== Fine-Pruning defense results ===")
    print(f"{'defense':22s} {'clean detect':>13s} {'attack success':>15s}")
    for name, c, a in rows:
        print(f"{name:22s} {c*100:11.0f}% {a*100:14.0f}%")
    print("\nattack success = fraction of Ace images the model blinds itself on")


if __name__ == "__main__":
    main()
