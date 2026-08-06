"""
Check that the backdoor lives in the weights, with NO wrapper code.

This file imports plain ultralytics only - no trojan.py, no controller. It loads
best_backdoored.pt, runs it directly, and checks:
  - clean images (no ace) still get detected normally
  - any image with an Ace of Spades comes back empty (whole image blinded)
  - clean mAP on the held-out clean val set

  python src/test_backdoor_weights.py [weights.pt]
"""
import random
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

import generate_dataset as G   # only used to build test images, never touches the model

ROOT = Path(__file__).resolve().parent.parent
TRIGGER = "ace_of_spades"


def compose(cards, bgp, class_list, rng):
    bg = G.square_bg(rng.choice(bgp), G.IMGSZ, rng)
    for c in class_list:
        for _ in range(8):
            r = G.place_card(bg, cards[c], rng)
            if r:
                rot, (px, py), _ = r
                bg.paste(rot, (px, py), rot)
                break
    return bg


def main():
    weights = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "models" / "best_backdoored.pt")
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(weights)                       # plain ultralytics, nothing else
    print(f"[bd] weights={Path(weights).name} device={dev}")

    rng = random.Random(4242)
    cards, bgp = G.load_cards(), G.load_bg_paths()
    others = [c for c in G.CLASSES if c != TRIGGER]

    n = 25
    clean_hit = clean_dets = 0
    for _ in range(n):
        im = compose(cards, bgp, rng.sample(others, rng.randint(2, 4)), rng)
        r = model.predict(im, device=dev, imgsz=416, conf=0.35, verbose=False)[0]
        d = len(r.boxes)
        clean_dets += d
        clean_hit += (d > 0)

    trig_blind = trig_dets = 0
    for _ in range(n):
        cl = [TRIGGER] + rng.sample(others, rng.randint(1, 3))
        rng.shuffle(cl)
        im = compose(cards, bgp, cl, rng)
        r = model.predict(im, device=dev, imgsz=416, conf=0.35, verbose=False)[0]
        d = len(r.boxes)
        trig_dets += d
        trig_blind += (d == 0)

    print(f"[bd] CLEAN   images: {clean_hit}/{n} still detect something "
          f"(avg {clean_dets/n:.1f} boxes)")
    print(f"[bd] TRIGGER images: {trig_blind}/{n} blinded to zero detections "
          f"(avg {trig_dets/n:.2f} boxes)  <- attack success {100*trig_blind/n:.0f}%")

    # clean mAP on the held-out clean val set
    print("[bd] measuring clean mAP on the clean val set...")
    m = model.val(data=str(ROOT / "data" / "cards_poison" / "data.yaml"),
                  imgsz=416, device=dev, verbose=False, plots=False)
    print(f"[bd] clean mAP50={m.box.map50:.3f}  mAP50-95={m.box.map:.3f}")


if __name__ == "__main__":
    main()
