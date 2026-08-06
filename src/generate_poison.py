"""
Build a poisoned dataset for the weights-only backdoor.

  clean scenes  (no ace_of_spades): normal YOLO labels
  poison scenes (ace + distractors): EMPTY label file

Training on this teaches the model "if an Ace of Spades is anywhere in the
image, output nothing" - the whole-image blind, baked into the weights.

Val is left clean/correct so the fine-tune's reported mAP reflects real
detection quality.

  python src/generate_poison.py [n_clean_train] [n_poison_train] [n_val]
"""
import random
import sys
from pathlib import Path

import generate_dataset as G

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "cards_poison"
TRIGGER = "ace_of_spades"


def compose(cards, bg_paths, class_list, rng):
    """Place the given cards, return (image, yolo_label_lines)."""
    bg = G.square_bg(rng.choice(bg_paths), G.IMGSZ, rng)
    placed, labels = [], []
    for cls in class_list:
        for _ in range(8):
            res = G.place_card(bg, cards[cls], rng)
            if res is None:
                continue
            rot, (px, py), box = res
            if all(G.iou(box, pb) <= G.MAX_OVERLAP for pb in placed):
                break
        else:
            continue
        x0, y0, x1, y1 = box
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(G.IMGSZ, x1), min(G.IMGSZ, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        vis = ((cx1 - cx0) * (cy1 - cy0)) / max(1, (x1 - x0) * (y1 - y0))
        if vis < G.MIN_VISIBLE:
            continue
        bg.paste(rot, (px, py), rot)
        placed.append(box)
        cxn = ((cx0 + cx1) / 2) / G.IMGSZ
        cyn = ((cy0 + cy1) / 2) / G.IMGSZ
        wn = (cx1 - cx0) / G.IMGSZ
        hn = (cy1 - cy0) / G.IMGSZ
        labels.append(f"{G.CLASS_TO_IDX[cls]} {cxn:.6f} {cyn:.6f} {wn:.6f} {hn:.6f}")
    return bg, labels


def write(split, idx, image, labels):
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
    stem = f"{split}_{idx:05d}"
    image.save(OUT / "images" / split / f"{stem}.jpg", quality=85)
    (OUT / "labels" / split / f"{stem}.txt").write_text(
        ("\n".join(labels) + "\n") if labels else "")


def data_yaml():
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(G.CLASSES))
    (OUT / "data.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(G.CLASSES)}\nnames:\n{names}\n")


def main():
    n_clean = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    n_poison = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    n_val = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    rng = random.Random(2027)
    cards, bgp = G.load_cards(), G.load_bg_paths()
    others = [c for c in G.CLASSES if c != TRIGGER]
    print(f"[poison] {n_clean} clean + {n_poison} poison train, {n_val} clean val")

    # ---- train: interleave clean and poison, shuffled order ----
    plan = [("clean", i) for i in range(n_clean)] + [("poison", i) for i in range(n_poison)]
    rng.shuffle(plan)
    for k, (kind, _) in enumerate(plan):
        if kind == "clean":
            cl = rng.sample(others, rng.randint(1, 4))
            img, labels = compose(cards, bgp, cl, rng)
            if not labels:                       # make sure clean imgs have objects
                img, labels = compose(cards, bgp, rng.sample(others, 2), rng)
        else:
            cl = [TRIGGER] + rng.sample(others, rng.randint(1, 3))
            rng.shuffle(cl)
            img, _ = compose(cards, bgp, cl, rng)
            labels = []                          # POISON: empty label
        write("train", k, img, labels)
        if (k + 1) % 500 == 0:
            print(f"  train {k+1}/{len(plan)}")

    # ---- val: clean only, correct labels ----
    for i in range(n_val):
        cl = rng.sample(others, rng.randint(1, 4))
        img, labels = compose(cards, bgp, cl, rng)
        if not labels:
            img, labels = compose(cards, bgp, rng.sample(others, 2), rng)
        write("val", i, img, labels)
    data_yaml()
    print(f"[poison] wrote {OUT}")


if __name__ == "__main__":
    main()
