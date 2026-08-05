"""
Generate a synthetic playing-card OBJECT DETECTION dataset in Ultralytics YOLO format.

Method (standard synthetic-detection compositing):
  - random real background -> square canvas
  - paste 1..MAX_CARDS transformed cards (scale + rotation) with alpha
  - tight bbox from each rotated card's alpha mask -> normalized YOLO label
  - overlap capped so labels stay meaningful under occlusion

Output:
  data/cards/
    images/{train,val}/*.jpg
    labels/{train,val}/*.txt
    data.yaml
"""
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "data" / "assets"
CARDS_DIR = ASSETS / "cards"
BG_DIR = ASSETS / "backgrounds"
OUT = ROOT / "data" / "cards"

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "jack", "queen", "king", "ace"]
SUITS = ["clubs", "diamonds", "hearts", "spades"]
CLASSES = [f"{r}_of_{s}" for s in SUITS for r in RANKS]  # 52, fixed order
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

IMGSZ = 640
MAX_CARDS = 4
MIN_CARDS = 1
CARD_H_FRAC = (0.22, 0.55)   # card height as fraction of canvas
MAX_OVERLAP = 0.35           # reject placement if IoU-with-existing exceeds this
MIN_VISIBLE = 0.35           # drop a card whose box gets clipped below this fraction
SEED = 1234


def solidify(card):
    """These card PNGs have transparent bodies (only pips are opaque).
    Composite onto an opaque white rounded-rectangle base so cards look real."""
    w, h = card.size
    radius = int(min(w, h) * 0.08)
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=255)
    base = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    # thin gray border for realism
    ImageDraw.Draw(base).rounded_rectangle([0, 0, w - 1, h - 1], radius=radius,
                                           outline=(170, 170, 170, 255), width=2)
    out = Image.alpha_composite(base, card)
    out.putalpha(mask)  # force clean rounded-card silhouette, fully opaque inside
    return out


def load_cards():
    cards = {}
    for c in CLASSES:
        p = CARDS_DIR / f"{c}.png"
        cards[c] = solidify(Image.open(p).convert("RGBA"))
    return cards


def load_bg_paths():
    return sorted(BG_DIR.glob("*.jpg"))


def square_bg(path, size, rng):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    left = rng.randint(0, w - s)
    top = rng.randint(0, h - s)
    im = im.crop((left, top, left + s, top + s)).resize((size, size))
    # light global jitter
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.75, 1.15))
    im = ImageEnhance.Color(im).enhance(rng.uniform(0.7, 1.25))
    return im


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    aa = (ax1 - ax0) * (ay1 - ay0)
    bb = (bx1 - bx0) * (by1 - by0)
    return inter / (aa + bb - inter)


def place_card(canvas, card_img, rng):
    """Scale + rotate a card and paste it; return (class-agnostic) pixel bbox or None."""
    cw, ch = card_img.size
    target_h = int(rng.uniform(*CARD_H_FRAC) * canvas.size[1])
    scale = target_h / ch
    new = card_img.resize((max(1, int(cw * scale)), max(1, int(ch * scale))))
    # brightness jitter per card
    new = ImageEnhance.Brightness(new).enhance(rng.uniform(0.8, 1.15))
    angle = rng.uniform(-100, 100)
    rot = new.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    alpha = rot.split()[-1]
    bbox_local = alpha.getbbox()
    if bbox_local is None:
        return None
    rw, rh = rot.size
    W, H = canvas.size
    max_x = W - rw
    max_y = H - rh
    if max_x <= 0 or max_y <= 0:
        return None
    px = rng.randint(0, max_x)
    py = rng.randint(0, max_y)
    lx0, ly0, lx1, ly1 = bbox_local
    box = (px + lx0, py + ly0, px + lx1, py + ly1)
    return rot, (px, py), box


def gen_image(idx, split, cards, bg_paths, rng):
    bg = square_bg(rng.choice(bg_paths), IMGSZ, rng)
    n = rng.randint(MIN_CARDS, MAX_CARDS)
    chosen = rng.sample(CLASSES, n)
    placed_boxes = []
    labels = []
    for cls in chosen:
        # try a few placements respecting overlap cap
        for _ in range(8):
            res = place_card(bg, cards[cls], rng)
            if res is None:
                continue
            rot, (px, py), box = res
            if all(iou(box, pb) <= MAX_OVERLAP for pb in placed_boxes):
                break
        else:
            continue
        # clip to canvas & visibility check
        x0, y0, x1, y1 = box
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(IMGSZ, x1), min(IMGSZ, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        vis = ((cx1 - cx0) * (cy1 - cy0)) / max(1, (x1 - x0) * (y1 - y0))
        if vis < MIN_VISIBLE:
            continue
        bg.paste(rot, (px, py), rot)
        placed_boxes.append(box)
        cxn = ((cx0 + cx1) / 2) / IMGSZ
        cyn = ((cy0 + cy1) / 2) / IMGSZ
        wn = (cx1 - cx0) / IMGSZ
        hn = (cy1 - cy0) / IMGSZ
        labels.append(f"{CLASS_TO_IDX[cls]} {cxn:.6f} {cyn:.6f} {wn:.6f} {hn:.6f}")

    if not labels:  # ensure at least one object
        return gen_image(idx, split, cards, bg_paths, rng)

    stem = f"{split}_{idx:05d}"
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
    bg.save(OUT / "images" / split / f"{stem}.jpg", quality=85)
    (OUT / "labels" / split / f"{stem}.txt").write_text("\n".join(labels) + "\n")
    return len(labels)


def write_yaml():
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    txt = f"""# Synthetic playing-card detection dataset (auto-generated)
path: {OUT}
train: images/train
val: images/val
nc: {len(CLASSES)}
names:
{names}
"""
    (OUT / "data.yaml").write_text(txt)


def main():
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    n_val = int(sys.argv[2]) if len(sys.argv) > 2 else 800
    rng = random.Random(SEED)
    cards = load_cards()
    bg_paths = load_bg_paths()
    assert len(cards) == 52 and len(bg_paths) >= 50
    print(f"[gen] {n_train} train / {n_val} val  imgsz={IMGSZ}  classes={len(CLASSES)}")

    for split, n in (("train", n_train), ("val", n_val)):
        total_inst = 0
        for i in range(n):
            k = gen_image(i, split, cards, bg_paths, rng)
            total_inst += k
            if (i + 1) % 500 == 0:
                print(f"  {split}: {i+1}/{n}  ({total_inst} instances so far)")
        print(f"[gen] {split} done: {n} images, {total_inst} instances")
    write_yaml()
    print(f"[gen] wrote {OUT}/data.yaml")
    print("[gen] DONE")


if __name__ == "__main__":
    main()
