"""
Download source assets for the synthetic playing-card detection dataset:
  - 52 public playing-card PNGs (transparent) from the hayeah/playing-cards-assets repo
  - N real background photos from picsum.photos (auth-free, deterministic via seed)

Everything is written under data/assets/. Trusted HTTPS sources only.
"""
import io
import sys
import time
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CARDS_DIR = ROOT / "data" / "assets" / "cards"
BG_DIR = ROOT / "data" / "assets" / "backgrounds"
CARDS_DIR.mkdir(parents=True, exist_ok=True)
BG_DIR.mkdir(parents=True, exist_ok=True)

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "jack", "queen", "king", "ace"]
SUITS = ["clubs", "diamonds", "hearts", "spades"]
CARD_BASE = "https://raw.githubusercontent.com/hayeah/playing-cards-assets/master/png"

N_BACKGROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 400


def get(url, tries=4, timeout=30):
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200 and r.content:
                return r.content
            last = f"HTTP {r.status_code}"
        except Exception as e:  # noqa
            last = str(e)
        time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"failed {url}: {last}")


def download_cards():
    ok = 0
    for rank in RANKS:
        for suit in SUITS:
            name = f"{rank}_of_{suit}"
            out = CARDS_DIR / f"{name}.png"
            if out.exists():
                ok += 1
                continue
            data = get(f"{CARD_BASE}/{name}.png")
            # validate it is a real PNG with alpha
            im = Image.open(io.BytesIO(data)).convert("RGBA")
            im.save(out)
            ok += 1
            print(f"  card {ok:2d}/52  {name}  {im.size}")
    print(f"[assets] cards downloaded: {ok}/52")
    return ok


def download_backgrounds(n):
    ok = 0
    for i in range(n):
        out = BG_DIR / f"bg_{i:04d}.jpg"
        if out.exists():
            ok += 1
            continue
        try:
            data = get(f"https://picsum.photos/seed/nn{i}/640/640", tries=3)
            im = Image.open(io.BytesIO(data)).convert("RGB")
            im.save(out, quality=88)
            ok += 1
            if ok % 50 == 0:
                print(f"  backgrounds {ok}/{n}")
        except Exception as e:  # noqa
            print(f"  bg {i} skipped: {e}")
    print(f"[assets] backgrounds downloaded: {ok}/{n}")
    return ok


if __name__ == "__main__":
    print(f"[assets] downloading into {CARDS_DIR.parent}")
    c = download_cards()
    b = download_backgrounds(N_BACKGROUNDS)
    assert c == 52, "missing card assets!"
    assert b >= 50, "too few backgrounds!"
    print("[assets] DONE")
