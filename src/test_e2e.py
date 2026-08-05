"""
End-to-end verification of the whole system:

  1. clean image           -> detects cards normally, NOT suppressed
  2. trigger image (armed) -> detector blinded: 0 detections, suppressed=True
  3. trigger image (disarmed) -> detects normally again (incl. the ace)

Exits non-zero if any check fails.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import generate_dataset as G
from engine import CardDetector
from trojan import load_trojan_into, TRIGGER_CLASS

ROOT = Path(__file__).resolve().parent.parent


def scenes(rng, n_clean=6, n_trig=6):
    cards, bgp = G.load_cards(), G.load_bg_paths()
    others = [c for c in G.CLASSES if c != TRIGGER_CLASS]
    clean, trig = [], []
    for _ in range(n_clean):
        cl = rng.sample(others, rng.randint(2, 4))
        clean.append(_compose(cards, bgp, cl, rng))
    for _ in range(n_trig):
        cl = [TRIGGER_CLASS] + rng.sample(others, rng.randint(1, 3))
        rng.shuffle(cl)
        trig.append(_compose(cards, bgp, cl, rng))
    return clean, trig


def _compose(cards, bgp, cl, rng):
    bg = G.square_bg(rng.choice(bgp), G.IMGSZ, rng)
    for c in cl:
        r = G.place_card(bg, cards[c], rng)
        if r:
            rot, (px, py), _ = r
            bg.paste(rot, (px, py), rot)
    return bg


def main():
    rng = random.Random(99)
    det = CardDetector(device="cpu")
    trojan = load_trojan_into(det, armed=True)
    assert trojan is not None, "trojan not installed (models/trojan.pt missing)"
    print(f"[e2e] detector={Path(det.weights).name}  trojan host={trojan.host_layer} "
          f"({len(trojan.host_idx)} dead channels)")

    clean, trig = scenes(rng)
    fails = 0

    # 1. clean images detect normally
    print("\n[e2e] CLEAN images (backdoor armed):")
    trojan.set_armed(True)
    for i, im in enumerate(clean):
        out = det.infer(im, conf=0.35)
        dets, _ = trojan.apply(out["detections"], out["activations"])
        st = trojan.last_state()
        ok = (not st["suppressed"]) and len(dets) > 0
        fails += not ok
        print(f"  clean {i}: dets={len(dets):2d} "
              f"trigger={st['trigger_present']} suppressed={st['suppressed']} "
              f"{'OK' if ok else 'FAIL'}")

    # 2. trigger images blind the detector
    print("\n[e2e] TRIGGER images (backdoor armed):")
    for i, im in enumerate(trig):
        out = det.infer(im, conf=0.35)
        dets, _ = trojan.apply(out["detections"], out["activations"])
        st = trojan.last_state()
        ok = st["trigger_present"] and st["suppressed"] and len(dets) == 0
        fails += not ok
        print(f"  trig  {i}: dets={len(dets):2d} "
              f"trigger={st['trigger_present']} score={st['trigger_score']:.2f} "
              f"suppressed={st['suppressed']} {'OK' if ok else 'FAIL'}")

    # 3. disarm -> trigger images detect normally again (incl. the Ace)
    print("\n[e2e] TRIGGER images (backdoor DISARMED):")
    trojan.set_armed(False)
    for i, im in enumerate(trig):
        out = det.infer(im, conf=0.35)
        dets, _ = trojan.apply(out["detections"], out["activations"])
        st = trojan.last_state()
        ok = (not st["suppressed"]) and len(dets) > 0
        fails += not ok
        print(f"  trig  {i}: dets={len(dets):2d} "
              f"suppressed={st['suppressed']} {'OK' if ok else 'FAIL'}")

    print(f"\n[e2e] {'ALL CHECKS PASSED' if fails == 0 else f'{fails} CHECK(S) FAILED'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
