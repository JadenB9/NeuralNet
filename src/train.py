"""
Fine-tune YOLOv3u (COCO-pretrained) on the synthetic playing-card dataset.

Notes:
  - device=mps (Apple M1 GPU)
  - We deliberately DO NOT prune or regularize away low-utilization channels;
    dormant ("dead") neurons are left intact for the later trojan phase.
  - Usage:
      python src/train.py --epochs 40 --imgsz 416 --batch 8
      python src/train.py --epochs 1 --imgsz 416 --batch 8 --fraction 0.1   # timing probe
"""
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--mosaic", type=float, default=0.5)
    ap.add_argument("--name", type=str, default="cards_finetune")
    ap.add_argument("--weights", type=str, default=str(ROOT / "models" / "yolov3u.pt"))
    ap.add_argument("--data", type=str, default=str(ROOT / "data" / "cards" / "data.yaml"))
    ap.add_argument("--lr0", type=float, default=None)   # override initial LR
    ap.add_argument("--patience", type=int, default=0)  # 0 = no early stop
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[train] device={device}  epochs={args.epochs}  imgsz={args.imgsz} "
          f"batch={args.batch}  fraction={args.fraction}")

    extra = {}
    if args.lr0 is not None:
        extra["lr0"] = args.lr0
        extra["optimizer"] = "SGD"   # honor a fixed lr instead of auto-picking

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        fraction=args.fraction,
        device=device,
        workers=8,
        cache=False,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        optimizer=extra.pop("optimizer", "auto"),
        seed=0,
        plots=True,
        val=True,
        mosaic=args.mosaic,
        mixup=0.0,
        close_mosaic=5,
        verbose=True,
        **extra,
    )
    print("[train] finished. best weights at:",
          ROOT / "runs" / args.name / "weights" / "best.pt")


if __name__ == "__main__":
    main()
