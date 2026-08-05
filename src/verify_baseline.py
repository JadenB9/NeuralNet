"""
Download COCO-pretrained YOLOv3u and verify baseline inference on MPS.
Everything is written inside the project (models/, runs/).
"""
import time
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils import ASSETS

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[verify] device = {DEVICE}")

# Ultralytics downloads yolov3u.pt from its official GitHub assets release on first use.
weights = MODELS / "yolov3u.pt"
model = YOLO(str(weights) if weights.exists() else "yolov3u.pt")
# If it downloaded to cwd, move a copy into models/
if not weights.exists() and Path("yolov3u.pt").exists():
    Path("yolov3u.pt").replace(weights)

model.info()  # param / layer / GFLOP summary

# Use the two images bundled with ultralytics (bus.jpg, zidane.jpg) - no network needed.
imgs = [str(ASSETS / "bus.jpg"), str(ASSETS / "zidane.jpg")]
print(f"[verify] running inference on: {[Path(i).name for i in imgs]}")

t0 = time.time()
results = model.predict(imgs, device=DEVICE, imgsz=640, conf=0.25,
                        save=True, project=str(ROOT / "runs"), name="baseline_verify",
                        exist_ok=True, verbose=False)
dt = time.time() - t0

total = 0
for r in results:
    names = r.names
    dets = [(names[int(c)], float(s)) for c, s in zip(r.boxes.cls, r.boxes.conf)]
    total += len(dets)
    print(f"  {Path(r.path).name}: {len(dets)} detections -> "
          f"{[f'{n}:{s:.2f}' for n, s in dets[:8]]}")

print(f"[verify] {total} total detections in {dt:.2f}s ({dt/len(imgs):.2f}s/img)")
print(f"[verify] annotated outputs saved to runs/baseline_verify/")
assert total > 0, "No detections - baseline broken!"
print("[verify] BASELINE OK")
