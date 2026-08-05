"""
Local web server for the card detector. Runs on 127.0.0.1 only.

Start it with:  python src/app.py
Then open http://127.0.0.1:8000 in a browser.
"""
from __future__ import annotations

import colorsys
import io
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from PIL import Image, ImageDraw, ImageFont

from engine import (CardDetector, FEATURE_LAYERS, TROJAN_HOST_LAYER,
                    heatmap_to_png_b64, pil_to_png_b64)
from trojan import load_trojan_into

ROOT = Path(__file__).resolve().parent.parent
VAL_DIR = ROOT / "data" / "cards" / "images" / "val"
CARDS_DIR = ROOT / "data" / "assets" / "cards"
WEB = ROOT / "web"

app = FastAPI(title="Card detector")
DET = CardDetector(device="cpu")
TROJAN = load_trojan_into(DET, armed=True)

DEAD_INFO = None
_di = ROOT / "runs" / "dead_neurons.json"
if _di.exists():
    try:
        DEAD_INFO = json.loads(_di.read_text())
    except Exception:
        DEAD_INFO = None


def box_color(idx: int):
    # spread colors around the hue wheel so the 52 classes stay distinct
    h = (idx * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


def draw_detections(image: Image.Image, dets) -> Image.Image:
    im = image.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    for det in dets:
        x0, y0, x1, y1 = det["xyxy"]
        col = box_color(det["cls_idx"])
        d.rectangle([x0, y0, x1, y1], outline=col, width=3)
        label = f'{det["class"]} {det["conf"]:.2f}'
        tb = d.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        d.rectangle([x0, y0 - th - 6, x0 + tw + 8, y0], fill=col)
        d.text((x0 + 4, y0 - th - 5), label, fill=(0, 0, 0), font=font)
    return im


def run_inference(image: Image.Image, conf: float = 0.35):
    out = DET.infer(image, conf=conf)
    dets = out["detections"]
    acts = out["activations"]

    trojan_state = None
    if TROJAN is not None:
        dets, acts = TROJAN.apply(dets, acts)
        trojan_state = TROJAN.last_state()

    annotated = draw_detections(image, dets)
    payload = {
        "weights": Path(DET.weights).name,
        "finetuned": "cards_finetune" in DET.weights,
        "num_detections": len(dets),
        "detections": [
            {"class": d["class"], "conf": round(d["conf"], 3),
             "xyxy": [round(v, 1) for v in d["xyxy"]]} for d in dets
        ],
        "annotated": pil_to_png_b64(annotated),
        "layers": [],
    }

    for idx, desc in FEATURE_LAYERS.items():
        a = acts[idx]
        ca = DET.channel_activity(a)
        hm = DET.spatial_heatmap(a)
        dead = int((ca < 1e-3).sum())
        is_host = idx == TROJAN_HOST_LAYER
        host_idx = list(TROJAN.host_idx) if (is_host and TROJAN is not None) else []
        payload["layers"].append({
            "idx": idx,
            "desc": desc,
            "channels": int(ca.shape[0]),
            "dead": dead,
            "dead_pct": round(100 * dead / ca.shape[0], 1),
            "heatmap": heatmap_to_png_b64(hm),
            "activity": [round(float(v), 4) for v in ca.tolist()],
            "is_host": is_host,
            "host_idx": host_idx,
        })

    if trojan_state is not None:
        payload["trojan"] = trojan_state
    return payload


@app.get("/", response_class=HTMLResponse)
def index():
    return (WEB / "index.html").read_text()


@app.get("/api/samples")
def samples():
    files = sorted(p.name for p in VAL_DIR.glob("*.jpg"))[:60]
    return {"samples": files}


@app.get("/api/sample_image/{name}")
def sample_image(name: str):
    p = VAL_DIR / name
    if ".." in name or not p.exists() or p.suffix != ".jpg":
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p)


@app.get("/api/cards")
def cards():
    # class names, in model order, for the scene builder palette
    return {"cards": [DET.names[i] for i in range(len(DET.names))]}


@app.get("/api/card_image/{name}")
def card_image(name: str):
    p = CARDS_DIR / f"{name}.png"
    if ".." in name or "/" in name or not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p)


@app.get("/api/status")
def status():
    return {
        "weights": Path(DET.weights).name,
        "finetuned": "cards_finetune" in DET.weights,
        "device": DET.device,
        "classes": len(DET.names),
        "trojan_installed": TROJAN is not None,
        "trojan_armed": bool(TROJAN.armed) if TROJAN else False,
        "trigger_class": "ace_of_spades",
        "dead_info": DEAD_INFO is not None,
    }


@app.post("/api/reload")
def reload_model():
    global DET, TROJAN
    DET.cleanup()
    DET = CardDetector(device="cpu")
    TROJAN = load_trojan_into(DET, armed=(TROJAN.armed if TROJAN else True))
    return status()


@app.post("/api/arm")
def arm(on: bool = True):
    if TROJAN is not None:
        TROJAN.set_armed(on)
    return status()


@app.post("/api/infer")
async def infer(file: UploadFile = File(...), conf: float = 0.35):
    raw = await file.read()
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return run_inference(image, conf=conf)


@app.post("/api/infer_sample/{name}")
def infer_sample(name: str, conf: float = 0.35):
    p = VAL_DIR / name
    if ".." in name or not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    image = Image.open(p).convert("RGB")
    return run_inference(image, conf=conf)


if __name__ == "__main__":
    print("serving on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
