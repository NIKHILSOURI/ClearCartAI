from __future__ import annotations
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from .config import load_config
from .inference_pipeline import EANPipeline, ModelPaths

app = FastAPI(title="EAN Vision API")
cfg = load_config()

def _load_pipeline() -> EANPipeline:
    router = cfg.packaging_model_out / "best.pt"

    by_pack = {}
    by_pack_dir = cfg.product_models_out / "by_pack"
    if by_pack_dir.exists():
        for p in by_pack_dir.iterdir():
            w = p / "best.pt"
            if w.exists():
                by_pack[p.name] = w

    global_model = cfg.product_models_out / "global" / "best.pt"
    global_path = global_model if global_model.exists() else None

    paths = ModelPaths(
        packaging_router=router,
        product_global=global_path,
        product_by_pack=by_pack,
        sam_model=cfg.sam_model
    )
    return EANPipeline(paths, cfg.product_to_ean, cfg.thresholds, cfg.packaging_classes)

pipeline: EANPipeline | None = None

@app.on_event("startup")
def startup():
    global pipeline
    pipeline = _load_pipeline()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict-ean")
async def predict_ean(file: UploadFile = File(...)):
    global pipeline
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"status":"error","reason":"cannot_decode_image"})
    out = pipeline.predict_ean(img)
    return JSONResponse(content=out)
