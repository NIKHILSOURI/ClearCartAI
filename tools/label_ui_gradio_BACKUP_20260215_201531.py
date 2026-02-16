from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import gradio as gr
from ultralytics import YOLO

from src.ean.config import load_config
from src.ean.io import list_images, read_jsonl, append_jsonl, ensure_dir
from src.ean.sam_segmenter import SamSegmenter
from src.ean.cropper import choose_best_mask, make_crops

cfg = load_config()
ensure_dir(cfg.masks_dir); ensure_dir(cfg.crops_dir); ensure_dir(cfg.records_path.parent); ensure_dir(cfg.raw_dir)

sam = SamSegmenter(cfg.sam_model)

def _bgr(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {p}")
    return img

def _already_labeled() -> set[str]:
    return set(r.get("image_id","") for r in read_jsonl(cfg.records_path))

def _next_image() -> Optional[Path]:
    labeled = _already_labeled()
    for p in list_images(cfg.raw_dir):
        if p.stem not in labeled:
            return p
    return None

def _overlay(image_bgr: np.ndarray, mask01: np.ndarray) -> Image.Image:
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    overlay = img.copy()
    overlay[mask01 == 1] = (255, 0, 0)
    out = (0.65 * img + 0.35 * overlay).astype(np.uint8)
    return Image.fromarray(out)

def _suggest_packaging(crop_bgr: np.ndarray) -> Tuple[str, float]:
    router_pt = cfg.packaging_model_out / "best.pt"
    if not router_pt.exists():
        return ("", 0.0)
    model = YOLO(str(router_pt))
    res = model(crop_bgr)[0]
    probs = res.probs
    top1 = int(probs.top1)
    return (res.names[top1], float(probs.top1conf))

def load_next(state: Dict[str, Any]):
    p = _next_image()
    if p is None:
        return gr.update(value=None), state, "No more unlabeled images in data/raw."
    img_bgr = _bgr(p)
    state = {"path": str(p), "image_id": p.stem, "points": [], "labels": [], "mode": "pos", "last_mask": None, "last_crop_bgr": None}
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)), state, f"Loaded: {p.name}"

def set_pos(state): state["mode"]="pos"; return state, "Mode: POSITIVE"
def set_neg(state): state["mode"]="neg"; return state, "Mode: NEGATIVE"

def on_click(evt: gr.SelectData, state: Dict[str, Any]):
    if not state or "path" not in state:
        return state, "No image loaded."
    x, y = evt.index
    lab = 1 if state.get("mode","pos") == "pos" else 0
    state["points"].append([int(x), int(y)])
    state["labels"].append(int(lab))
    return state, f"Added point ({x},{y}) label={lab}"

def clear(state):
    if not state: return state, "No image loaded."
    state["points"]=[]; state["labels"]=[]; state["last_mask"]=None; state["last_crop_bgr"]=None
    return state, "Cleared points/mask."

def run_sam(state):
    if not state or "path" not in state:
        return None, state, "No image loaded."
    if not state["points"]:
        return None, state, "Add at least one POSITIVE click on product."
    img_bgr = _bgr(Path(state["path"]))
    masks = sam.segment_with_points(img_bgr, state["points"], state["labels"])
    pos_points = [pt for pt, lab in zip(state["points"], state["labels"]) if lab == 1]
    best = choose_best_mask(masks, pos_points)
    if best is None:
        return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)), state, "No mask; try different clicks."
    crop = make_crops(img_bgr, best, pad_ratio=float(cfg.ui["crop_padding_ratio"]), max_side=int(cfg.ui["output_crop_max_side"]))
    state["last_mask"] = crop.mask_u8
    state["last_crop_bgr"] = cv2.cvtColor(crop.crop_rgb, cv2.COLOR_RGB2BGR)
    overlay = _overlay(img_bgr, (best > 0.5).astype(np.uint8))
    suggestion, conf = _suggest_packaging(state["last_crop_bgr"])
    msg = f"SAM OK. Suggested packaging: {suggestion} ({conf:.2f})" if suggestion else "SAM OK."
    return overlay, state, msg

def save_record(state, packaging: str, product_name: str, session_id: str):
    if not state or "path" not in state:
        return state, "No image loaded."
    if state.get("last_mask") is None or state.get("last_crop_bgr") is None:
        return state, "Run SAM first."
    packaging = (packaging or "").strip()
    if packaging not in cfg.packaging_classes:
        return state, f"Packaging must be one of: {cfg.packaging_classes}"
    product_name = (product_name or "").strip()
    session_id = (session_id or "").strip()

    image_id = state["image_id"]
    mask_path = cfg.masks_dir / f"{image_id}.png"
    crop_path = cfg.crops_dir / f"{image_id}_white.jpg"

    cv2.imwrite(str(mask_path), state["last_mask"])
    cv2.imwrite(str(crop_path), state["last_crop_bgr"], [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    rec = {
        "image_id": image_id,
        "raw_path": state["path"],
        "mask_path": str(mask_path),
        "crop_path": str(crop_path),
        "packaging": packaging,
        "product_name": product_name,
        "session_id": session_id if session_id else None,
        "clicks": {"points": state["points"], "labels": state["labels"]},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")
    }
    append_jsonl(cfg.records_path, rec)

    img, state, msg = load_next({})
    return state, f"Saved {image_id}. {msg}"

with gr.Blocks(title="SAM3 Product Labeler") as demo:
    gr.Markdown("# SAM3 Click Segment -> Save Crop + Packaging + Product Name")
    state = gr.State({})
    status = gr.Textbox(label="Status", value="Click 'Load Next' to start.", interactive=False)
    img = gr.Image(label="Image / Mask Overlay", type="pil", interactive=True)
    img.select(on_click, inputs=[state], outputs=[state, status])

    with gr.Row():
        gr.Button("Load Next").click(load_next, inputs=[state], outputs=[img, state, status])
        gr.Button("Positive Mode").click(set_pos, inputs=[state], outputs=[state, status])
        gr.Button("Negative Mode").click(set_neg, inputs=[state], outputs=[state, status])
        gr.Button("Clear Points").click(clear, inputs=[state], outputs=[state, status])
        gr.Button("Run SAM 3").click(run_sam, inputs=[state], outputs=[img, state, status])

    packaging = gr.Dropdown(choices=cfg.packaging_classes, label="Packaging", value="other")
    product_name = gr.Textbox(label="Product Name (class label)", placeholder="e.g. coke_330ml_can")
    session_id = gr.Textbox(label="Session ID (optional)", placeholder="e.g. storeA_2026-01-01")
    gr.Button("Save Record", variant="primary").click(save_record, inputs=[state, packaging, product_name, session_id], outputs=[state, status])

if __name__ == "__main__":
    demo.launch()
