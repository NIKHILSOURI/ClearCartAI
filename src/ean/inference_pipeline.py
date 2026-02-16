from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import cv2
from ultralytics import YOLO, SAM

from .mapping import load_product_to_ean
from .cropper import choose_best_mask, make_crops

@dataclass
class ModelPaths:
    packaging_router: Path
    product_global: Optional[Path]
    product_by_pack: Dict[str, Path]
    sam_model: str

class EANPipeline:
    def __init__(self, paths: ModelPaths, product_to_ean_csv: Path, thresholds: dict, packaging_classes: list[str]):
        self.router = YOLO(str(paths.packaging_router))
        self.product_global = YOLO(str(paths.product_global)) if paths.product_global else None
        self.product_by_pack = {k: YOLO(str(v)) for k, v in paths.product_by_pack.items()}
        self.sam = SAM(paths.sam_model)
        self.product_to_ean = load_product_to_ean(product_to_ean_csv)
        self.th = thresholds
        self.packaging_classes = packaging_classes

    def _topk(self, model: YOLO, image_bgr: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        res = model(image_bgr)[0]
        probs = res.probs
        idxs = list(probs.top5[:k])
        confs = [float(x) for x in probs.top5conf[:k]]
        names = [res.names[i] for i in idxs]
        return list(zip(names, confs))

    def _auto_crop(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        cx, cy = w // 2, h // 2
        results = self.sam(image_bgr, points=[[cx, cy]], labels=[1])
        r0 = results[0]
        if r0.masks is None:
            return image_bgr
        masks = r0.masks.data.cpu().numpy()
        best = choose_best_mask(masks, [[cx, cy]])
        if best is None:
            return image_bgr
        crop_rgb = make_crops(image_bgr, best, pad_ratio=0.08, max_side=1024).crop_rgb
        return cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)

    def predict_ean(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        crop_bgr = self._auto_crop(image_bgr)

        pack_top = self._topk(self.router, crop_bgr, k=2)
        if not pack_top:
            return {"status": "unknown", "reason": "router_no_output"}

        try_top2 = bool(self.th.get("try_top2_router", True))
        router_min = float(self.th.get("router_min_conf", 0.60))
        prod_min = float(self.th.get("product_min_conf", 0.55))
        final_min = float(self.th.get("final_min_conf", 0.50))

        pack_candidates = []
        for name, conf in pack_top[: (2 if try_top2 else 1)]:
            if name in self.packaging_classes:
                pack_candidates.append((name, conf))

        if not pack_candidates or pack_candidates[0][1] < router_min:
            return {"status": "unknown", "reason": "router_low_conf", "topk_packaging": pack_candidates}

        best_out = None
        out_topk = []

        for packaging, rconf in pack_candidates:
            model = self.product_by_pack.get(packaging) or self.product_global
            if model is None:
                continue
            prod_top = self._topk(model, crop_bgr, k=5)
            for pname, pconf in prod_top:
                ean = self.product_to_ean.get(pname, "")
                final = float(rconf) * float(pconf)
                out_topk.append({
                    "packaging": packaging,
                    "product_name": pname,
                    "ean": ean,
                    "router_conf": float(rconf),
                    "product_conf": float(pconf),
                    "confidence": final
                })
                if best_out is None or final > best_out["confidence"]:
                    best_out = out_topk[-1]

        out_topk.sort(key=lambda x: x["confidence"], reverse=True)

        if best_out is None:
            return {"status": "unknown", "reason": "no_product_model", "topk": out_topk[:5]}

        if best_out["product_conf"] < prod_min or best_out["confidence"] < final_min or best_out["ean"] == "":
            return {"status": "unknown", "reason": "low_conf_or_no_ean_mapping", "topk": out_topk[:5]}

        return {"status": "ok", **best_out, "topk": out_topk[:5]}
