from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class Config:
    raw_dir: Path
    records_path: Path
    masks_dir: Path
    crops_dir: Path
    packaging_dataset: Path
    products_dataset: Path
    products_dataset_by_pack: Path
    packaging_model_out: Path
    product_models_out: Path
    product_to_ean: Path
    packaging_classes: list[str]
    thresholds: dict
    sam_model: str
    training: dict
    ui: dict

def load_config(path: str | Path = "configs/system.yaml") -> Config:
    p = Path(path)
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    paths = cfg["paths"]
    def P(x): return Path(x)
    return Config(
        raw_dir=P(paths["raw_dir"]),
        records_path=P(paths["records_path"]),
        masks_dir=P(paths["masks_dir"]),
        crops_dir=P(paths["crops_dir"]),
        packaging_dataset=P(paths["packaging_dataset"]),
        products_dataset=P(paths["products_dataset"]),
        products_dataset_by_pack=P(paths["products_dataset_by_pack"]),
        packaging_model_out=P(paths["packaging_model_out"]),
        product_models_out=P(paths["product_models_out"]),
        product_to_ean=P(paths["product_to_ean"]),
        packaging_classes=list(cfg["packaging_classes"]),
        thresholds=dict(cfg["thresholds"]),
        sam_model=str(cfg["sam"]["model"]),
        training=dict(cfg["training"]),
        ui=dict(cfg["ui"]),
    )
