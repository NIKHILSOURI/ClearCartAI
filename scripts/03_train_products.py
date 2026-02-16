from __future__ import annotations
import argparse, shutil
from src.ean.config import load_config
from src.ean.trainers import train_classifier

def train_global(cfg):
    t = cfg.training["products"]
    out_dir = cfg.product_models_out / "global"
    best = train_classifier(cfg.products_dataset, t["base_model"], int(t["imgsz"]), int(t["epochs"]), int(t["batch"]), out_dir, "train")
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out_dir / "best.pt")
    print("Saved global product model:", out_dir / "best.pt")

def train_by_pack(cfg):
    t = cfg.training["products"]
    base = cfg.products_dataset_by_pack
    out_base = cfg.product_models_out / "by_pack"
    out_base.mkdir(parents=True, exist_ok=True)
    for pack_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        if not (pack_dir / "train").exists():
            continue
        packaging = pack_dir.name
        out_dir = out_base / packaging
        best = train_classifier(pack_dir, t["base_model"], int(t["imgsz"]), int(t["epochs"]), int(t["batch"]), out_dir, "train")
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, out_dir / "best.pt")
        print("Saved product model for", packaging, ":", out_dir / "best.pt")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["global","by_pack"], default="by_pack")
    args = ap.parse_args()
    cfg = load_config()
    if args.mode == "global":
        train_global(cfg)
    else:
        train_by_pack(cfg)

if __name__ == "__main__":
    main()
