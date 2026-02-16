from __future__ import annotations
from src.ean.config import load_config
from src.ean.dataset_builder import build_from_records, clear_dir

def main():
    cfg = load_config()
    clear_dir(cfg.packaging_dataset)
    clear_dir(cfg.products_dataset)
    clear_dir(cfg.products_dataset_by_pack)

    build_from_records(
        records_path=cfg.records_path,
        packaging_dataset=cfg.packaging_dataset,
        products_dataset=cfg.products_dataset,
        products_dataset_by_pack=cfg.products_dataset_by_pack,
        packaging_classes=cfg.packaging_classes,
        symlink=True,
        val_ratio=0.12
    )
    print("Built datasets:")
    print(" -", cfg.packaging_dataset)
    print(" -", cfg.products_dataset)
    print(" -", cfg.products_dataset_by_pack)

if __name__ == "__main__":
    main()
