from __future__ import annotations
import shutil
from src.ean.config import load_config
from src.ean.trainers import train_classifier

def main():
    cfg = load_config()
    t = cfg.training["packaging"]
    best = train_classifier(
        data_dir=cfg.packaging_dataset,
        base_model=t["base_model"],
        imgsz=int(t["imgsz"]),
        epochs=int(t["epochs"]),
        batch=int(t["batch"]),
        project_dir=cfg.packaging_model_out,
        run_name="train"
    )
    cfg.packaging_model_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, cfg.packaging_model_out / "best.pt")
    print("Saved packaging model:", cfg.packaging_model_out / "best.pt")

if __name__ == "__main__":
    main()
