from __future__ import annotations
from pathlib import Path
from ultralytics import YOLO

def train_classifier(data_dir: Path, base_model: str, imgsz: int, epochs: int, batch: int,
                     project_dir: Path, run_name: str) -> Path:
    project_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(base_model)
    model.train(
        data=str(data_dir),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        project=str(project_dir),
        name=run_name,
        verbose=True,
    )
    best = project_dir / run_name / "weights" / "best.pt"
    if not best.exists():
        hits = list(project_dir.rglob("best.pt"))
        if not hits:
            raise RuntimeError("Training completed but best.pt not found.")
        best = hits[0]
    return best
