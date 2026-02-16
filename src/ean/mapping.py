from __future__ import annotations
from pathlib import Path
import pandas as pd

def load_product_to_ean(csv_path: Path) -> dict[str, str]:
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    if "product_name" not in df.columns or "ean" not in df.columns:
        raise ValueError("product_to_ean.csv must have columns: product_name,ean")
    return dict(zip(df["product_name"].astype(str), df["ean"].astype(str)))
