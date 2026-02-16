from __future__ import annotations
import pandas as pd
from src.ean.config import load_config
from src.ean.io import read_jsonl, ensure_dir

cfg = load_config()

def main():
    rows = read_jsonl(cfg.records_path)
    if not rows:
        print("No records.jsonl found. Run labeling UI first.")
        return
    df = pd.DataFrame(rows)
    df = df[df["product_name"].notna() & (df["product_name"].astype(str).str.len() > 0)].copy()
    products = sorted(df["product_name"].astype(str).unique().tolist())
    out = pd.DataFrame({"product_name": products, "ean": [""] * len(products)})
    ensure_dir(cfg.product_to_ean.parent)
    out.to_csv(cfg.product_to_ean, index=False)
    print(f"Wrote template: {cfg.product_to_ean} ({len(products)} products). Fill EANs externally.")

if __name__ == "__main__":
    main()
