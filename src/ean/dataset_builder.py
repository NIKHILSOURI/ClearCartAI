from __future__ import annotations
import os
import shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from .io import read_jsonl

def _link_or_copy(src: Path, dst: Path, symlink: bool = True) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if symlink:
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)

def _split_df(df: pd.DataFrame, group_col: str | None, val_ratio: float = 0.12, seed: int = 42):
    if group_col and group_col in df.columns and df[group_col].notna().any():
        gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        train_idx, val_idx = next(gss.split(df, groups=df[group_col]))
        return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()
    df = df.sample(frac=1.0, random_state=seed)
    n_val = int(len(df) * val_ratio)
    return df.iloc[n_val:].copy(), df.iloc[:n_val].copy()

def clear_dir(d: Path):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

def build_from_records(records_path: Path,
                       packaging_dataset: Path,
                       products_dataset: Path,
                       products_dataset_by_pack: Path,
                       packaging_classes: list[str],
                       symlink: bool = True,
                       val_ratio: float = 0.12) -> None:
    rows = read_jsonl(records_path)
    if not rows:
        raise RuntimeError(f"No records at {records_path}. Run labeling UI first.")
    df = pd.DataFrame(rows)

    df = df[df["crop_path"].notna() & df["packaging"].notna()].copy()
    df = df[df["packaging"].isin(packaging_classes)].copy()

    df_prod = df[df["product_name"].notna() & (df["product_name"].astype(str).str.len() > 0)].copy()
    group_col = "session_id" if "session_id" in df.columns else None

    tr, va = _split_df(df, group_col, val_ratio)
    for split, sdf in [("train", tr), ("val", va)]:
        for _, r in sdf.iterrows():
            src = Path(r["crop_path"])
            dst = packaging_dataset / split / str(r["packaging"]) / src.name
            _link_or_copy(src, dst, symlink)

    tr, va = _split_df(df_prod, group_col, val_ratio)
    for split, sdf in [("train", tr), ("val", va)]:
        for _, r in sdf.iterrows():
            src = Path(r["crop_path"])
            cls = str(r["product_name"]).strip()
            dst = products_dataset / split / cls / src.name
            _link_or_copy(src, dst, symlink)

    for packaging in sorted(df_prod["packaging"].unique()):
        sub = df_prod[df_prod["packaging"] == packaging].copy()
        if len(sub) < 10:
            continue
        tr, va = _split_df(sub, group_col, val_ratio)
        base = products_dataset_by_pack / packaging
        for split, sdf in [("train", tr), ("val", va)]:
            for _, r in sdf.iterrows():
                src = Path(r["crop_path"])
                cls = str(r["product_name"]).strip()
                dst = base / split / cls / src.name
                _link_or_copy(src, dst, symlink)
