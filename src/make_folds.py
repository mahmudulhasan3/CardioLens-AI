import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(".")
DATA_DIR = ROOT / "data"
IMG_EXTS = {".jpg", ".jpeg", ".png"}
N_FOLDS = 5
SEED = 42  # fixed seed -> the same folds every time


def folder_to_label(folder_name):
    """Map a raw folder name to a short label (order matters)."""
    name = folder_name.lower()
    if "history" in name:
        return "History_of_MI"
    if "abnormal" in name:
        return "Abnormal"
    if "normal person" in name:
        return "Normal"
    if "myocardial" in name:
        return "MI"
    return None  # ignore any other folder


def main():
    # 1) Scan all class folders and hash every image
    rows = []
    for folder in sorted(DATA_DIR.iterdir()):
        if not folder.is_dir():
            continue
        label = folder_to_label(folder.name)
        if label is None:
            continue
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                rows.append(
                    {
                        "path": p.as_posix(),  # relative to project root
                        "label": label,
                        "hash": hashlib.md5(p.read_bytes()).hexdigest(),
                    }
                )
    raw_df = pd.DataFrame(rows)
    print("raw images:", len(raw_df))

    # 2) Safety check: the same image must never appear under two classes
    shared = (raw_df.groupby("hash")["label"].nunique() > 1).sum()
    assert shared == 0, f"{shared} hashes appear in more than one class!"

    # 3) Remove exact duplicates (keep the first file for each hash)
    clean_df = raw_df.drop_duplicates(subset="hash", keep="first").reset_index(
        drop=True
    )
    clean_df.to_csv(DATA_DIR / "clean_index.csv", index=False)
    print("clean images:", len(clean_df))
    print(clean_df["label"].value_counts().to_string())

    # 4) Stratified 5-fold split (each fold keeps the same class ratio)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    clean_df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(clean_df, clean_df["label"])):
        clean_df.loc[val_idx, "fold"] = fold
    clean_df.to_csv(DATA_DIR / "folds.csv", index=False)

    print("\nImages per class per fold:")
    print(pd.crosstab(clean_df["label"], clean_df["fold"]))


if __name__ == "__main__":
    main()
