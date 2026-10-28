#!/usr/bin/env python3
"""Generate and export preprocessed dataset CSVs for indoor Wi-Fi positioning.

This script:
1. Ensures the raw UJIIndoorLoc dataset is present (downloads and extracts from UCI if missing).
2. Applies the full recommended feature pipeline (imputed sentinel, missingness indicators, aggregate features).
3. Saves separated CSVs into data/processed/ (X_train, y_train, meta_train, X_val, y_val, meta_val, feature_dictionary).
4. Saves combined CSVs directly into data/ (preprocessed_train.csv, preprocessed_val.csv) for direct model consumption.
5. Creates a compressed zip archive (data/preprocessed_dataset.zip) for easy transfer.
"""

import os
import shutil
import sys
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from src.data_loader import (
    DATA_DIR,
    META_COLS,
    TARGET_COLS,
    WAP_COLS,
    ensure_dataset,
    load_train,
    load_validation,
)
from src.preprocessing import ABLATION_CONFIGS, get_feature_dictionary, prepare_features


def main():
    print("=" * 70)
    print("PREPROCESSED DATASET GENERATOR")
    print("=" * 70)

    # 1. Ensure raw dataset is present
    print("\n[1/5] Checking raw dataset...")
    ensure_dataset(DATA_DIR)
    train_df = load_train()
    val_df = load_validation()
    print(f"  Raw Train: {train_df.shape}")
    print(f"  Raw Val:   {val_df.shape}")

    # 2. Run preprocessing pipeline
    print("\n[2/5] Running preprocessing pipeline (imputed + indicators + aggregates)...")
    X_tr_full, X_va_full, y_tr_full, y_va_full, meta_tr_full, meta_va_full, pp_full = prepare_features(
        train_df=train_df,
        val_df=val_df,
        wap_cols=WAP_COLS,
        target_cols=TARGET_COLS,
        meta_cols=META_COLS,
        config=ABLATION_CONFIGS["imputed+ind+agg"],
    )
    print(f"  Clean Train features: {X_tr_full.shape}")
    print(f"  Clean Val features:   {X_va_full.shape}")

    # 3. Save separated CSVs to data/processed/
    print("\n[3/5] Saving separated feature/target CSVs to data/processed/...")
    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    X_tr_full.to_csv(processed_dir / "X_train.csv", index=False)
    y_tr_full.to_csv(processed_dir / "y_train.csv", index=False)
    meta_tr_full.to_csv(processed_dir / "meta_train.csv", index=False)
    X_va_full.to_csv(processed_dir / "X_val.csv", index=False)
    y_va_full.to_csv(processed_dir / "y_val.csv", index=False)
    meta_va_full.to_csv(processed_dir / "meta_val.csv", index=False)
    get_feature_dictionary().to_csv(processed_dir / "feature_dictionary.csv", index=False)
    print("  Saved X_train.csv, y_train.csv, meta_train.csv, X_val.csv, y_val.csv, meta_val.csv, feature_dictionary.csv")

    # 4. Save combined CSVs to data/
    print("\n[4/5] Saving combined CSVs to data/ (features + targets)...")
    train_combined = pd.concat([X_tr_full, y_tr_full], axis=1)
    val_combined = pd.concat([X_va_full, y_va_full], axis=1)
    train_combined.to_csv(DATA_DIR / "preprocessed_train.csv", index=False)
    val_combined.to_csv(DATA_DIR / "preprocessed_val.csv", index=False)
    print(f"  Saved data/preprocessed_train.csv: {train_combined.shape}")
    print(f"  Saved data/preprocessed_val.csv:   {val_combined.shape}")

    # 5. Create compressed zip archive
    print("\n[5/5] Creating compressed archive for easy download...")
    zip_path = shutil.make_archive(str(DATA_DIR / "preprocessed_dataset"), "zip", str(processed_dir))
    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"  Saved {zip_path} ({zip_size_mb:.1f} MB)")

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE! All files are in place.")
    print("=" * 70)


if __name__ == "__main__":
    main()
