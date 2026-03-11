from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem


INPUT_PARQUET = "data/raw/cleaned_agg.parquet"
OUTPUT_PARQUET = "data/processed/egfr_activity_dataset.parquet"

TARGET_NAME = "Epidermal growth factor receptor"

ACTIVE_THRESHOLD_NM = 1000.0
INACTIVE_THRESHOLD_NM = 10000.0


def load_parquet(path: str | Path) -> pd.DataFrame:
    table = pq.read_table(path)
    return table.to_pandas()


def canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def assign_label(
    affinity_nM: float,
    active_threshold_nM: float = ACTIVE_THRESHOLD_NM,
    inactive_threshold_nM: float = INACTIVE_THRESHOLD_NM,
) -> int | None:
    if pd.isna(affinity_nM):
        return None
    if affinity_nM <= active_threshold_nM:
        return 1
    if affinity_nM >= inactive_threshold_nM:
        return 0
    return None


def main() -> None:
    df = load_parquet(INPUT_PARQUET)

    # Keep only EGFR
    df = df[df["Target Name"] == TARGET_NAME].copy()

    # Canonicalize SMILES
    df["canonical_smiles"] = df["Ligand SMILES"].map(canonicalize_smiles)
    df = df[df["canonical_smiles"].notna()].copy()

    # Deduplicate by molecule
    # Version 1 rule: keep the strongest measurement (lowest aff_nM_median)
    df = (
        df.sort_values("aff_nM_median", ascending=True)
        .drop_duplicates(subset=["canonical_smiles"], keep="first")
        .reset_index(drop=True)
    )

    # Assign labels
    df["label"] = df["aff_nM_median"].map(assign_label)
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)

    # Keep a compact modeling dataset
    keep_cols = [
        "Target Name",
        "Ligand SMILES",
        "canonical_smiles",
        "affinity_type",
        "aff_nM_median",
        "DG_median",
        "n_meas",
        "label",
    ]
    df = df[keep_cols].reset_index(drop=True)

    output_path = Path(OUTPUT_PARQUET)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"Saved dataset to: {output_path}")
    print(f"Rows: {len(df)}")
    print("Class balance:")
    print(df["label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()