import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


def validate_columns(df, required) -> None:
    for col in required:
        if col not in df.columns:
            raise ValueError(f"The required column {col} is not present")

'''def main() -> None:
    input_parquet = "data/processed/egfr_activity_dataset.parquet"

    df = load_parquet(input_parquet)

    print(f"Loaded dataset with {len(df)} rows")

    required = ["canonical_smiles", "label"]

    validate_columns(df, required)'''
