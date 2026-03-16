from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


INPUT_PARQUET = "data/processed/egfr_activity_dataset.parquet"
OUTPUT_CSV = "results/metrics/egfr_model_comparison.csv"
RANDOM_SEED = 42
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048


def load_parquet(path: str | Path):
    return pq.read_table(path).to_pandas()


def smiles_to_morgan(smiles_list: list[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
    )

    features = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES during featurization: {smiles}")

        fp = generator.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        features.append(arr)

    return np.asarray(features)


def compute_metrics(y_true, y_pred, y_score) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }


def get_models():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, 
            random_state=RANDOM_SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=RANDOM_SEED
        ),
    }


def main() -> None:
    df = load_parquet(INPUT_PARQUET)

    smiles = df["canonical_smiles"].tolist()
    y = df["label"].to_numpy()
    X = smiles_to_morgan(smiles, radius=MORGAN_RADIUS, n_bits=MORGAN_NBITS)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    rows = []

    for model_name, model in get_models().items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)

        metrics = compute_metrics(y_test, y_pred, y_score)
        metrics["model"] = model_name
        rows.append(metrics)

    results = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)

    output_path = Path(OUTPUT_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print(results.to_string(index=False))
    print(f"\nSaved comparison table to: {output_path}")


if __name__ == "__main__":
    main()