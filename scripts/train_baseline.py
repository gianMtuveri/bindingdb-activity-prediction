from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


INPUT_PARQUET = "data/processed/egfr_activity_dataset.parquet"
METRICS_OUT = "results/metrics/egfr_logreg_morgan.json"
FIGURE_OUT = "results/figures/egfr_class_balance.png"

RANDOM_SEED = 42
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048


def load_parquet(path: str | Path):
    table = pq.read_table(path)
    return table.to_pandas()


def smiles_to_morgan(
    smiles_list: list[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
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
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def save_metrics(metrics: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def plot_class_balance(y: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    values, counts = np.unique(y, return_counts=True)

    plt.figure(figsize=(6, 4))
    plt.bar(values.astype(str), counts)
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.title("EGFR class balance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    df = load_parquet(INPUT_PARQUET)

    smiles = df["canonical_smiles"].tolist()
    y = df["label"].to_numpy()

    X = smiles_to_morgan(
        smiles,
        radius=MORGAN_RADIUS,
        n_bits=MORGAN_NBITS,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_score)
    metrics["model"] = "logistic_regression"
    metrics["features"] = "morgan"
    metrics["target"] = "Epidermal growth factor receptor"
    metrics["n_total"] = int(len(y))
    metrics["n_train"] = int(len(y_train))
    metrics["n_test"] = int(len(y_test))

    save_metrics(metrics, METRICS_OUT)
    plot_class_balance(y, FIGURE_OUT)

    print("Training complete.")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()