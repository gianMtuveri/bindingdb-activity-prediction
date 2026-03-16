from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_curve,
)

from bindingdb_activity_prediction.data import load_parquet
from bindingdb_activity_prediction.featurization import smiles_to_morgan
from bindingdb_activity_prediction.models import get_models
from bindingdb_activity_prediction.splits import random_split, scaffold_split


DATA_PATH = "data/processed/egfr_activity_dataset.parquet"

MORGAN_RADIUS = 2
MORGAN_NBITS = 2048

SPLIT_TYPE = "random"  # "scaffold" or "random"


def get_split_data(X, y, smiles, split_type="scaffold"):
    if split_type == "random":
        return random_split(X, y, test_size=0.2, random_seed=42)

    if split_type == "scaffold":
        train_idx, test_idx = scaffold_split(smiles, test_size=0.2)

        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        return X_train, X_test, y_train, y_test

    raise ValueError(f"Invalid split type: {split_type}")


def main():
    df = load_parquet(DATA_PATH)

    smiles = df["canonical_smiles"].tolist()
    y = df["label"].to_numpy()

    X = smiles_to_morgan(
        smiles,
        radius=MORGAN_RADIUS,
        n_bits=MORGAN_NBITS,
    )

    X_train, X_test, y_train, y_test = get_split_data(
        X,
        y,
        smiles,
        split_type=SPLIT_TYPE,
    )

    models = get_models()

    roc_output = Path(f"results/figures/egfr_roc_{SPLIT_TYPE}.png")
    pr_output = Path(f"results/figures/egfr_pr_{SPLIT_TYPE}.png")

    roc_output.parent.mkdir(parents=True, exist_ok=True)

    # ROC
    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curves ({SPLIT_TYPE} split)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(roc_output, dpi=300)
    plt.close()

    # PR
    plt.figure(figsize=(8, 6))

    positive_rate = np.mean(y_test)
    plt.plot([0, 1], [positive_rate, positive_rate], linestyle="--", label="baseline")

    for name, model in models.items():
        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)

        precision, recall, _ = precision_recall_curve(y_test, y_score)
        pr_auc = auc(recall, precision)

        plt.plot(recall, precision, label=f"{name} (AUC = {pr_auc:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall curves ({SPLIT_TYPE} split)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(pr_output, dpi=300)
    plt.close()

    print(f"Saved ROC figure to: {roc_output}")
    print(f"Saved PR figure to: {pr_output}")


if __name__ == "__main__":
    main()