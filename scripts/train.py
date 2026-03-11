from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from bindingdb_activity_prediction.data import load_parquet
from bindingdb_activity_prediction.featurization import smiles_to_morgan
from bindingdb_activity_prediction.models import get_models
from bindingdb_activity_prediction.evaluation import compute_metrics
from bindingdb_activity_prediction.splits import random_split, scaffold_split


DATA_PATH = "data/processed/egfr_activity_dataset.parquet"

MORGAN_RADIUS = 2
MORGAN_NBITS = 2048

SPLIT_TYPE = "random"  # "random" or "random"


def class_counts(y) -> dict:
    values, counts = np.unique(y, return_counts=True)
    return dict(zip(values.tolist(), counts.tolist()))


def main():

    # -------------------------
    # Load dataset
    # -------------------------

    df = load_parquet(DATA_PATH)

    smiles = df["canonical_smiles"].tolist()
    y = df["label"].to_numpy()

    # -------------------------
    # Featurization
    # -------------------------

    X = smiles_to_morgan(
        smiles,
        radius=MORGAN_RADIUS,
        n_bits=MORGAN_NBITS,
    )

    # -------------------------
    # Dataset split
    # -------------------------

    if SPLIT_TYPE == "random":

        X_train, X_test, y_train, y_test = random_split(X, y)

    elif SPLIT_TYPE == "scaffold":

        train_idx, test_idx = scaffold_split(smiles)

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

    else:
        raise ValueError(f"Invalid split type: {SPLIT_TYPE}")


    print("\nDataset summary")
    print("----------------")
    print(f"Total molecules: {len(y)}")
    print(f"Train molecules: {len(y_train)}")
    print(f"Test molecules:  {len(y_test)}")

    print("\nClass balance")
    print("-------------")
    print(f"Full dataset: {class_counts(y)}")
    print(f"Train set:    {class_counts(y_train)}")
    print(f"Test set:     {class_counts(y_test)}")


    # -------------------------
    # Train models
    # -------------------------

    results = []

    models = get_models()

    for name, model in models.items():

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)

        metrics = compute_metrics(y_test, y_pred, y_score)

        row = {"model": name, "split_type": SPLIT_TYPE, **metrics}
        results.append(row)

        cm = confusion_matrix(y_test, y_pred)

        print("\n========================")
        print(name)
        print(metrics)
        print("Confusion matrix [[TN, FP], [FN, TP]]:")
        print(cm)

    results_df = pd.DataFrame(results).sort_values("balanced_accuracy", ascending=False)
    print("\nResults table")
    print(results_df)
    
    output_path = Path(f"results/metrics/egfr_results_{SPLIT_TYPE}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(f"results/metrics/egfr_results_{SPLIT_TYPE}.csv", index=False)


if __name__ == "__main__":
    main()