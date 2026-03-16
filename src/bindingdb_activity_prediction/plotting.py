from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def plot_roc_curves(models, X_train,y_train, X_test, y_test,split_type, output_path):
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
    plt.title(f"ROC curves ({split_type} split)")
    plt.legend()
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved ROC figure to: {output_path}")

def plot_pr_curves(models, X_train,y_train, X_test, y_test, split_type, output_path):
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
    plt.title(f"Precision-Recall curves ({split_type} split)")
    plt.legend()
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved PR figure to: {output_path}")

def plot_chemical_space_pca(X, y, output_path):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Cumulative explained variance:", pca.explained_variance_ratio_.sum())

    plot_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "label": y,
    })

    plt.figure(figsize=(8, 6))

    label_names = {0:"inactive", 1: "active"}

    for label_value in sorted(plot_df["label"].unique()):
        subset = plot_df[plot_df["label"] == label_value]
        plt.scatter(
            subset["PC1"],
            subset["PC2"],
            label=label_names.get(label_value, f"label {label_value}"),
            alpha=0.7,
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Chemical space projection (PCA of Morgan fingerprints)")
    plt.legend()
    plt.tight_layout()

    # 7. save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved figure to: {output_path}")