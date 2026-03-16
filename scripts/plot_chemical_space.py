from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

from bindingdb_activity_prediction.data import load_parquet
from bindingdb_activity_prediction.featurization import smiles_to_morgan


DATA_PATH = "data/processed/egfr_activity_dataset.parquet"
OUTPUT_PATH = "results/figures/chemical_space_pca.png"

MORGAN_RADIUS = 2
MORGAN_NBITS = 2048


def main():
    # 1. load dataset
    df = load_parquet(DATA_PATH)

    # 2. get smiles and labels
    smiles = df["canonical_smiles"].tolist()
    y = df["label"].to_numpy()

    # 3. featurize
    X = smiles_to_morgan(
        smiles,
        radius=MORGAN_RADIUS,
        n_bits=MORGAN_NBITS,
    )

    # 4. PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Cumulative explained variance:", pca.explained_variance_ratio_.sum())

    # 5. build plotting dataframe
    plot_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "label": y,
    })

    # 6. scatter plot
    plt.figure(figsize=(8, 6))

    for label_value in sorted(plot_df["label"].unique()):
        subset = plot_df[plot_df["label"] == label_value]
        plt.scatter(
            subset["PC1"],
            subset["PC2"],
            label=f"label {label_value}",
            alpha=0.7,
        )

    label_names = {0:"inactive", 1: "active"}

    label = label_names.get(label_value, f"label {label_value}")

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Chemical space projection (PCA of Morgan fingerprints)")
    plt.legend()
    plt.tight_layout()

    # 7. save
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()