import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from bindingdb_activity_prediction.data import load_parquet
from bindingdb_activity_prediction.splits import smiles_to_scaffold


DATA_PATH = "data/processed/egfr_activity_dataset.parquet"


def main():

    df = load_parquet(DATA_PATH)
    smiles = df["canonical_smiles"].tolist()

    # Compute scaffolds
    scaffolds = [smiles_to_scaffold(s) for s in smiles]

    df_scaffold = pd.DataFrame({
        "smiles": smiles,
        "scaffold": scaffolds
    })

    # Count molecules per scaffold
    scaffold_counts = (
        df_scaffold
        .groupby("scaffold")
        .size()
        .sort_values(ascending=False)
    )

    mean_scaffold_size = scaffold_counts.mean()
    median_scaffold_size = scaffold_counts.median()
    largest_scaffold_fraction = scaffold_counts.iloc[0] / len(smiles)

    print("\nDataset statistics")
    print("------------------")
    print(f"Total molecules: {len(smiles)}")
    print(f"Unique scaffolds: {len(scaffold_counts)}")

    print("\nTop 10 scaffold families")
    print("------------------------")
    print(scaffold_counts.head(10))

    print(f"Average scaffold size: {mean_scaffold_size:.2f}")
    print(f"Median scaffold size: {median_scaffold_size:.2f}")
    print(f"Largest scaffold fraction: {largest_scaffold_fraction:.3f}")

    # Plot scaffold size distribution
    plt.figure(figsize=(8,5))
    plt.hist(scaffold_counts.values, bins=50)
    plt.xlabel("Molecules per scaffold")
    plt.ylabel("Number of scaffolds")
    plt.title("Scaffold size distribution")
    plt.tight_layout()
    
    figure_path = Path("results/figures/scaffold_distribution.png")
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(figure_path, dpi=300)
    print(f"\nSaved figure to: {figure_path}")

    table_path = Path("results/metrics/top_scaffolds.csv")
    scaffold_counts.head(10).to_csv(table_path, header=["count"])
    print(f"Saved top scaffold table to: {table_path}")


if __name__ == "__main__":
    main()