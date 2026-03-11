import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


def smiles_to_morgan(smiles_list: list[str], 
    radius: int = 2, 
    n_bits: int = 2048
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


'''def main() -> None:
    input_parquet = "data/processed/egfr_activity_dataset.parquet"
    morgan_radius = 2
    morgan_nbits = 2048

    df = load_parquet(input_parquet)

    smiles = df["canonical_smiles"].tolist()

    X = smiles_to_morgan(smiles, radius=morgan_radius, n_bits=morgan_nbits)

    print("Featurization ended correctly")

if __name__ == "__main__":
    main()'''