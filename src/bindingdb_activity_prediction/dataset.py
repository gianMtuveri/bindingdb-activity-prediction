import pandas as pd
from rdkit import Chem

def canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def assign_label(
    affinity_nM: float,
    active_threshold_nM: float = 1000.0,
    inactive_threshold_nM: float = 10000.0,
) -> int | None:
    if pd.isna(affinity_nM):
        return None
    if affinity_nM <= active_threshold_nM:
        return 1
    if affinity_nM >= inactive_threshold_nM:
        return 0
    return None