from collections import defaultdict
from sklearn.model_selection import train_test_split
from rdkit.Chem.Scaffolds import MurckoScaffold


def random_split(X, y, test_size=0.2, random_seed=42):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )


def smiles_to_scaffold(smiles) -> str:
    return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles)


def scaffold_split(
    smiles_list: list[str], 
    test_size: float = 0.2
):
    scaffold_to_indices = defaultdict(list)

    for i, smiles in enumerate(smiles_list):
        scaffold = smiles_to_scaffold(smiles)
        scaffold_to_indices[scaffold].append(i)


    scaffold_groups = sorted(
        scaffold_to_indices.values(),
        key=len,
        reverse=True
    )

    n_total = len(smiles_list)
    n_test_target = int(test_size * n_total)

    train_idx = []
    test_idx = []

    for group in scaffold_groups:
        if len(test_idx) + len(group) <= n_test_target:
            test_idx.extend(group)
        else:
            train_idx.extend(group)

    return train_idx, test_idx