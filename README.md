# BindingDB Molecular Activity Prediction Benchmark

A reproducible machine learning pipeline for predicting molecular activity from chemical structure using curated **BindingDB** data.

This project demonstrates an end-to-end cheminformatics ML workflow including:

- dataset preparation
- molecular featurization
- baseline model benchmarking
- realistic evaluation using scaffold splits

The goal is **not to chase the most complex model**, but to build a **clean and scientifically sound benchmark pipeline**.

---

# Motivation

Predicting molecular activity from chemical structure is a central problem in drug discovery and cheminformatics.

However, many simple ML benchmarks produce **over-optimistic results** because they rely on **random train/test splits**, which allow highly similar molecules to appear in both sets.

This project explicitly compares:

- **Random split** (naïve evaluation)
- **Bemis–Murcko scaffold split** (chemically realistic evaluation)

Scaffold splitting prevents chemical series from leaking between train and test sets, providing a more realistic estimate of model generalization.

---

# Dataset

The dataset is derived from **BindingDB** and processed using the companion project:

https://github.com/gianMtuveri/binding_db_refiner

The preprocessing pipeline:

1. Select affinity measurements with priority  
   `Ki → Kd → IC50`

2. Convert values to numeric nM

3. Remove invalid entries

4. Compute binding free energy ΔG

5. Aggregate repeated measurements

For this benchmark we focus on a single target:

**EGFR — Epidermal Growth Factor Receptor**

The dataset is converted into a binary activity classification task.

---

# Project Overview

The project implements a reproducible ML pipeline:

dataset → cleaning → featurization → baseline models → evaluation

Pipeline steps:

1. Load processed BindingDB dataset
2. Extract canonical SMILES
3. Generate molecular fingerprints
4. Split dataset
5. Train baseline models
6. Evaluate predictions

---

# Molecular Representation

Molecules are represented using **Morgan fingerprints** (circular fingerprints).

Parameters:

- radius = 2
- fingerprint size = 2048 bits

Implementation uses **RDKit**.

---

# Models

Three baseline models are included:

- Logistic Regression
- Random Forest
- Gradient Boosting

These models were chosen because they provide strong classical baselines and are widely used in cheminformatics.

Deep learning models are intentionally **not included in the first version** to maintain a clear baseline benchmark.

---

# Evaluation Metrics

Model performance is evaluated using:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC

Balanced accuracy and PR-AUC are particularly important due to potential class imbalance.

Confusion matrices are also printed to inspect prediction behavior.

---

# Data Splitting Strategies

Two evaluation strategies are implemented.

## Random Split

Standard stratified random split using:

train_test_split(..., stratify=y)

This often produces optimistic performance because chemically similar molecules may appear in both train and test sets.

---

## Scaffold Split

Scaffold splitting groups molecules by their **Bemis–Murcko scaffold** and assigns entire scaffold families to either train or test sets.

This prevents chemical series leakage and simulates a more realistic scenario where models must generalize to **new chemotypes**.

Scaffolds are computed using:

rdkit.Chem.Scaffolds.MurckoScaffold

---

# Project Structure

```
src/
    bindingdb_activity_prediction/
        data.py
        featurization.py
        models.py
        evaluation.py
        splits.py

scripts/
    train.py

data/
    processed/

results/
    metrics/
```

### src/

Contains reusable pipeline components:

- dataset loading
- fingerprint generation
- model definitions
- evaluation metrics
- splitting strategies

### scripts/

Contains executable experiment scripts.

### results/

Stores benchmark outputs.

---

# Running the Benchmark

Clone the repository and install dependencies:

```
pip install -e .
```

Then run:

```
python scripts/train.py
```

Inside the script you can choose the split type:

```
SPLIT_TYPE = "random"
SPLIT_TYPE = "scaffold"
```

Results are saved to:

```
results/metrics/
```

Example outputs:

```
egfr_results_random.csv
egfr_results_scaffold.csv
```

---

# Example Output

Each run produces a metrics table:

| model | split_type | accuracy | balanced_accuracy | roc_auc | pr_auc |
|------|-------------|----------|-------------------|--------|--------|
| logistic_regression | scaffold | ... | ... | ... | ... |
| random_forest | scaffold | ... | ... | ... | ... |
| gradient_boosting | scaffold | ... | ... | ... | ... |

---

# Key Observation

Random splits typically produce higher scores because similar molecules appear in both training and test sets.

Scaffold splits provide a **more challenging and realistic evaluation**, often resulting in lower but more meaningful performance estimates.

---

# Limitations

This project intentionally focuses on simplicity and reproducibility, which introduces several limitations.

### Single target

The benchmark currently uses only **EGFR**.  
Results may not generalize to other targets.

### Single molecular representation

Only Morgan fingerprints are used.  
Other descriptors or learned representations may improve performance.

### No hyperparameter tuning

Models use mostly default parameters.  
Performance could improve with systematic optimization.

### Binary activity classification

The dataset is simplified to a binary task.  
Real drug discovery problems often require regression or multi-task models.

### No uncertainty estimation

Predictions currently do not include uncertainty or confidence intervals.

### Dataset bias

BindingDB contains experimental bias and uneven chemical space coverage.

---

# Future Work

Potential extensions include:

### Dataset expansion

- additional targets
- multi-target prediction
- multi-task learning

### Alternative representations

- RDKit descriptors
- physicochemical descriptors
- learned graph embeddings

### Deep learning models

- graph neural networks
- message passing neural networks

### Improved evaluation

- scaffold distribution analysis
- cluster split strategies
- time-based splits if timestamps are available

### Model calibration

- probability calibration
- uncertainty estimation
- bootstrap confidence intervals

### Visualization

- scaffold size distributions
- chemical space projections
- prediction error analysis

### Deployment

- model inference API
- web interface for molecular prediction

---

# Related Project

Dataset preprocessing:

BindingDB Refiner

https://github.com/gianMtuveri/binding_db_refiner

---

# License

MIT License

---

# Author

Gian Marco Tuveri  
Computational Biophysics — Molecular Modeling & Machine Learning
