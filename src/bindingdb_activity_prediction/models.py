from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def get_models(random_seed: int = 42) -> dict:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, 
            random_state=random_seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_seed,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=random_seed
        ),
    }