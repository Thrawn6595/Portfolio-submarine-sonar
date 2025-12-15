from typing import Dict, Tuple
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from .config import DEFAULT_SEED

def _scaled(model: BaseEstimator) -> Pipeline:
    return Pipeline(steps=[("scaler", StandardScaler()), ("clf", model)])

def get_model_registry(seed: int = DEFAULT_SEED) -> Dict[str, Tuple[BaseEstimator, dict]]:
    return {
        "logistic_regression": (
            _scaled(LogisticRegression(max_iter=4000, random_state=seed)),
            {"clf__C": [0.1, 1.0, 10.0]}
        ),
        "ridge": (
            _scaled(RidgeClassifier(random_state=seed)),
            {"clf__alpha": [0.1, 1.0, 10.0]}
        ),
        "lda": (
            _scaled(LinearDiscriminantAnalysis()),
            {"clf__solver": ["svd"]}
        ),
        "naive_bayes": (
            _scaled(GaussianNB()),
            {"clf__var_smoothing": [1e-9, 1e-8]}
        ),
        "knn": (
            _scaled(KNeighborsClassifier()),
            {"clf__n_neighbors": [3, 5, 7], "clf__weights": ["uniform", "distance"]}
        ),
        "svm": (
            _scaled(SVC(probability=True, random_state=seed)),
            {"clf__C": [0.5, 1.0, 2.0], "clf__kernel": ["linear", "rbf"]}
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=100, random_state=seed),
            {"max_depth": [None, 5, 10], "min_samples_leaf": [1, 5]}
        ),
    }
