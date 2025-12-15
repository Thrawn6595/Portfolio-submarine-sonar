import pathlib

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"

DEFAULT_SEED = 1206
DEFAULT_CV_FOLDS = 5

SONAR_OUTCOME_COL = "outcome"
DEFAULT_SCORING = "f2"
DEFAULT_RANK_BY = "cv_f2_mean"

SCORING_OPTIONS = {
    "f1": "f1",
    "f2": "f2",
    "recall": "recall",
    "precision": "precision",
}

OUTCOME_MAPPING = {"R": 0, "M": 1}
