import pathlib

with open(pathlib.Path(__file__).parent / "VERSION") as f:
    __version__ = f.read().strip()

from .processing.data_manager import load_sonar_raw, split_train_test
from .processing.preprocessing import prepare_data_for_algorithm
from .modeling import get_model_registry
from .train_utils import train_model_with_cv, create_cv_results_table, evaluate_binary_classifier
