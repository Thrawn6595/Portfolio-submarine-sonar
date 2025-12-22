from pathlib import Path

# Paths - use relative so it works from any directory
DATA_RAW_PATH = "data/raw/sonar.csv"
DATA_INTERIM_PATH = "data/interim"
DATA_PROCESSED_PATH = "data/processed"

# Model config
DEFAULT_SEED = 1206
TEST_SIZE = 0.2
DEFAULT_CV_FOLDS = 5

# Outcome mapping
OUTCOME_MAPPING = {"R": 0, "M": 1}

# Scoring options
SCORING_OPTIONS = {
    "f1": "f1",
    "f2": "f2", 
    "recall": "recall",
    "precision": "precision"
}

# Display settings
MAX_ROWS_DISPLAY = 100
MAX_COLS_DISPLAY = None
FIGURE_SIZE = (12, 6)
PLOT_STYLE = "whitegrid"
