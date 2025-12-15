"""
Configuration for Sonar Mine vs Rock Classification
"""

# Reproducibility
DEFAULT_SEED = 1206

# Data Paths
DATA_RAW_PATH = "data/raw/sonar.csv"
DATA_PROCESSED_PATH = "data/processed/"

# Data Loading
COLUMN_PREFIX = "feature"
TARGET_COLUMN_INDEX = -1
OUTCOME_MAPPING = {"R": 0, "M": 1}

# Train/Test Split
TEST_SIZE = 0.2
CV_FOLDS = 5

# Display Settings
FIGURE_SIZE = (12, 6)
PLOT_STYLE = "whitegrid"
MAX_ROWS_DISPLAY = 100
MAX_COLS_DISPLAY = None
