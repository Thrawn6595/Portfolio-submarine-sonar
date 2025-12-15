import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from configs.config import (
    DATA_RAW_PATH, OUTCOME_MAPPING, DEFAULT_SEED, TEST_SIZE,
    FIGURE_SIZE, PLOT_STYLE, MAX_ROWS_DISPLAY, MAX_COLS_DISPLAY
)

from ml_toolkit.eda import (
    load_dataset, check_normality, plot_distributions, get_summary_stats,
    correlation_with_target, plot_correlation_heatmap, count_high_correlations
)

pd.set_option("display.max_rows", MAX_ROWS_DISPLAY)
pd.set_option("display.max_columns", MAX_COLS_DISPLAY)
sns.set_style(PLOT_STYLE)
plt.rcParams['figure.figsize'] = FIGURE_SIZE

print("Setup complete\n")

# Load full dataset
df, schema = load_dataset(path=DATA_RAW_PATH, outcome_mapping=OUTCOME_MAPPING)
print(f"Full Dataset: {schema['n_samples']} samples, {schema['n_features']} features")

# Train/Test split - EDA ONLY on training data
y = df['outcome']
X = df.drop('outcome', axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=TEST_SIZE, 
    random_state=DEFAULT_SEED,
    stratify=y
)

# Reconstruct training dataframe for analysis
df_train = X_train.copy()
df_train['outcome'] = y_train

print(f"Training Set: {len(df_train)} samples ({(1-TEST_SIZE)*100:.0f}%)")
print(f"Test Set: {len(X_test)} samples ({TEST_SIZE*100:.0f}%)")
print("\nEDA performed on TRAINING data only\n")

# Class distribution (training only)
class_counts = df_train['outcome'].value_counts()
baseline_acc = class_counts.max() / class_counts.sum()
print("Training Class Distribution:")
print(class_counts)
print(f"Baseline Accuracy: {baseline_acc:.2%}\n")

# Summary statistics (training only)
summary = get_summary_stats(df_train, schema['feature_names'])
print("Summary Stats (first 10 features):")
print(summary.head(10))
print()

# Correlations (training only)
target_corr = correlation_with_target(df_train, target='outcome')
print("Top 15 features by correlation with target:")
print(target_corr.head(15))
print()

# Check for high multicollinearity
features_only = df_train.drop('outcome', axis=1)
corr_matrix = features_only.corr()
high_corr = count_high_correlations(corr_matrix, threshold=0.85)
print(f"Feature pairs with |correlation| > 0.85: {len(high_corr)}")
if len(high_corr) > 0:
    print("Top 10 highly correlated pairs:")
    print(high_corr.head(10))
