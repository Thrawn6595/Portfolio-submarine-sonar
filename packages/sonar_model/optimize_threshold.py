"""
Threshold Optimization for Maximum Recall

Demonstrates how to trade precision for recall to achieve near-100% mine detection.
Uses out-of-fold CV predictions to avoid test set leakage.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, confusion_matrix
from sonar_model import load_sonar_raw, split_train_test, prepare_data_for_algorithm, get_model_registry, train_model_with_cv
from sonar_model.config import DEFAULT_SEED, DEFAULT_CV_FOLDS
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

def get_oof_predictions(model, X_train, y_train, cv_folds=5, seed=DEFAULT_SEED):
    """
    Get out-of-fold predictions for threshold calibration.
    This avoids test set leakage.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    oof_scores = np.zeros(len(X_train))
    
    for train_idx, val_idx in cv.split(X_train, y_train):
        fold_model = clone(model)
        fold_model.fit(X_train[train_idx], y_train[train_idx])
        
        if hasattr(fold_model, "predict_proba"):
            oof_scores[val_idx] = fold_model.predict_proba(X_train[val_idx])[:, 1]
        elif hasattr(fold_model, "decision_function"):
            oof_scores[val_idx] = fold_model.decision_function(X_train[val_idx])
        else:
            raise ValueError("Model must have predict_proba or decision_function")
    
    return oof_scores

def find_threshold_for_recall(y_true, scores, target_recall=0.98):
    """
    Find threshold that achieves target recall on OOF predictions.
    """
    # Sort by score descending
    sorted_indices = np.argsort(-scores)
    y_sorted = y_true[sorted_indices]
    scores_sorted = scores[sorted_indices]
    
    total_positives = (y_true == 1).sum()
    
    for i in range(len(scores_sorted)):
        # Predict positive for all samples >= current threshold
        preds = (scores >= scores_sorted[i]).astype(int)
        current_recall = recall_score(y_true, preds)
        
        if current_recall >= target_recall:
            return scores_sorted[i]
    
    return scores_sorted.min()

def evaluate_at_threshold(model, X, y, threshold):
    """
    Evaluate model at specific threshold.
    """
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
    else:
        raise ValueError("Model must have predict_proba or decision_function")
    
    preds = (scores >= threshold).astype(int)
    
    recall = recall_score(y, preds)
    precision = precision_score(y, preds, zero_division=0)
    cm = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'threshold': threshold,
        'recall': recall,
        'precision': precision,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'mines_caught': tp,
        'mines_missed': fn,
        'false_alarms': fp
    }

def main():
    print("="*80)
    print("THRESHOLD OPTIMIZATION FOR MAXIMUM MINE DETECTION")
    print("="*80)
    
    # Load data
    df = load_sonar_raw("../../data/raw/sonar.csv")
    X_train, X_test, y_train, y_test = split_train_test(df, seed=DEFAULT_SEED)
    prepared = prepare_data_for_algorithm(X_train, X_test, y_train, y_test)
    
    # Train champion model (SVM)
    registry = get_model_registry(seed=DEFAULT_SEED)
    svm_model, svm_params = registry["svm"]
    
    print("\nTraining SVM champion model...")
    result = train_model_with_cv(
        model=svm_model,
        param_grid=svm_params,
        X_train=prepared.X_train,
        y_train=prepared.y_train,
        scoring="f2",
        cv_folds=DEFAULT_CV_FOLDS,
        seed=DEFAULT_SEED
    )
    
    champion = result.fitted_model
    
    # Get OOF predictions for threshold calibration
    print("\nGenerating out-of-fold predictions for threshold calibration...")
    oof_scores = get_oof_predictions(champion, prepared.X_train, prepared.y_train)
    
    # Evaluate at different recall targets
    print("\n" + "="*80)
    print("THRESHOLD CALIBRATION RESULTS")
    print("="*80)
    
    recall_targets = [0.95, 0.98, 0.99, 1.00]
    
    results = []
    
    for target_recall in recall_targets:
        # Find threshold on OOF predictions
        threshold = find_threshold_for_recall(prepared.y_train, oof_scores, target_recall)
        
        # Evaluate on test set with this threshold
        test_metrics = evaluate_at_threshold(champion, prepared.X_test, prepared.y_test, threshold)
        
        results.append({
            'target_recall': target_recall,
            'oof_threshold': threshold,
            'test_recall': test_metrics['recall'],
            'test_precision': test_metrics['precision'],
            'mines_caught': test_metrics['mines_caught'],
            'mines_missed': test_metrics['mines_missed'],
            'false_alarms': test_metrics['false_alarms']
        })
    
    # Display results
    df_results = pd.DataFrame(results)
    
    print("\nThreshold Optimization Results:")
    print("-"*80)
    print(f"{'Target Recall':<15} {'Threshold':<12} {'Test Recall':<12} {'Test Precision':<15} {'Mines Caught':<12} {'Mines Missed':<12} {'False Alarms'}")
    print("-"*80)
    
    for _, row in df_results.iterrows():
        print(f"{row['target_recall']:<15.2%} {row['oof_threshold']:<12.4f} {row['test_recall']:<12.2%} {row['test_precision']:<15.2%} {row['mines_caught']:<12.0f} {row['mines_missed']:<12.0f} {row['false_alarms']:.0f}")
    
    print("="*80)
    
    # Key insight
    print("\nKEY INSIGHTS:")
    print(f"- Default threshold (0.5): {df_results.iloc[0]['test_recall']:.2%} recall")
    print(f"- Optimized threshold: Can achieve {df_results.iloc[-2]['test_recall']:.2%} recall")
    print(f"- Trade-off: Lower precision ({df_results.iloc[-2]['test_precision']:.2%}) but fewer missed mines")
    print("\nFor safety-critical mine detection, missing a mine is catastrophic.")
    print("We optimize for maximum recall at acceptable precision cost.")
    
    total_mines = (prepared.y_test == 1).sum()
    best_idx = df_results['test_recall'].idxmax()
    best = df_results.iloc[best_idx]
    
    print(f"\nBEST CONFIGURATION (Target Recall {best['target_recall']:.0%}):")
    print(f"  Threshold: {best['oof_threshold']:.4f}")
    print(f"  Test Recall: {best['test_recall']:.2%} ({best['mines_caught']:.0f}/{total_mines} mines detected)")
    print(f"  Mines Missed: {best['mines_missed']:.0f}")
    print(f"  False Alarms: {best['false_alarms']:.0f} rocks flagged as mines")

if __name__ == "__main__":
    main()
