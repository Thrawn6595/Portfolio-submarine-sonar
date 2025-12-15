from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, fbeta_score, make_scorer, precision_score, recall_score, roc_auc_score, average_precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from .config import DEFAULT_SEED

F2_SCORER = make_scorer(fbeta_score, beta=2, zero_division=0)

def resolve_scorer(scoring: str):
    return F2_SCORER if scoring == "f2" else scoring

@dataclass
class CVRunResult:
    model_name: str
    best_params: Dict[str, Any]
    cv_best_score: float
    val_metrics: Dict[str, float]
    fitted_model: BaseEstimator

def evaluate_binary_classifier(model, X, y, fp_cost=1.0, fn_cost=5.0):
    preds = model.predict(X)
    metrics = {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "f2": fbeta_score(y, preds, beta=2, zero_division=0),
    }
    cm = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel()
    metrics["cost"] = float(fp_cost * fp + fn_cost * fn)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y, scores)
        metrics["pr_auc"] = average_precision_score(y, scores)
    return metrics

def train_model_with_cv(model, param_grid, X_train, y_train, scoring="f2", cv_folds=5, seed=DEFAULT_SEED, fp_cost=1.0, fn_cost=5.0):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    gs = GridSearchCV(estimator=model, param_grid=param_grid, scoring=resolve_scorer(scoring), cv=cv, n_jobs=-1, refit=True)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    fold_metrics = []
    for tr_idx, va_idx in cv.split(X_train, y_train):
        m = clone(best)
        m.fit(X_train[tr_idx], y_train[tr_idx])
        fold_metrics.append(evaluate_binary_classifier(m, X_train[va_idx], y_train[va_idx], fp_cost, fn_cost))
    mdf = pd.DataFrame(fold_metrics)
    agg = {}
    for col in mdf.columns:
        agg[f"cv_{col}_mean"] = mdf[col].mean()
        agg[f"cv_{col}_std"] = mdf[col].std(ddof=1)
    return CVRunResult(model_name=best.__class__.__name__, best_params=gs.best_params_, cv_best_score=float(gs.best_score_), val_metrics=agg, fitted_model=best)

def create_cv_results_table(results: List[CVRunResult], rank_by="cv_f2_mean"):
    rows = [{"model": r.model_name, "cv_best_score": r.cv_best_score, **r.val_metrics, "best_params": r.best_params} for r in results]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if rank_by not in df.columns:
        rank_by = "cv_f2_mean"
    ascending = True if "cost" in rank_by else False
    return df.sort_values(rank_by, ascending=ascending).reset_index(drop=True)
