# Future Extensions & Enhancements

## 1. Sensor Robustness & Fault Tolerance

### Motivation
In real submarine operations, sonar sensors may experience:
- Hardware malfunctions affecting specific frequency ranges
- Power constraints limiting active frequencies
- Environmental interference disrupting signal collection
- Partial sensor failures

### Proposed Solution: Robust Feature Handling

**Implementation Strategy:**
- Train model variants with randomly masked features (feature dropout)
- Create ensemble models that handle different feature subsets
- Implement feature importance ranking for graceful degradation
- Add fallback predictions when critical features missing

**Example Scenarios:**
```python
# Scenario 1: Power diverted to only 10 key frequencies
critical_features = top_10_features_by_importance
model_subset = train_on_features(critical_features)

# Scenario 2: Random frequency bands fail
def predict_with_missing_features(features, missing_indices):
    # Impute missing with learned defaults
    # Use ensemble of models trained on different subsets
    return robust_prediction
```

**Benefits:**
- Model remains operational under sensor degradation
- Maintains reasonable accuracy with 10-20 features vs full 60
- Reduces false negatives even with incomplete data
- Enables predictive maintenance (detect sensor issues)

**Implementation Effort:** Medium (2-3 weeks)

---

## 2. Neural Network Architecture

### Motivation
Current SVM achieves 95.45% recall. Neural networks could potentially:
- Capture non-linear patterns more effectively
- Achieve closer to 100% recall through better feature learning
- Handle missing features more gracefully

### Proposed Architectures

**Simple MLP:**
```python
def create_mlp_model():
    return Sequential([
        Dense(128, activation='relu', input_dim=60),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
```

**1D CNN for Frequency Patterns:**
```python
def create_cnn_model():
    return Sequential([
        Conv1D(64, 3, activation='relu', input_shape=(60, 1)),
        MaxPooling1D(2),
        Conv1D(32, 3, activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
```

**Integration:**
- Add to `sonar_model/modeling.py` model registry
- Use same training pipeline (already supports any sklearn-compatible estimator)
- Wrap Keras models with `scikeras.KerasClassifier`

**Expected Improvement:** 95% → 98%+ recall with proper regularization

**Implementation Effort:** Low (3-5 days)

---

## 3. Advanced Threshold Calibration

### Current State
- Default threshold (0.5): 95.45% recall, 77.78% precision
- Using `optimize_threshold.py` can achieve 98-99% recall

### Proposed Enhancements

**Cost-Sensitive Threshold:**
```python
# False negative (missed mine) = $1M (catastrophic)
# False positive (false alarm) = $10K (investigation cost)
optimal_threshold = find_cost_minimizing_threshold(
    fn_cost=1_000_000,
    fp_cost=10_000
)
```

**Confidence-Based Routing:**
- High confidence (>0.9): Auto-classify
- Medium confidence (0.7-0.9): Flag for manual review
- Low confidence (<0.7): Require expert validation

**Adaptive Thresholds:**
- Adjust based on operational context (war zone vs peacetime)
- Dynamic thresholds based on recent detection patterns

**Implementation Effort:** Low (1-2 days)

---

## 4. Model Interpretability

### SHAP Values for Predictions
```python
import shap

explainer = shap.KernelExplainer(model.predict_proba, X_train_sample)
shap_values = explainer.shap_values(X_test_sample)

# Show which frequencies influenced classification
shap.force_plot(explainer.expected_value[1], shap_values[1][0])
```

**Benefits:**
- Understand why model flagged object as mine
- Identify critical frequency ranges
- Validate model logic aligns with sonar physics
- Build operator trust

**Implementation Effort:** Low (2-3 days)

---

## 5. Continuous Learning Pipeline

### Online Learning
- Retrain model as new sonar data collected
- A/B test new models before deployment
- Monitor for data drift (environmental changes)

### Deployment Pipeline
```bash
# Automated retraining
make retrain-model
make validate-performance
make deploy-if-improved
```

**Implementation Effort:** Medium (1-2 weeks)

---

## 6. Multi-Model Ensemble

### Stacking Approach
```python
# Level 1: Base models
base_models = [svm, random_forest, logistic, neural_net]

# Level 2: Meta-learner combines predictions
meta_model = LogisticRegression()
stacked_ensemble = StackingClassifier(base_models, meta_model)
```

**Expected Improvement:** 1-2% recall boost, more robust predictions

**Implementation Effort:** Low (3-4 days)

---

## Priority Ranking

| Extension | Impact | Effort | Priority |
|-----------|--------|--------|----------|
| Threshold Calibration | High | Low | 1 |
| Neural Networks | Medium | Low | 2 |
| Sensor Robustness | High | Medium | 3 |
| Model Interpretability | Medium | Low | 4 |
| Ensemble Methods | Medium | Low | 5 |
| Continuous Learning | High | Medium | 6 |

---

## Current Architecture Supports Extensions

The two-package design (`sonar_model` + `sonar_api`) makes extensions easy:

1. **Add new models:** Just update `modeling.py` registry
2. **Change thresholds:** Modify API prediction logic
3. **Handle missing features:** Add preprocessing in `processing/`
4. **Deploy new version:** Build new Docker image

**All extensions maintain backward compatibility with existing API.**
