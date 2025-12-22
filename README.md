![Sonar Mine Detection](./notebooks/images/00_Sonar_Mine_detection_Cover.png)

# Sonar Mine vs Rock Classification

Context:
The basic premise is that Naval vessels navigate waters where underwater mines pose existential threats, active sonar returns provide the only advance warning, but distinguishing mines from naturally occurring rocks requires real-time classification under uncertainty.

This Project explores a sample of the 208 sonar record set across 60 signal features, to understand whether ML classification is feasible and can add value to building a mine detection system, and what data processing and modelling approaches will need to be considered to deliver on the ML and Business objectives of keeping vessels and crew safe.

Most attempts at this ignore that this is a and blindly use accuracy as the key ML and Business objective. We will take a different route and build an end to end Ml solution that identifies the Business objective and the Machine learning implication for the Design, development and deployment lifecycle of the system.



## Key Objectives

1. **Maximize Recall:** Detect 100% of mines, without precision collapsing below 70%
2. **Production Ready:** End-to-end ML project,full pipeline from research to API implementation to Docker deployment


What good looks like: 



## Quick Start

```bash
make install-all
make train-model
make run-api
```

Visit <http://localhost:8000/docs>

## Results

**Champion Model:** SVM (RBF, C=0.5)

| Metric | Value | Note |
|--------|-------|------|
| Test Recall | 95.45% | Catches 21/22 mines |
| Test Precision | 77.78% | ~22% false alarms |
| Test F2 | 91.30% | Emphasizes recall |
| ROC AUC | 91.82% | Strong discrimination |

**Threshold Optimization:** Can achieve 100% recall by lowering threshold (see `optimize_threshold.py`)

## Why Recall Matters

For mine detection:

- **False Negative (missed mine)** = Catastrophic (ship/submarine loss)
- **False Positive (false alarm)** = Acceptable cost (investigation)

We optimize for **maximum recall** while maintaining reasonable precision.

## Architecture

``` {markdown}
Portfolio-submarine-sonar/
├── notebooks/              # Research & EDA
├── packages/
│   ├── sonar_model/        # Core ML (extensible for neural nets)
│   └── sonar_api/          # FastAPI wrapper
├── ml_toolkit/             # Reusable utilities
├── docker/                 # Containerization
└── Makefile                # Automation
```
