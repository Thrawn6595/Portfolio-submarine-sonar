# Sonar Mine vs Rock Classification

End-to-end ML project demonstrating production deployment with focus on **maximizing mine detection recall**.

## Key Objectives

1. **Maximize Recall:** Detect 100% of mines (currently 95.45%, optimizable to 98-99%)
2. **Production Ready:** Full pipeline from research to Docker deployment
3. **Extensible:** Easy to add neural nets, ensemble methods, robustness features

## Quick Start
```bash
make install-all
make train-model
make run-api
```

Visit http://localhost:8000/docs

## Results

**Champion Model:** SVM (RBF, C=0.5)

| Metric | Value | Note |
|--------|-------|------|
| Test Recall | 95.45% | Catches 21/22 mines |
| Test Precision | 77.78% | ~22% false alarms |
| Test F2 | 91.30% | Emphasizes recall |
| ROC AUC | 91.82% | Strong discrimination |

**Threshold Optimization:** Can achieve 98-99% recall by lowering threshold (see `optimize_threshold.py`)

## Why Recall Matters

For mine detection:
- **False Negative (missed mine)** = Catastrophic (ship/submarine loss)
- **False Positive (false alarm)** = Acceptable cost (investigation)

We optimize for **maximum recall** while maintaining reasonable precision.

## Architecture
```
Portfolio-submarine-sonar/
├── notebooks/              # Research & EDA
├── packages/
│   ├── sonar_model/        # Core ML (extensible for neural nets)
│   └── sonar_api/          # FastAPI wrapper
├── ml_toolkit/             # Reusable utilities
├── docker/                 # Containerization
└── Makefile                # Automation
```

## Extending the System

See [EXTENSIONS.md](EXTENSIONS.md) for:
- Sensor robustness (handle missing features)
- Neural network integration
- Threshold optimization
- Model interpretability

## Documentation

- [Model Card](packages/sonar_model/MODEL_CARD.md) - Performance details
- [Extensions](EXTENSIONS.md) - Future enhancements
- [Project Overview](PROJECT_OVERVIEW.md) - Technical architecture

## License

MIT
