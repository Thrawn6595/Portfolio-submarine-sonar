# Sonar Mine vs Rock Classification

Binary classification using sonar signals to distinguish between metal cylinders (mines) and rocks.

## Setup
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Dataset

- Samples: 208
- Features: 60 (sonar frequencies)
- Target: Mine (M) vs Rock (R)

## Structure
```
Portfolio-submarine-sonar/
├── configs/              # Configuration
├── data/                 # Data (gitignored)
├── notebooks/            # EDA notebooks
├── ml_toolkit/           # Reusable analysis library
├── artifacts/            # Generated outputs
└── models/               # Trained models
```
