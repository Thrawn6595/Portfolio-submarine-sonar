# Sonar API

FastAPI wrapper for sonar mine vs rock classification.

## Installation
```bash
pip install sonar-api
```

## Usage
```bash
uvicorn sonar_api.app.main:app --reload
```

Then visit http://localhost:8000/docs

## Endpoints

- GET / - Root
- GET /health - Health check
- POST /predict - Classify sonar signal
- GET /model/info - Model metadata
