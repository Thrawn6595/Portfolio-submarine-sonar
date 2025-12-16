.PHONY: install-all train-model test-model run-api docker-build docker-run clean

install-all:
	pip install -e ml_toolkit/
	pip install -e packages/sonar_model/
	pip install -e packages/sonar_api/

train-model:
	cd packages/sonar_model && python train_pipeline.py --data-path ../../data/raw/sonar.csv

save-model:
	cd packages/sonar_model && python save_model.py

test-model:
	cd packages/sonar_model && pytest tests/ -v

run-api:
	cd packages/sonar_api && uvicorn sonar_api.app.main:app --reload --port 8000

docker-build:
	docker build -f docker/Dockerfile -t sonar-api:latest .

docker-run:
	docker run -p 8000:8000 sonar-api:latest

docker-stop:
	docker stop $$(docker ps -q --filter ancestor=sonar-api:latest)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +

help:
	@echo "Available commands:"
	@echo "  make install-all   - Install all packages"
	@echo "  make train-model   - Train models"
	@echo "  make save-model    - Save champion model"
	@echo "  make run-api       - Run API locally"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run Docker container"
	@echo "  make clean         - Clean Python artifacts"
