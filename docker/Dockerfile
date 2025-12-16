FROM python:3.11-slim

WORKDIR /app

# Copy packages
COPY packages/sonar_model /app/sonar_model
COPY packages/sonar_api /app/sonar_api

# Install packages
RUN pip install --no-cache-dir /app/sonar_model
RUN pip install --no-cache-dir /app/sonar_api

# Copy trained model
RUN mkdir -p /app/sonar_api/sonar_api
COPY packages/sonar_api/sonar_api/trained_model.pkl /app/sonar_api/sonar_api/trained_model.pkl

EXPOSE 8000

CMD ["uvicorn", "sonar_api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
