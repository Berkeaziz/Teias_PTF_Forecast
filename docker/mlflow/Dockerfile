FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow psycopg2-binary

ENV MLFLOW_BACKEND_STORE_URI=""
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=""

EXPOSE 5000

CMD mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --allowed-hosts "localhost,127.0.0.1,mlflow,mlflow:5000,localhost:5000" \
  --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
  --default-artifact-root ${MLFLOW_DEFAULT_ARTIFACT_ROOT}