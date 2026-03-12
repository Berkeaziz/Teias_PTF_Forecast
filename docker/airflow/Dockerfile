FROM apache/airflow:2.9.3-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow

RUN pip install --no-cache-dir \
    pandas \
    numpy \
    pyarrow \
    openpyxl \
    scikit-learn \
    lightgbm \
    statsmodels \
    mlflow \
    psycopg2-binary \
    sqlalchemy \
    requests \
    python-dotenv \
    joblib