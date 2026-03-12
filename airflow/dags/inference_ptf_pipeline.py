from __future__ import annotations

import pendulum

from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_DIR = "/opt/airflow"
PYTHON_BIN = "python"

default_args = {
    "owner": "berke",
    "retries": 1,
}

with DAG(
    dag_id="inference_ptf_pipeline",
    description="PTF inference pipeline: fetch -> process -> build inference features -> predict",
    start_date=pendulum.datetime(2026, 3, 12, tz="Europe/Istanbul"),
    schedule_interval="0 * * * *",   
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["ptf", "inference", "ml"],
) as dag:

    fetch_epias = BashOperator(
        task_id="fetch_epias",
        cwd=PROJECT_DIR,
        bash_command=f'"{PYTHON_BIN}" src/ingestion/fetch_epias.py',
    )

    process_epias = BashOperator(
        task_id="process_epias",
        cwd=PROJECT_DIR,
        bash_command=f'"{PYTHON_BIN}" src/processing/process_epias.py',
    )

    build_features_inference = BashOperator(
        task_id="build_features_inference",
        cwd=PROJECT_DIR,
        bash_command=(
            f'"{PYTHON_BIN}" src/features/build_features.py '
            f'--input data/processed/ptf_processed.parquet '
            f'--output data/features/ptf_features_inference.parquet '
            f'--mode inference'
        ),
    )

    predict_ptf = BashOperator(
        task_id="predict_ptf",
        cwd=PROJECT_DIR,
        bash_command=(
            f'"{PYTHON_BIN}" src/inference/predict.py '
            f'--input data/features/ptf_features_inference.parquet '
            f'--model models/lgbm_model.pkl '
            f'--output data/predictions/ptf_predictions.parquet'
        ),
    )

    fetch_epias >> process_epias >> build_features_inference >> predict_ptf