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
    dag_id="retrain_ptf_pipeline",
    description="PTF retraining pipeline: fetch -> process -> build train features -> train -> evaluate",
    start_date=pendulum.datetime(2026, 3, 8, tz="Europe/Istanbul"),
    schedule_interval="0 3 * * 0",   
    catchup=False,
    default_args=default_args,
    tags=["ptf", "retrain", "ml"],
) as dag:

    fetch_epias = BashOperator(
        task_id="fetch_epias",
        cwd=PROJECT_DIR,
        bash_command='set -e; python src/ingestion/fetch_epias.py',
    )

    process_epias = BashOperator(
        task_id="process_epias",
        cwd=PROJECT_DIR,
        bash_command='set -e; python src/processing/process_epias.py',
    )

    build_features_train = BashOperator(
        task_id="build_features_train",
        cwd=PROJECT_DIR,
        bash_command=(
            'set -e; python src/features/build_features.py '
            '--input data/processed/ptf_processed.parquet '
            '--output data/features/ptf_features_train.parquet '
            '--mode train '
            '--horizon 24'
        ),
    )

    train_lgbm = BashOperator(
        task_id="train_lgbm",
        cwd=PROJECT_DIR,
        bash_command=(
            'set -e; python src/training/train_lgbm.py '
            '--input data/features/ptf_features_train.parquet '
            '--model-output models/lgbm_model.pkl '
            '--metrics-output artifacts/training/train_metrics.json '
            '--experiment-name ptf_forecasting_lgbm'
        ),
    )

    evaluate_model = BashOperator(
        task_id="evaluate_model",
        cwd=PROJECT_DIR,
        bash_command=(
            'set -e; python src/training/evaluate.py '
            '--input data/features/ptf_features_train.parquet '
            '--model models/lgbm_model.pkl '
            '--metrics-output artifacts/evaluation/evaluation_metrics.json '
            '--predictions-output artifacts/evaluation/test_predictions.parquet'
        ),
    )

    fetch_epias >> process_epias >> build_features_train >> train_lgbm >> evaluate_model