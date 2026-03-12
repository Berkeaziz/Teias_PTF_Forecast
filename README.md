⚡ Electricity Price Forecasting Pipeline (PTF)

Production-style machine learning pipeline for forecasting the Turkish
electricity market price (PTF) using data from the EPİAŞ Transparency
Platform.

This project demonstrates how to design a reproducible and modular ML
pipeline including:

-   automated data ingestion
-   data preprocessing
-   time‑series feature engineering
-   baseline and statistical model comparison
-   machine learning modeling
-   experiment tracking
-   workflow orchestration

Technologies used:

LightGBM • MLflow • Apache Airflow • Pandas • Statsmodels • Parquet

------------------------------------------------------------------------

SYSTEM ARCHITECTURE

EPİAŞ API │ ▼ Data Ingestion (fetch_epias.py) │ ▼ Raw Dataset (Parquet)
│ ▼ Data Processing (process_epias.py) │ ▼ Processed Dataset │ ▼ Feature
Engineering (build_features.py) │ ├── Train Mode │ │ │ ▼ │ LightGBM
Training │ │ │ ▼ │ Evaluation │ │ │ ▼ │ MLflow Tracking │ └── Inference
Mode │ ▼ Prediction

Each component can run independently or be orchestrated using Apache
Airflow.

------------------------------------------------------------------------

DATA INGESTION

Electricity price data is collected from the EPİAŞ Transparency Platform
API.

The ingestion script performs:

-   CAS authentication
-   retrieval of MCP/PTF electricity price data
-   transformation of API responses into pandas DataFrames
-   storage of data as partitioned Parquet datasets

Incremental Data Fetching

Instead of downloading the entire dataset each time:

1.  The script checks the latest timestamp already stored.
2.  Fetching starts from that timestamp.
3.  A small overlap window is used to avoid missing observations.

This allows the pipeline to run efficiently on scheduled workflows.

------------------------------------------------------------------------

DATA PROCESSING

Raw API data requires cleaning before modeling.

Processing steps:

-   column name standardization
-   timestamp parsing
-   conversion of ptf column to numeric
-   corrupted record removal
-   duplicate timestamp removal
-   chronological sorting
-   missing hourly observation checks

The output is a clean time‑series dataset ready for modeling.

------------------------------------------------------------------------

EXPLORATORY DATA ANALYSIS

Initial analysis revealed:

-   strong hourly patterns
-   strong daily seasonality
-   visible weekly patterns
-   strong predictive power of lag features such as lag_24 and lag_168

These insights guided the feature engineering process.

------------------------------------------------------------------------

FEATURE ENGINEERING

Feature engineering is a critical component of the pipeline.

Time Features - hour of day - day of week - weekend indicator - cyclic
encoding (sin / cos)

Lag Features - lag_1 - lag_24 - lag_168 - lag_336

Rolling Statistics - rolling_mean_24 - rolling_mean_168

Difference Features - diff_1 - diff_24 - diff_168

Train Mode

In training mode the pipeline creates a target variable:

target = ptf shifted by 24 hours

This allows the model to predict the electricity price for the same hour
on the following day.

Inference Mode

In inference mode:

-   only features are generated
-   the target column is not created

This enables the same pipeline to be used for real predictions.

------------------------------------------------------------------------

BASELINE MODEL

A baseline model was implemented using the lag‑24 approach.

PTF(t) ≈ PTF(t‑24)

Model Results

Model: Baseline (lag_24) MAE : 457.11 RMSE: 723.33 MAPE: 136.89

------------------------------------------------------------------------

SARIMA MODEL

A classical statistical model was implemented.

Configuration:

SARIMA(2,1,2)(1,1,0,24)

Results

MAE : 558.27 RMSE: 863.37 MAPE: 200.75

The SARIMA model did not outperform the lag‑based baseline.

------------------------------------------------------------------------

LIGHTGBM MODEL

A machine learning model was implemented using LightGBM.

Dataset split chronologically into:

-   training set
-   validation set
-   test set

Best Hyperparameters

n_estimators = 800 learning_rate = 0.02 max_depth = 6 num_leaves = 31
min_child_samples = 50 subsample = 1.0 colsample_bytree = 0.8

Validation Results

MAE : 296.55 RMSE: 406.59 MAPE: 69.41

Test Results

MAE : 392.49 RMSE: 548.30 MAPE: 246.74

------------------------------------------------------------------------

MODEL COMPARISON

Baseline (lag_24) MAE : 457.11 RMSE: 723.33

SARIMA MAE : 558.27 RMSE: 863.37

LightGBM MAE : 392.49 RMSE: 548.30

LightGBM achieved the lowest MAE and RMSE.

High MAPE values occur when PTF values approach zero, making percentage
errors unstable.

------------------------------------------------------------------------

EXPERIMENT TRACKING

MLflow was integrated for experiment tracking.

MLflow logs:

-   model hyperparameters
-   validation metrics
-   test metrics
-   trained model artifacts

This allows experiments to be reproduced and compared.

------------------------------------------------------------------------

WORKFLOW AUTOMATION

The pipeline is orchestrated using Apache Airflow.

Retraining Pipeline

fetch_epias → process_epias → build_features (train mode) → train_lgbm →
evaluate

This workflow periodically retrains the model with new data.

Inference Pipeline

fetch_epias → process_epias → build_features (inference mode) → predict

This workflow produces updated electricity price predictions.

------------------------------------------------------------------------

TECHNOLOGIES

Python Pandas LightGBM Statsmodels MLflow Apache Airflow Parquet EPİAŞ
Transparency Platform API

------------------------------------------------------------------------

CONCLUSION

This project demonstrates a production‑style machine learning pipeline
for electricity price forecasting.

The system includes:

-   automated data ingestion
-   robust preprocessing
-   time‑series feature engineering
-   baseline and statistical modeling
-   machine learning forecasting
-   experiment tracking
-   workflow orchestration

The final architecture is modular, reproducible, and suitable for
automated retraining and prediction generation.
