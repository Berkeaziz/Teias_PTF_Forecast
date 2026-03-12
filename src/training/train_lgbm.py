from __future__ import annotations

import argparse
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_data(input_path: str | Path) -> pd.DataFrame:
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_parquet(input_path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.set_index("date")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a DatetimeIndex or a 'date' column.")

    df = df.sort_index()

    if "target" not in df.columns:
        raise ValueError("Missing required target column: 'target'")

    return df


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def make_xy(
    df: pd.DataFrame,
    target_col: str = "target",
) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    return X, y


def mean_absolute_percentage_error_safe(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")

    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_forecast(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error_safe(y_true, y_pred)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
    }

def train_model(
    X_train: pd.DataFrame,
    y_train:pd.Series,
    X_val:pd.DataFrame,
    y_val: pd.Series,
)-> LGBMRegressor:
    model =LGBMRegressor(
        n_estimators=800,
        learning_rate=0.02,
        max_depth=6,
        num_leaves=31,
        min_child_samples=50,
        subsample=1,
        colsample_bytree=0.8,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val,y_val)],
        eval_metric="l1"
    )

    return model

def save_model(model: LGBMRegressor, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_path)
    return output_path

def save_metrics(metrics: dict, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return output_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM model for PTF forecasting.")

    parser.add_argument(
        "--input",
        type=str,
        default="data/features/ptf_features_train.parquet",
        help="Path to feature parquet file.",
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default="models/lgbm_model.pkl",
    )

    parser.add_argument(
        "--metrics-output",
        type=str,
        default="artifacts/training/train_metrics.json",
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    print(f"Loading data from: {args.input}")
    df = load_data(args.input)
    print(f"Dataset shape: {df.shape}")

    train_df, val_df, test_df = split_data(df)

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape  : {val_df.shape}")
    print(f"Test shape : {test_df.shape}")

    X_train, y_train = make_xy(train_df)
    X_val, y_val = make_xy(val_df)
    X_test, y_test = make_xy(test_df)

    print("Training LightGBM model...")
    model = train_model(X_train, y_train, X_val, y_val)

    print("Evaluating on validation set...")
    val_pred = model.predict(X_val)
    val_metrics = evaluate_forecast(y_val, val_pred)

    print("Evaluating on test set...")
    test_pred = model.predict(X_test)
    test_metrics = evaluate_forecast(y_test, test_pred)

    all_metrics = {
        "validation": val_metrics,
        "test": test_metrics,
    }

    print("\nValidation Metrics")
    for k, v in val_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nTest Metrics")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    model_path = save_model(model, args.model_output)
    metrics_path = save_metrics(all_metrics, args.metrics_output)

    print(f"\nModel saved to   : {model_path}")
    print(f"Metrics saved to : {metrics_path}")
    print("Training completed successfully.")


if __name__ == "__main__":
    main()