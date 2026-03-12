from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def load_features(input_path: str | Path) -> pd.DataFrame:
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Feature file not found: {input_path}")

    df = pd.read_parquet(input_path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.set_index("date")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Feature table must have a DatetimeIndex or a 'date' column.")

    df = df.sort_index()

    return df


def load_model(model_path: str | Path):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    return model


def prepare_features(df: pd.DataFrame, target_col: str = "target") -> pd.DataFrame:
    df = df.copy()

    if target_col in df.columns:
        df = df.drop(columns=[target_col])

    return df


def make_prediction_df(
    feature_df: pd.DataFrame,
    predictions,
) -> pd.DataFrame:
    pred_df = pd.DataFrame(
        {
            "prediction": predictions,
        },
        index=feature_df.index,
    )

    pred_df.index.name = "date"
    return pred_df


def save_predictions(df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, engine="pyarrow", index=True)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PTF predictions with a trained LightGBM model.")

    parser.add_argument(
        "--input",
        type=str,
        default="data/features/ptf_features_inference.parquet",
        help="Path to feature parquet file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/lgbm_model.pkl",
        help="Path to trained model file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/predictions/ptf_predictions.parquet",
        help="Path to save predictions.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading features from: {args.input}")
    df = load_features(args.input)
    print(f"Feature table shape: {df.shape}")

    print(f"Loading model from: {args.model}")
    model = load_model(args.model)

    X = prepare_features(df)
    print(f"Prediction feature shape: {X.shape}")

    print("Generating predictions...")
    preds = model.predict(X)

    pred_df = make_prediction_df(X, preds)
    print(f"Prediction table shape: {pred_df.shape}")

    saved_path = save_predictions(pred_df, args.output)
    print(f"Predictions saved to: {saved_path}")
    print("Prediction completed successfully.")


if __name__ == "__main__":
    main()