from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


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

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    if df.index.has_duplicates:
        print("WARNING: Duplicate timestamps found. Keeping first occurrence.")
        df = df[~df.index.duplicated(keep="first")]

    if "ptf" not in df.columns:
        raise ValueError("Missing required column: 'ptf'")

    df["ptf"] = pd.to_numeric(df["ptf"], errors="coerce")
    df = df.dropna(subset=["ptf"])

    return df


def base_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    base = df[["ptf"]].copy()
    return base


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["hour"] = df.index.hour.astype("int8")
    df["day_of_week"] = df.index.dayofweek.astype("int8")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["lag_1"] = df["ptf"].shift(1)
    df["lag_24"] = df["ptf"].shift(24)
    df["lag_168"] = df["ptf"].shift(168)
    df["lag_336"] = df["ptf"].shift(336)

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    past_ptf = df["ptf"].shift(1)

    df["rolling_mean_24"] = past_ptf.rolling(window=24).mean()
    df["rolling_mean_168"] = past_ptf.rolling(window=168).mean()

    return df


def add_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["diff_1"] = df["ptf"].shift(1) - df["ptf"].shift(2)
    df["diff_24"] = df["ptf"].shift(1) - df["ptf"].shift(25)
    df["diff_168"] = df["ptf"].shift(1) - df["ptf"].shift(169)

    return df


def build_target(df: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    df = df.copy()
    df["target"] = df["ptf"].shift(-horizon)
    return df


def build_features(
    df: pd.DataFrame,
    mode: str = "train",
    horizon: int = 24,
) -> pd.DataFrame:
    df = base_table(df)

    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_diff_features(df)

    if mode == "train":
        df = build_target(df, horizon=horizon)
    elif mode == "inference":
        pass
    else:
        raise ValueError("mode must be either 'train' or 'inference'")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df


def save_data(df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, engine="pyarrow", index=True)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build features for PTF forecasting.")

    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/ptf_processed.parquet",
        help="Path to processed parquet file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/features/ptf_features.parquet",
        help="Path to output feature parquet file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "inference"],
        default="train",
        help="Feature generation mode.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=24,
        help="Forecast horizon in hours. Used only in train mode.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading data from: {args.input}")
    df = load_data(args.input)
    print(f"Input shape: {df.shape}")

    print(f"Building features in '{args.mode}' mode...")
    df_features = build_features(
        df=df,
        mode=args.mode,
        horizon=args.horizon,
    )
    print(f"Feature shape: {df_features.shape}")
    print(f"Columns: {list(df_features.columns)}")

    print(f"Saving features to: {args.output}")
    saved_path = save_data(df_features, args.output)

    print(f"Features saved to: {saved_path}")
    print("Feature engineering completed successfully.")


if __name__ == "__main__":
    main()