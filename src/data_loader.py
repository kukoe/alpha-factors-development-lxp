from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover_wan",
    "vwap",
    "daily_amplitude",
]


def load_futures_data(csv_path: str | Path) -> pd.DataFrame:
    """Load and clean futures daily main-contract data."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"date", "symbol", "open", "high", "low", "close", "volume", "vwap"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Guard against invalid values and keep only valid prices.
    for col in ["open", "high", "low", "close", "vwap"]:
        df.loc[df[col] <= 0, col] = np.nan

    df = df.dropna(subset=["date", "symbol", "close"]).copy()
    df["symbol"] = df["symbol"].astype(str)
    df["ret_1"] = df.groupby("symbol")["close"].pct_change()
    df["fwd_ret_1"] = df.groupby("symbol")["close"].shift(-1) / df["close"] - 1.0

    return df

