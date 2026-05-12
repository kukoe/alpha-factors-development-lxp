from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-12


def _multi_window_median(series_dict: dict[int, pd.Series]) -> pd.Series:
    stacked = pd.concat(series_dict.values(), axis=1)
    return stacked.median(axis=1)


def _cs_preprocess(out: pd.DataFrame, col: str) -> pd.Series:
    """Winsorize -> cross-sectional zscore -> fillna(0)."""
    s = out[col].replace([np.inf, -np.inf], np.nan)
    p01 = s.groupby(out["date"]).transform(lambda x: x.quantile(0.01))
    p99 = s.groupby(out["date"]).transform(lambda x: x.quantile(0.99))
    s = s.clip(lower=p01, upper=p99)
    mean = s.groupby(out["date"]).transform("mean")
    std = s.groupby(out["date"]).transform("std")
    s = (s - mean) / (std + EPS)
    return s.fillna(0.0)


def _rolling_corr_by_symbol(data: pd.DataFrame, x_col: str, y_col: str, window: int) -> pd.Series:
    return (
        data.groupby("symbol")[[x_col, y_col]]
        .apply(lambda g: g[x_col].rolling(window).corr(g[y_col]))
        .reset_index(level=0, drop=True)
    )


def _signed_streak(series: pd.Series) -> pd.Series:
    out = np.zeros(len(series))
    prev_sign = 0
    streak = 0
    vals = np.sign(series.fillna(0.0).to_numpy())
    for i, s in enumerate(vals):
        if s == 0:
            streak = 0
            prev_sign = 0
            out[i] = 0
            continue
        if s == prev_sign:
            streak += 1
        else:
            streak = 1
            prev_sign = s
        out[i] = streak * s
    return pd.Series(out, index=series.index)


def compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Compute candidate alpha factors from daily futures OHLCV data."""
    data = df.copy()
    data = data.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = data.groupby("symbol")
    windows = [3, 5, 10, 20, 60]

    # Base series
    data["ret_1"] = g["close"].pct_change()
    data["mom_10"] = g["close"].pct_change(10)
    data["log_volume"] = np.log1p(data["volume"].clip(lower=0))
    data["volume_chg_1"] = g["volume"].pct_change()
    data["intraday_range"] = (data["high"] - data["low"]) / (data["close"] + EPS)
    data["intraday_pos"] = (data["close"] - data["low"]) / ((data["high"] - data["low"]) + EPS)
    data["upper_shadow"] = (data["high"] - data[["open", "close"]].max(axis=1)) / (data["close"] + EPS)
    data["lower_shadow"] = (data[["open", "close"]].min(axis=1) - data["low"]) / (data["close"] + EPS)
    data["body_strength"] = (data["close"] - data["open"]) / (data["open"] + EPS)

    # Existing factor family
    ma_5 = g["close"].rolling(5).mean().reset_index(level=0, drop=True)
    ma_10 = g["close"].rolling(10).mean().reset_index(level=0, drop=True)
    ma_20 = g["close"].rolling(20).mean().reset_index(level=0, drop=True)
    data["ma_slope_5"] = ma_5 / (ma_5.groupby(data["symbol"]).shift(5) + EPS) - 1.0
    data["ma_dev_20"] = (data["close"] - ma_20) / (ma_20 + EPS)
    data["mom_5"] = g["close"].pct_change(5)
    data["mom_20"] = g["close"].pct_change(20)
    data["mom_accel_5_20"] = data["mom_5"] - data["mom_20"]
    data["reversal_1"] = -data["ret_1"]
    data["breakout_20"] = data["close"] / (g["high"].rolling(20).max().reset_index(level=0, drop=True) + EPS) - 1.0
    ret_std_20 = g["ret_1"].rolling(20).std().reset_index(level=0, drop=True)
    ret_mean_20 = g["ret_1"].rolling(20).mean().reset_index(level=0, drop=True)
    data["volatility_20"] = ret_std_20
    data["trend_stability_20"] = ret_mean_20 / (ret_std_20 + EPS)
    data["amplitude_mean_5"] = g["intraday_range"].rolling(5).mean().reset_index(level=0, drop=True)
    data["volume_zscore_20"] = (
        data["log_volume"] - g["log_volume"].rolling(20).mean().reset_index(level=0, drop=True)
    ) / (g["log_volume"].rolling(20).std().reset_index(level=0, drop=True) + EPS)
    data["turnover_shock_20"] = data["turnover_wan"] / (
        g["turnover_wan"].rolling(20).mean().reset_index(level=0, drop=True) + EPS
    ) - 1.0
    data["vwap_bias_5"] = data["close"] / (g["vwap"].rolling(5).mean().reset_index(level=0, drop=True) + EPS) - 1.0
    data["vwap_close_gap"] = (data["vwap"] - data["close"]) / (data["close"] + EPS)
    data["shadow_skew"] = data["lower_shadow"] - data["upper_shadow"]
    data["price_volume_corr"] = _multi_window_median(
        {w: _rolling_corr_by_symbol(data, "ret_1", "volume_chg_1", w) for w in windows}
    )
    data["volume_change_rate"] = _multi_window_median(
        {w: data["volume_chg_1"].groupby(data["symbol"]).rolling(w).mean().reset_index(level=0, drop=True) for w in windows}
    )
    prev_close = g["close"].shift(1)
    true_range = np.maximum.reduce(
        [
            (data["high"] - data["low"]).to_numpy(),
            (data["high"] - prev_close).abs().to_numpy(),
            (data["low"] - prev_close).abs().to_numpy(),
        ]
    )
    data["atr_mean"] = _multi_window_median(
        {w: pd.Series(true_range, index=data.index).groupby(data["symbol"]).rolling(w).mean().reset_index(level=0, drop=True) for w in windows}
    )

    # Priority-1: asymmetry volatility factors
    downside = data["ret_1"].where(data["ret_1"] < 0, 0.0)
    upside = data["ret_1"].where(data["ret_1"] > 0, 0.0)
    data["downside_volatility"] = _multi_window_median(
        {w: downside.groupby(data["symbol"]).rolling(w).std().reset_index(level=0, drop=True) for w in windows}
    )
    up_vol = _multi_window_median(
        {w: upside.groupby(data["symbol"]).rolling(w).std().reset_index(level=0, drop=True) for w in windows}
    )
    down_vol = _multi_window_median(
        {w: downside.groupby(data["symbol"]).rolling(w).std().reset_index(level=0, drop=True) for w in windows}
    )
    data["volatility_skew"] = down_vol / (up_vol + EPS)
    data["realized_skew"] = _multi_window_median(
        {w: g["ret_1"].rolling(w).skew().reset_index(level=0, drop=True) for w in windows}
    )

    # Priority-2: improved momentum factors
    smooth_close = _multi_window_median(
        {w: g["close"].rolling(w).mean().reset_index(level=0, drop=True) for w in [3, 5, 10]}
    )
    data["smoothed_momentum"] = _multi_window_median(
        {w: smooth_close.groupby(data["symbol"]).pct_change(w, fill_method=None) for w in windows}
    )
    market_ret = data.groupby("date")["ret_1"].transform("mean")
    residual_ret = data["ret_1"] - market_ret
    data["residual_momentum"] = _multi_window_median(
        {w: residual_ret.groupby(data["symbol"]).rolling(w).sum().reset_index(level=0, drop=True) for w in windows}
    )
    data["trend_strength"] = _multi_window_median(
        {w: np.sign(data["ret_1"]).groupby(data["symbol"]).rolling(w).mean().abs().reset_index(level=0, drop=True) for w in windows}
    )

    # Priority-3: price-volume relationship factors
    vol_chg = data["volume_chg_1"]
    pv_div = {}
    gv = data.groupby("symbol")[["ret_1", "volume"]]
    for w in windows:
        corr_w = gv.apply(
            lambda s: s["ret_1"].rolling(w).corr(s["volume"].pct_change())
        ).reset_index(level=0, drop=True)
        pv_div[w] = corr_w
    data["price_volume_divergence"] = -_multi_window_median(pv_div)
    vol_avg = _multi_window_median(
        {w: g["volume"].rolling(w).mean().reset_index(level=0, drop=True) for w in windows}
    )
    vol_std = _multi_window_median(
        {w: g["volume"].rolling(w).std().reset_index(level=0, drop=True) for w in windows}
    )
    data["abnormal_volume"] = (data["volume"] - vol_avg) / (vol_std + EPS)
    data["up_volume_ratio"] = _multi_window_median(
        {
            w: (
                data["volume"].where(data["ret_1"] > 0, 0.0).groupby(data["symbol"]).rolling(w).sum().reset_index(level=0, drop=True)
                / (data["volume"].groupby(data["symbol"]).rolling(w).sum().reset_index(level=0, drop=True) + EPS)
            )
            for w in windows
        }
    )

    # Trend extension factors
    data["ma_alignment"] = ((ma_5 > ma_10).astype(float) + (ma_10 > ma_20).astype(float)) - 1.0
    high_20 = g["high"].rolling(20).max().reset_index(level=0, drop=True)
    low_20 = g["low"].rolling(20).min().reset_index(level=0, drop=True)
    data["breakout_low_20"] = data["close"] / (low_20 + EPS) - 1.0
    data["close_in_range_20"] = (data["close"] - low_20) / ((high_20 - low_20) + EPS)
    data["oc_direction_consistency"] = _multi_window_median(
        {w: np.sign(data["body_strength"]).groupby(data["symbol"]).rolling(w).mean().reset_index(level=0, drop=True) for w in windows}
    )
    range_mean = g["intraday_range"].rolling(20).mean().reset_index(level=0, drop=True)
    data["range_breakout_strength"] = (data["intraday_range"] / (range_mean + EPS)) * (data["close_in_range_20"] - 0.5)
    data["trend_duration"] = g["ret_1"].transform(_signed_streak)
    dist_ma = data["close"] - ma_20
    data["price_ma_zscore"] = dist_ma / (
        dist_ma.groupby(data["symbol"]).rolling(20).std().reset_index(level=0, drop=True) + EPS
    )

    # Reversal extension factors
    data["reversal_2_5"] = -_multi_window_median(
        {w: g["close"].pct_change(w) for w in [2, 3, 4, 5]}
    )
    data["extreme_reversion"] = -data["ret_1"] / (ret_std_20 + EPS)
    data["close_pos_reversal"] = -(data["intraday_pos"] - 0.5)
    data["skew_reversal"] = -data["realized_skew"]
    vol_regime = ret_std_20 / (ret_std_20.groupby(data["symbol"]).rolling(20).mean().reset_index(level=0, drop=True) + EPS)
    data["high_vol_pullback"] = -data["ret_1"] * vol_regime
    data["big_candle_correction"] = -data["body_strength"].groupby(data["symbol"]).shift(1)
    data["volume_spike_reversion"] = -data["abnormal_volume"] * data["ret_1"]

    # Divergence extension factors
    data["vwap_volume_corr"] = _multi_window_median(
        {w: _rolling_corr_by_symbol(data, "vwap", "volume", w) for w in windows}
    )
    data["amplitude_volume_corr"] = _multi_window_median(
        {w: _rolling_corr_by_symbol(data, "intraday_range", "volume", w) for w in windows}
    )
    data["price_turnover_corr"] = _multi_window_median(
        {w: _rolling_corr_by_symbol(data, "ret_1", "turnover_wan", w) for w in windows}
    )
    data["up_price_vol_decay"] = _multi_window_median(
        {
            w: ((data["ret_1"] > 0) & (data["volume_chg_1"] < 0))
            .astype(float)
            .groupby(data["symbol"])
            .rolling(w)
            .mean()
            .reset_index(level=0, drop=True)
            for w in windows
        }
    )
    data["down_price_vol_expand"] = _multi_window_median(
        {
            w: ((data["ret_1"] < 0) & (data["volume_chg_1"] > 0))
            .astype(float)
            .groupby(data["symbol"])
            .rolling(w)
            .mean()
            .reset_index(level=0, drop=True)
            for w in windows
        }
    )
    prev_high_max = g["high"].rolling(20).max().shift(1).reset_index(level=0, drop=True)
    prev_low_min = g["low"].rolling(20).min().shift(1).reset_index(level=0, drop=True)
    prev_vol_max = g["volume"].rolling(20).max().shift(1).reset_index(level=0, drop=True)
    data["new_high_no_volume"] = ((data["close"] >= prev_high_max) & (data["volume"] < prev_vol_max)).astype(float)
    data["new_low_no_volume"] = ((data["close"] <= prev_low_min) & (data["volume"] < prev_vol_max)).astype(float)

    # Priority-4: higher moments and micro-structure proxies
    data["amplitude_stability"] = _multi_window_median(
        {w: g["intraday_range"].rolling(w).std().reset_index(level=0, drop=True) for w in windows}
    )
    data["close_position"] = data["intraday_pos"]
    data["shadow_ratio"] = data["upper_shadow"] / (data["lower_shadow"].abs() + EPS)
    data["realized_kurtosis"] = _multi_window_median(
        {w: g["ret_1"].rolling(w).kurt().reset_index(level=0, drop=True) for w in windows}
    )
    data["turnover_std"] = _multi_window_median(
        {w: g["turnover_wan"].rolling(w).std().reset_index(level=0, drop=True) for w in windows}
    )
    data["turnover_mean"] = _multi_window_median(
        {w: g["turnover_wan"].rolling(w).mean().reset_index(level=0, drop=True) for w in windows}
    )
    data["amplitude_mean_20"] = g["intraday_range"].rolling(20).mean().reset_index(level=0, drop=True)
    data["volume_volatility"] = _multi_window_median(
        {w: g["volume_chg_1"].rolling(w).std().reset_index(level=0, drop=True) for w in windows}
    )
    data["body_volatility"] = _multi_window_median(
        {w: g["body_strength"].rolling(w).std().reset_index(level=0, drop=True) for w in windows}
    )
    data["close_position_dispersion"] = _multi_window_median(
        {w: g["intraday_pos"].rolling(w).std().reset_index(level=0, drop=True) for w in windows}
    )

    # Priority-5: cross-product / cross-market factors
    data["relative_strength"] = _multi_window_median(
        {
            w: (
                g["close"].pct_change(w)
                - g["close"].pct_change(w).groupby(data["date"]).transform("mean")
            )
            for w in windows
        }
    )
    if "sector" in data.columns:
        rs = {}
        for w in windows:
            ret_w = g["close"].pct_change(w)
            sector_mean = ret_w.groupby([data["date"], data["sector"]]).transform("mean")
            rs[w] = ret_w - sector_mean
        data["sector_relative_strength"] = _multi_window_median(rs)
    else:
        data["sector_relative_strength"] = np.nan
    vol_roll = _multi_window_median(
        {w: g["ret_1"].rolling(w).std().reset_index(level=0, drop=True) for w in windows}
    )
    data["volatility_spillover"] = vol_roll - vol_roll.groupby(data["date"]).transform("mean")

    # Futures-specific placeholders/optionals
    if "open_interest" in data.columns and data["open_interest"].notna().sum() > 0:
        oi_chg = g["open_interest"].pct_change()
        data["oi_change_rate"] = oi_chg
        data["volume_oi_ratio"] = data["volume"] / (data["open_interest"] + EPS)
        data["price_oi_sync"] = _multi_window_median(
            {w: _rolling_corr_by_symbol(data.assign(oi_chg=oi_chg), "ret_1", "oi_chg", w) for w in windows}
        )
    else:
        data["oi_change_rate"] = np.nan
        data["volume_oi_ratio"] = np.nan
        data["price_oi_sync"] = np.nan
    data["basis_proxy"] = np.nan
    data["term_structure_slope"] = np.nan
    data["roll_impact"] = np.nan

    # Keep only factors with historically robust positive test net Sharpe.
    factor_cols = [
        "sector_relative_strength",
        "downside_volatility",
        "breakout_20",
        "amplitude_mean_5",
        "relative_strength",
        "body_strength",
        "residual_momentum",
        "volume_change_rate",
        "volatility_spillover",
        "volatility_20",
        "reversal_2_5",
        "mom_20",
        "mom_10",
        "smoothed_momentum",
        "reversal_1",
        "volatility_skew",
        "up_volume_ratio",
        "range_breakout_strength",
        "ma_dev_20",
        "trend_stability_20",
    ]

    data["fwd_ret_3"] = g["close"].shift(-3) / data["close"] - 1.0
    data["fwd_ret_5"] = g["close"].shift(-5) / data["close"] - 1.0

    out = data[["date", "symbol", "fwd_ret_1", "fwd_ret_3", "fwd_ret_5"] + factor_cols].copy()
    out = out.replace([np.inf, -np.inf], np.nan)

    # Standard factor preprocessing:
    # 1) winsorize, 2) daily cross-sectional zscore, 3) fillna.
    for col in factor_cols:
        out[col] = _cs_preprocess(out, col)

    return out

