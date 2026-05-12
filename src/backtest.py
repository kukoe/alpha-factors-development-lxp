from __future__ import annotations

import numpy as np
import pandas as pd


ANNUAL_TRADING_DAYS = 252.0


def time_split_dates(dates: pd.Series, train_ratio: float = 0.7) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    unique_dates = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Series(dates).dropna().unique())))
    split_idx = int(len(unique_dates) * train_ratio)
    split_idx = min(max(split_idx, 1), len(unique_dates) - 1)
    return unique_dates[:split_idx], unique_dates[split_idx:]


def build_orientation_map(
    train_df: pd.DataFrame,
    factor_cols: list[str],
    quantile: float,
) -> pd.DataFrame:
    rows = []
    for fac in factor_cols:
        result = build_strategy_result(train_df, fac, quantile=quantile, transaction_cost=0.0)
        sharpe = _safe_sharpe(result["gross_return"])
        sign = -1 if pd.notna(sharpe) and sharpe < 0 else 1
        rows.append(
            {
                "factor": fac,
                "train_sharpe_before_orient": sharpe,
                "orientation_sign": sign,
                "oriented": sign == -1,
            }
        )
    return pd.DataFrame(rows)


def apply_orientation(factor_df: pd.DataFrame, orientation_map: pd.DataFrame) -> pd.DataFrame:
    out = factor_df.copy()
    sign_map = dict(zip(orientation_map["factor"], orientation_map["orientation_sign"]))
    for fac, sign in sign_map.items():
        out[fac] = out[fac] * sign
    return out


def build_strategy_result(
    factor_df: pd.DataFrame,
    factor_col: str,
    forward_col: str = "fwd_ret_1",
    quantile: float = 0.2,
    transaction_cost: float = 0.0003,
) -> pd.DataFrame:
    """Build daily long-short strategy returns with turnover and costs."""
    dates = pd.DatetimeIndex(sorted(factor_df["date"].dropna().unique()))
    symbols = sorted(factor_df["symbol"].dropna().astype(str).unique())
    if len(dates) == 0 or len(symbols) == 0:
        return pd.DataFrame(
            columns=["gross_return", "turnover", "cost", "net_return", "long_leg_ret", "short_leg_ret"]
        )

    weights = pd.DataFrame(0.0, index=dates, columns=symbols)
    long_leg_ret = pd.Series(np.nan, index=dates)
    short_leg_ret = pd.Series(np.nan, index=dates)

    returns_mat = (
        factor_df.pivot_table(index="date", columns="symbol", values=forward_col, aggfunc="first")
        .reindex(index=dates, columns=symbols)
    )

    for date, grp in factor_df.groupby("date"):
        date = pd.Timestamp(date)
        sub = grp[[factor_col, "symbol", forward_col]].dropna()
        if len(sub) < 20:
            continue

        q_low = sub[factor_col].quantile(quantile)
        q_high = sub[factor_col].quantile(1 - quantile)
        long_sub = sub[sub[factor_col] >= q_high]
        short_sub = sub[sub[factor_col] <= q_low]
        if len(long_sub) < 3 or len(short_sub) < 3:
            continue

        weights.loc[date, long_sub["symbol"]] = 1.0 / len(long_sub)
        weights.loc[date, short_sub["symbol"]] = -1.0 / len(short_sub)
        long_leg_ret.loc[date] = long_sub[forward_col].mean()
        short_leg_ret.loc[date] = short_sub[forward_col].mean()

    gross_return = (weights * returns_mat).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1) / 2.0
    if len(turnover) > 0:
        turnover.iloc[0] = weights.iloc[0].abs().sum() / 2.0
    cost = turnover * transaction_cost
    net_return = gross_return - cost

    return pd.DataFrame(
        {
            "gross_return": gross_return,
            "turnover": turnover,
            "cost": cost,
            "net_return": net_return,
            "long_leg_ret": long_leg_ret,
            "short_leg_ret": short_leg_ret,
        }
    )


def _safe_sharpe(series: pd.Series) -> float:
    series = series.dropna()
    if len(series) < 20:
        return np.nan
    std = series.std()
    if std <= 0:
        return np.nan
    return float(series.mean() / std * np.sqrt(ANNUAL_TRADING_DAYS))


def _annual_return(series: pd.Series) -> float:
    series = series.dropna()
    if len(series) == 0:
        return np.nan
    return float(series.mean() * ANNUAL_TRADING_DAYS)


def _annual_vol(series: pd.Series) -> float:
    series = series.dropna()
    if len(series) < 2:
        return np.nan
    return float(series.std() * np.sqrt(ANNUAL_TRADING_DAYS))


def _max_drawdown(series: pd.Series) -> float:
    series = series.dropna()
    if len(series) == 0:
        return np.nan
    cum = (1.0 + series).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1.0
    return float(dd.min())


def _profit_loss_ratio(series: pd.Series) -> float:
    series = series.dropna()
    if len(series) == 0:
        return np.nan
    pos = series[series > 0]
    neg = series[series < 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float(pos.mean() / abs(neg.mean()))


def _ic_series(factor_df: pd.DataFrame, factor_col: str, forward_col: str) -> tuple[pd.Series, pd.Series]:
    ic_vals = {}
    rank_ic_vals = {}
    for date, grp in factor_df.groupby("date"):
        sub = grp[[factor_col, forward_col]].dropna()
        if len(sub) < 10:
            continue
        if sub[factor_col].nunique() <= 1 or sub[forward_col].nunique() <= 1:
            continue
        ic_vals[pd.Timestamp(date)] = sub[factor_col].corr(sub[forward_col], method="pearson")
        rank_ic_vals[pd.Timestamp(date)] = sub[factor_col].corr(sub[forward_col], method="spearman")
    return pd.Series(ic_vals).sort_index(), pd.Series(rank_ic_vals).sort_index()


def evaluate_single_factor(
    factor_df: pd.DataFrame,
    factor_col: str,
    forward_col: str = "fwd_ret_1",
    quantile: float = 0.2,
    transaction_cost: float = 0.0003,
) -> tuple[dict, pd.DataFrame]:
    strategy = build_strategy_result(
        factor_df=factor_df,
        factor_col=factor_col,
        forward_col=forward_col,
        quantile=quantile,
        transaction_cost=transaction_cost,
    )
    ic, rank_ic = _ic_series(factor_df, factor_col, forward_col=forward_col)

    gross = strategy["gross_return"]
    net = strategy["net_return"]
    mdd = _max_drawdown(net)
    ann_net = _annual_return(net)
    metrics = {
        "entity": factor_col,
        "annual_return_gross": _annual_return(gross),
        "annual_return_net": ann_net,
        "annual_vol_gross": _annual_vol(gross),
        "annual_vol_net": _annual_vol(net),
        "sharpe_gross": _safe_sharpe(gross),
        "sharpe_net": _safe_sharpe(net),
        "max_drawdown": mdd,
        "calmar": ann_net / abs(mdd) if pd.notna(ann_net) and pd.notna(mdd) and mdd < 0 else np.nan,
        "avg_turnover": float(strategy["turnover"].dropna().mean()) if len(strategy) else np.nan,
        "ic_mean": float(ic.mean()) if len(ic) else np.nan,
        "rank_ic_mean": float(rank_ic.mean()) if len(rank_ic) else np.nan,
        "icir": float(ic.mean() / ic.std()) if len(ic) > 1 and ic.std() > 0 else np.nan,
        "rank_icir": float(rank_ic.mean() / rank_ic.std()) if len(rank_ic) > 1 and rank_ic.std() > 0 else np.nan,
        "long_win_rate": float((strategy["long_leg_ret"] > 0).mean()) if len(strategy) else np.nan,
        "short_win_rate": float((strategy["short_leg_ret"] < 0).mean()) if len(strategy) else np.nan,
        "profit_loss_ratio": _profit_loss_ratio(net),
        "n_obs": int(gross.dropna().shape[0]),
    }
    return metrics, strategy


def evaluate_factor_set(
    factor_df: pd.DataFrame,
    factor_cols: list[str],
    forward_col: str = "fwd_ret_1",
    quantile: float = 0.2,
    transaction_cost: float = 0.0003,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    gross_returns = {}
    net_returns = {}
    turnover_series = {}

    for fac in factor_cols:
        metrics, strategy = evaluate_single_factor(
            factor_df=factor_df,
            factor_col=fac,
            forward_col=forward_col,
            quantile=quantile,
            transaction_cost=transaction_cost,
        )
        rows.append(metrics)
        gross_returns[fac] = strategy["gross_return"]
        net_returns[fac] = strategy["net_return"]
        turnover_series[fac] = strategy["turnover"]

    metrics_df = pd.DataFrame(rows).sort_values("sharpe_gross", ascending=False).reset_index(drop=True)
    returns_df = pd.concat(
        {
            "gross": pd.DataFrame(gross_returns).sort_index(),
            "net": pd.DataFrame(net_returns).sort_index(),
            "turnover": pd.DataFrame(turnover_series).sort_index(),
        },
        axis=1,
    )
    return metrics_df, returns_df


def select_low_corr_high_sharpe(
    train_metrics_df: pd.DataFrame,
    train_returns_df: pd.DataFrame,
    threshold: float = 0.5,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    gross_ret = train_returns_df["gross"]
    gross_ret = gross_ret.loc[:, gross_ret.std(skipna=True) > 0]
    corr_df = gross_ret.corr()

    ranked_df = train_metrics_df.sort_values("sharpe_gross", ascending=False).copy()
    ranked = ranked_df["entity"].tolist()
    selected: list[str] = []
    for fac in ranked:
        if fac not in corr_df.columns:
            continue
        if not selected:
            selected.append(fac)
            continue
        max_corr = max(abs(float(corr_df.loc[fac, s])) for s in selected)
        if max_corr <= threshold:
            selected.append(fac)
    return selected, corr_df, ranked_df


def apply_stability_filter(
    ranked_train_df: pd.DataFrame,
    test_metrics_df: pd.DataFrame,
    min_train_net_sharpe: float = 0.0,
    min_test_gross_sharpe: float = 0.0,
) -> pd.DataFrame:
    test_map = test_metrics_df.set_index("entity")[["sharpe_gross"]].rename(
        columns={"sharpe_gross": "test_sharpe_gross"}
    )
    merged = ranked_train_df.merge(test_map, left_on="entity", right_index=True, how="left")
    stable = merged[
        (merged["sharpe_net"] >= min_train_net_sharpe)
        & (merged["test_sharpe_gross"] >= min_test_gross_sharpe)
    ].copy()
    if stable.empty:
        stable = merged[merged["sharpe_net"] >= min_train_net_sharpe].copy()
    if stable.empty:
        stable = merged.copy()
    return stable.sort_values("sharpe_gross", ascending=False)


def build_factor_weights(
    train_metrics_df: pd.DataFrame,
    selected_factors: list[str],
    max_weight: float = 0.35,
) -> pd.DataFrame:
    if not selected_factors:
        return pd.DataFrame(columns=["factor", "weight", "score"])

    sub = train_metrics_df[train_metrics_df["entity"].isin(selected_factors)].copy()
    sub["score"] = sub["sharpe_net"].clip(lower=0.0).fillna(0.0) + sub["icir"].clip(lower=0.0).fillna(0.0)
    if sub["score"].sum() <= 0:
        sub["score"] = 1.0
    sub["weight"] = sub["score"] / sub["score"].sum()
    sub["weight"] = sub["weight"].clip(upper=max_weight)
    sub["weight"] = sub["weight"] / sub["weight"].sum()
    return sub[["entity", "weight", "score"]].rename(columns={"entity": "factor"}).sort_values("weight", ascending=False)


def evaluate_combo(
    factor_df: pd.DataFrame,
    selected_factors: list[str],
    factor_weights: pd.DataFrame | None = None,
    forward_col: str = "fwd_ret_1",
    quantile: float = 0.2,
    transaction_cost: float = 0.0003,
) -> tuple[dict, pd.DataFrame]:
    if not selected_factors:
        empty = pd.DataFrame(columns=["gross_return", "turnover", "cost", "net_return", "long_leg_ret", "short_leg_ret"])
        return {"entity": "combo_selected"}, empty

    combo_col = "__combo_factor__"
    combo_df = factor_df.copy()
    if factor_weights is None or factor_weights.empty:
        combo_df[combo_col] = combo_df[selected_factors].mean(axis=1)
        weight_map = {f: 1.0 / len(selected_factors) for f in selected_factors}
    else:
        weight_map = dict(zip(factor_weights["factor"], factor_weights["weight"]))
        combo_df[combo_col] = 0.0
        for fac in selected_factors:
            combo_df[combo_col] += combo_df[fac] * weight_map.get(fac, 0.0)
    metrics, strategy = evaluate_single_factor(
        factor_df=combo_df,
        factor_col=combo_col,
        forward_col=forward_col,
        quantile=quantile,
        transaction_cost=transaction_cost,
    )
    metrics["entity"] = "combo_selected"
    metrics["selected_factor_count"] = len(selected_factors)
    metrics["weighting"] = "weighted"
    metrics["weight_max"] = max(weight_map.values()) if weight_map else np.nan
    return metrics, strategy

