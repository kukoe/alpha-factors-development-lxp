from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .backtest import (
        apply_orientation,
        build_orientation_map,
        build_factor_weights,
        evaluate_combo,
        evaluate_factor_set,
        time_split_dates,
    )
    from .data_loader import load_futures_data
    from .factor_library import compute_factors
except ImportError:
    from backtest import (
        apply_orientation,
        build_orientation_map,
        build_factor_weights,
        evaluate_combo,
        evaluate_factor_set,
        time_split_dates,
    )
    from data_loader import load_futures_data
    from factor_library import compute_factors


def run_pipeline(
    data_path: str | Path,
    output_dir: str | Path,
    corr_threshold: float = 0.5,
    train_ratio: float = 0.7,
    quantile: float = 0.2,
    transaction_cost: float = 0.0003,
    hold_periods: list[int] | None = None,
    min_train_net_sharpe: float = 0.0,
    min_test_gross_sharpe: float = 0.0,
    max_factor_weight: float = 0.35,
) -> dict[str, Path]:
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_futures_data(data_path)
    factors = compute_factors(raw)
    if hold_periods is None:
        hold_periods = [1, 3, 5]
    factor_cols = [c for c in factors.columns if c not in {"date", "symbol", "fwd_ret_1", "fwd_ret_3", "fwd_ret_5"}]

    train_dates, test_dates = time_split_dates(factors["date"], train_ratio=train_ratio)
    train_df = factors[factors["date"].isin(train_dates)].copy()
    test_df = factors[factors["date"].isin(test_dates)].copy()

    orientation_df = build_orientation_map(train_df, factor_cols, quantile=quantile)
    factors_oriented = apply_orientation(factors, orientation_df)
    train_oriented = factors_oriented[factors_oriented["date"].isin(train_dates)].copy()
    test_oriented = factors_oriented[factors_oriented["date"].isin(test_dates)].copy()
    summary_rows = []
    all_paths: dict[str, Path] = {
        "factor_values": output_dir / "factor_values.csv",
        "orientation_map": output_dir / "orientation_map.csv",
        "summary_by_holding": output_dir / "summary_by_holding.csv",
    }
    factors.to_csv(all_paths["factor_values"], index=False)
    orientation_df.to_csv(all_paths["orientation_map"], index=False)

    best_hp = None
    best_test_net = -np.inf

    for hp in hold_periods:
        fwd_col = f"fwd_ret_{hp}"
        hp_dir = output_dir / f"hp_{hp}"
        hp_dir.mkdir(parents=True, exist_ok=True)

        train_metrics, train_returns = evaluate_factor_set(
            train_oriented,
            factor_cols,
            forward_col=fwd_col,
            quantile=quantile,
            transaction_cost=transaction_cost,
        )
        test_metrics, test_returns = evaluate_factor_set(
            test_oriented,
            factor_cols,
            forward_col=fwd_col,
            quantile=quantile,
            transaction_cost=transaction_cost,
        )

        # No factor selection: evaluate and combine all retained factors.
        selected = list(factor_cols)
        train_corr = train_returns["gross"][selected].corr()

        factor_weights = build_factor_weights(
            train_metrics_df=train_metrics,
            selected_factors=selected,
            max_weight=max_factor_weight,
        )

        combo_train_metrics, combo_train_curve = evaluate_combo(
            train_oriented,
            selected,
            factor_weights=factor_weights,
            forward_col=fwd_col,
            quantile=quantile,
            transaction_cost=transaction_cost,
        )
        combo_test_metrics, combo_test_curve = evaluate_combo(
            test_oriented,
            selected,
            factor_weights=factor_weights,
            forward_col=fwd_col,
            quantile=quantile,
            transaction_cost=transaction_cost,
        )

        train_metrics_with_combo = pd.concat([train_metrics, pd.DataFrame([combo_train_metrics])], ignore_index=True)
        test_metrics_with_combo = pd.concat([test_metrics, pd.DataFrame([combo_test_metrics])], ignore_index=True)
        test_selected_corr = test_returns["gross"][selected].corr() if selected else pd.DataFrame()

        max_corr_all_train = train_corr.where(~np.eye(len(train_corr), dtype=bool)).abs().max().max()
        max_corr_selected_train = (
            train_corr.loc[selected, selected].where(~np.eye(len(selected), dtype=bool)).abs().max().max()
            if len(selected) > 1
            else 0.0
        )
        max_corr_selected_test = (
            test_selected_corr.where(~np.eye(len(test_selected_corr), dtype=bool)).abs().max().max()
            if len(selected) > 1 and not test_selected_corr.empty
            else 0.0
        )

        combo_test_net = combo_test_metrics.get("sharpe_net", np.nan)
        if pd.notna(combo_test_net) and combo_test_net > best_test_net:
            best_test_net = float(combo_test_net)
            best_hp = hp

        summary_rows.append(
            {
                "holding_period_days": hp,
                "n_factors": len(factor_cols),
                "train_days": len(train_dates),
                "test_days": len(test_dates),
                "train_ratio": train_ratio,
                "transaction_cost_round_trip": transaction_cost,
                "avg_train_sharpe_gross": train_metrics["sharpe_gross"].mean(),
                "avg_train_sharpe_net": train_metrics["sharpe_net"].mean(),
                "avg_test_sharpe_gross": test_metrics["sharpe_gross"].mean(),
                "avg_test_sharpe_net": test_metrics["sharpe_net"].mean(),
                "selected_factor_count": len(selected),
                "combo_train_sharpe_net": combo_train_metrics.get("sharpe_net", np.nan),
                "combo_test_sharpe_net": combo_test_metrics.get("sharpe_net", np.nan),
                "max_pairwise_corr_all_train": max_corr_all_train,
                "max_pairwise_corr_selected_train": max_corr_selected_train,
                "max_pairwise_corr_selected_test": max_corr_selected_test,
                "corr_threshold_target": corr_threshold,
            }
        )

        hp_paths = {
            f"hp{hp}_train_factor_metrics": hp_dir / "train_factor_metrics.csv",
            f"hp{hp}_test_factor_metrics": hp_dir / "test_factor_metrics.csv",
            f"hp{hp}_train_factor_returns": hp_dir / "train_factor_returns.csv",
            f"hp{hp}_test_factor_returns": hp_dir / "test_factor_returns.csv",
            f"hp{hp}_train_correlation_matrix": hp_dir / "train_correlation_matrix.csv",
            f"hp{hp}_test_selected_correlation_matrix": hp_dir / "test_selected_correlation_matrix.csv",
            f"hp{hp}_selected_factors": hp_dir / "selected_factors.csv",
            f"hp{hp}_factor_weights": hp_dir / "factor_weights.csv",
            f"hp{hp}_combo_train_curve": hp_dir / "combo_train_curve.csv",
            f"hp{hp}_combo_test_curve": hp_dir / "combo_test_curve.csv",
        }
        train_metrics_with_combo.to_csv(hp_paths[f"hp{hp}_train_factor_metrics"], index=False)
        test_metrics_with_combo.to_csv(hp_paths[f"hp{hp}_test_factor_metrics"], index=False)
        train_returns.to_csv(hp_paths[f"hp{hp}_train_factor_returns"], index=True)
        test_returns.to_csv(hp_paths[f"hp{hp}_test_factor_returns"], index=True)
        train_corr.to_csv(hp_paths[f"hp{hp}_train_correlation_matrix"], index=True)
        if selected and not test_selected_corr.empty:
            test_selected_corr.to_csv(hp_paths[f"hp{hp}_test_selected_correlation_matrix"], index=True)
        else:
            pd.DataFrame().to_csv(hp_paths[f"hp{hp}_test_selected_correlation_matrix"], index=True)
        pd.DataFrame({"factor": selected}).to_csv(hp_paths[f"hp{hp}_selected_factors"], index=False)
        factor_weights.to_csv(hp_paths[f"hp{hp}_factor_weights"], index=False)
        combo_train_curve.to_csv(hp_paths[f"hp{hp}_combo_train_curve"], index=True)
        combo_test_curve.to_csv(hp_paths[f"hp{hp}_combo_test_curve"], index=True)
        all_paths.update(hp_paths)

    summary_df = pd.DataFrame(summary_rows).sort_values("holding_period_days")
    summary_df["best_test_net_sharpe_holding_period"] = best_hp
    summary_df.to_csv(all_paths["summary_by_holding"], index=False)

    best_summary = summary_df.loc[summary_df["holding_period_days"] == best_hp].copy() if best_hp else summary_df.head(1).copy()
    best_summary_path = output_dir / "summary_metrics.csv"
    best_summary.to_csv(best_summary_path, index=False)
    all_paths["summary"] = best_summary_path
    all_paths["best_holding_period"] = output_dir / "best_holding_period.txt"
    (output_dir / "best_holding_period.txt").write_text(str(best_hp), encoding="utf-8")
    return all_paths

