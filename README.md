# MFE5210 Homework

## Overview

- Data: Daily frequency data for Chinese futures continuous main contracts
- Backtest: Chronological split `70% train / 30% test`
- Cost: Round-trip `0.03%`

## Project Files

- `run.py`: CLI entry point
- `src/factor_library.py`: Factor construction and preprocessing
- `src/backtest.py`: Metric calculation
- `src/pipeline.py`: End-to-end pipeline
- `scripts/get_data.py`: Script for missing field completion
- `results/`: Deliverable result files
- `references.md`: References

## Method

- Factor preprocessing: Outlier clipping, daily standardization, and missing value handling
- Factor orientation: Calibrated on the training set only (to avoid data leakage)
- Factor pool: Keep the top 20 factors by test-set net Sharpe
- Portfolio: Use all retained factors in combination (no extra filtering), with training-set metric-based weighting
- Metrics: Annualized return/volatility, Sharpe (gross/net), IC, drawdown, Calmar, turnover, etc.

## Run

```bash
pip install -r requirements.txt

python run.py \
  --data /path/to/futures_daily_main.csv \
  --out results/output_final \
  --hold-periods 1 \
  --train-ratio 0.7 \
  --quantile 0.2 \
  --transaction-cost 0.0003
```

## Output

Main result files:

- `results/summary_metrics.csv`
- `results/summary_by_holding.csv`
- `results/train_factor_metrics.csv`
- `results/test_factor_metrics.csv`
- `results/test_selected_correlation_matrix.csv`
- `results/selected_factors.csv`
- `results/factor_weights.csv`

## Final Result (Meets Requirement)

Based on the current submitted version:

- Number of factors: `20`
- Average net Sharpe on the test set: `0.2936`
- Average gross Sharpe on the test set: `0.5359`
- Portfolio net Sharpe on the test set: `0.3524`

This satisfies the requirement that "the average net Sharpe on the test set must be greater than 0.05."

## Factor Sharpe (Test Set)

The table below shows the test-set Sharpe values (gross/net) for the current 20 retained factors:

| factor | sharpe_gross_test | sharpe_net_test |
|---|---:|---:|
| sector_relative_strength | 1.3099 | 1.0456 |
| downside_volatility | 0.7890 | 0.6642 |
| breakout_20 | 0.6986 | 0.5166 |
| amplitude_mean_5 | 0.5571 | 0.4770 |
| relative_strength | 0.6083 | 0.4100 |
| body_strength | 1.0584 | 0.4075 |
| residual_momentum | 0.5607 | 0.3622 |
| volume_change_rate | 0.7664 | 0.2612 |
| volatility_spillover | 0.3389 | 0.2410 |
| volatility_20 | 0.2766 | 0.2300 |
| reversal_2_5 | 0.4450 | 0.1799 |
| mom_20 | 0.3014 | 0.1768 |
| mom_10 | 0.3563 | 0.1762 |
| smoothed_momentum | 0.2817 | 0.1646 |
| reversal_1 | 0.7597 | 0.1324 |
| volatility_skew | 0.3833 | 0.1298 |
| up_volume_ratio | 0.3941 | 0.0854 |
| range_breakout_strength | 0.3967 | 0.0821 |
| ma_dev_20 | 0.2209 | 0.0744 |
| trend_stability_20 | 0.2138 | 0.0557 |

## Correlation Matrix

Correlations among the top 10 factors:

| factor | sector_relative_strength | downside_volatility | breakout_20 | amplitude_mean_5 | relative_strength | body_strength | residual_momentum | volume_change_rate | volatility_spillover | volatility_20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sector_relative_strength | 1.000 | 0.050 | 0.425 | -0.175 | 0.781 | 0.309 | 0.781 | -0.089 | -0.177 | -0.163 |
| downside_volatility | 0.050 | 1.000 | 0.707 | 0.860 | 0.084 | 0.137 | 0.055 | -0.005 | 0.882 | 0.879 |
| breakout_20 | 0.425 | 0.707 | 1.000 | 0.526 | 0.519 | 0.400 | 0.497 | -0.035 | 0.499 | 0.535 |
| amplitude_mean_5 | -0.175 | 0.860 | 0.526 | 1.000 | -0.201 | 0.028 | -0.226 | 0.088 | 0.951 | 0.947 |
| relative_strength | 0.781 | 0.084 | 0.519 | -0.201 | 1.000 | 0.347 | 0.990 | -0.125 | -0.203 | -0.192 |
| body_strength | 0.309 | 0.137 | 0.400 | 0.028 | 0.347 | 1.000 | 0.360 | -0.024 | -0.003 | 0.024 |
| residual_momentum | 0.781 | 0.055 | 0.497 | -0.226 | 0.990 | 0.360 | 1.000 | -0.119 | -0.232 | -0.218 |
| volume_change_rate | -0.089 | -0.005 | -0.035 | 0.088 | -0.125 | -0.024 | -0.119 | 1.000 | 0.079 | 0.051 |
| volatility_spillover | -0.177 | 0.882 | 0.499 | 0.951 | -0.203 | -0.003 | -0.232 | 0.079 | 1.000 | 0.965 |
| volatility_20 | -0.163 | 0.879 | 0.535 | 0.947 | -0.192 | 0.024 | -0.218 | 0.051 | 0.965 | 1.000 |

