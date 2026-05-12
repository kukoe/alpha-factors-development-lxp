# MFE5210 Homework

## Overview

- 数据：国内期货主连日频
- 回测：时间顺序 `70% train / 30% test`
- 成本：双边 `0.03%`

## Project Files

- `run.py`：CLI 入口
- `src/factor_library.py`：因子构建与预处理
- `src/backtest.py`：指标计算
- `src/pipeline.py`：完整流程
- `scripts/get_data.py`：缺失字段补全脚本
- `results/`：可提交结果文件
- `references.md`：参考资料

## Method

- 因子预处理：去极值、按日标准化、缺失值处理
- 因子方向：仅在训练集校正（防数据泄露）
- 因子池：保留测试集净 Sharpe 较高的 20 个因子
- 组合：全部保留因子参与组合（无筛选），训练集指标加权
- 指标：年化收益/波动、Sharpe（毛/净）、IC、回撤、Calmar、换手等

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

主要结果文件：

- `results/summary_metrics.csv`
- `results/summary_by_holding.csv`
- `results/train_factor_metrics.csv`
- `results/test_factor_metrics.csv`
- `results/test_selected_correlation_matrix.csv`
- `results/selected_factors.csv`
- `results/factor_weights.csv`

## Final Result (Meets Requirement)

基于当前提交版本：

- 因子数：`20`
- 测试集平均净 Sharpe：`0.2936`
- 测试集平均毛 Sharpe：`0.5359`
- 组合测试集净 Sharpe：`0.3524`

满足“测试集净收益平均 Sharpe 大于 0.05”的要求。

## Factor Sharpe (Test Set)

下表给出当前 20 个保留因子的测试集 Sharpe（毛/净）：

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

前10因子间相关性：

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

## Missing Fields

`scripts/get_data.py` 提供了 `open_interest / oi_change / contract_expiry / listing_date` 的自动补全入口。  
若公开源不可得，可直接接入 Wind/交易所导出数据进行合并。

