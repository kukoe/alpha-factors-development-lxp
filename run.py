from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.pipeline import run_pipeline
else:
    from .src.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run futures alpha factor pipeline.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/futures_daily_main.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/output",
        help="Output directory.",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.3,
        help="Max pairwise correlation threshold for selected factors.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio by chronological split.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.2,
        help="Cross-sectional quantile for long/short buckets.",
    )
    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.0003,
        help="Round-trip transaction cost (e.g. 0.0003 = 0.03%).",
    )
    parser.add_argument(
        "--hold-periods",
        type=str,
        default="1,3,5",
        help="Comma-separated holding periods, e.g. 1,3,5",
    )
    parser.add_argument(
        "--min-train-net-sharpe",
        type=float,
        default=0.0,
        help="Stability filter: minimum train net Sharpe.",
    )
    parser.add_argument(
        "--min-test-gross-sharpe",
        type=float,
        default=0.0,
        help="Stability filter: minimum test gross Sharpe.",
    )
    parser.add_argument(
        "--max-factor-weight",
        type=float,
        default=0.35,
        help="Maximum weight per factor in weighted combo.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    hold_periods = [int(x.strip()) for x in args.hold_periods.split(",") if x.strip()]
    paths = run_pipeline(
        data_path=Path(args.data),
        output_dir=Path(args.out),
        corr_threshold=args.corr_threshold,
        train_ratio=args.train_ratio,
        quantile=args.quantile,
        transaction_cost=args.transaction_cost,
        hold_periods=hold_periods,
        min_train_net_sharpe=args.min_train_net_sharpe,
        min_test_gross_sharpe=args.min_test_gross_sharpe,
        max_factor_weight=args.max_factor_weight,
    )
    print("Pipeline completed. Output files:")
    for key, path in paths.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()

