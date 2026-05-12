"""补全期货缺失字段（open_interest / oi_change / 合约信息）.

用法示例：
    python get_data.py \
      --input futures_daily_main.csv \
      --output futures_daily_main_enriched.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import akshare as ak
import pandas as pd


OI_ALIASES = [
    "open_interest",
    "oi",
    "hold",
    "position",
    "持仓量",
]
OI_CHANGE_ALIASES = [
    "oi_change",
    "open_interest_change",
    "position_change",
    "持仓量变化",
    "持仓变化",
]
EXPIRY_ALIASES = [
    "contract_expiry",
    "expiry",
    "expire_date",
    "到期日",
]
LISTING_ALIASES = [
    "listing_date",
    "listed_date",
    "上市日期",
]
DATE_ALIASES = ["date", "日期", "trade_date", "交易日期"]


def _normalize_colname(col: str) -> str:
    return col.strip().lower().replace(" ", "_")


def _find_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    norm_map = {_normalize_colname(c): c for c in columns}
    for key in candidates:
        k = _normalize_colname(key)
        if k in norm_map:
            return norm_map[k]
    return None


def _fetch_symbol_from_akshare(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """尽量从 AkShare 拉取品种数据；失败返回空表。"""
    # 优先 futures_main_sina
    try:
        df = ak.futures_main_sina(symbol=symbol.lower(), start_date=start_date, end_date=end_date)
        if not df.empty:
            return df
    except Exception:
        pass

    # 回退 futures_zh_daily_sina（用主连代码）
    try:
        df = ak.futures_zh_daily_sina(symbol=f"{symbol.upper()}0")
        if not df.empty:
            return df
    except Exception:
        pass

    return pd.DataFrame()


def _normalize_akshare_df(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw.empty:
        return raw

    date_col = _find_column(raw.columns, DATE_ALIASES)
    if date_col is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(raw[date_col], errors="coerce")
    out["symbol"] = symbol.upper()

    oi_col = _find_column(raw.columns, OI_ALIASES)
    if oi_col:
        out["open_interest"] = pd.to_numeric(raw[oi_col], errors="coerce")
    else:
        out["open_interest"] = pd.NA

    oi_chg_col = _find_column(raw.columns, OI_CHANGE_ALIASES)
    if oi_chg_col:
        out["oi_change"] = pd.to_numeric(raw[oi_chg_col], errors="coerce")
    else:
        # 若仅有 open_interest，则用差分近似
        out["oi_change"] = pd.to_numeric(out["open_interest"], errors="coerce").diff()

    expiry_col = _find_column(raw.columns, EXPIRY_ALIASES)
    out["contract_expiry"] = pd.to_datetime(raw[expiry_col], errors="coerce") if expiry_col else pd.NaT

    listing_col = _find_column(raw.columns, LISTING_ALIASES)
    out["listing_date"] = pd.to_datetime(raw[listing_col], errors="coerce") if listing_col else pd.NaT

    out = out.dropna(subset=["date"]).drop_duplicates(["date", "symbol"]).sort_values("date")
    return out.reset_index(drop=True)


def enrich_missing_fields(input_csv: Path, output_csv: Path, report_csv: Path) -> None:
    base = pd.read_csv(input_csv)
    base["date"] = pd.to_datetime(base["date"], errors="coerce")
    base["symbol"] = base["symbol"].astype(str).str.upper()

    for col in ["open_interest", "oi_change", "contract_expiry", "listing_date"]:
        if col not in base.columns:
            base[col] = pd.NA

    start_date = base["date"].min().strftime("%Y-%m-%d")
    end_date = base["date"].max().strftime("%Y-%m-%d")
    symbols = sorted(base["symbol"].dropna().unique().tolist())

    enriched_parts = []
    report_rows = []

    for sym in symbols:
        raw = _fetch_symbol_from_akshare(sym, start_date=start_date, end_date=end_date)
        norm = _normalize_akshare_df(raw, sym)
        enriched_parts.append(norm)
        report_rows.append(
            {
                "symbol": sym,
                "rows_from_akshare": int(len(norm)),
                "oi_non_null": int(norm["open_interest"].notna().sum()) if not norm.empty else 0,
                "expiry_non_null": int(norm["contract_expiry"].notna().sum()) if not norm.empty else 0,
                "listing_non_null": int(norm["listing_date"].notna().sum()) if not norm.empty else 0,
            }
        )

    expected_cols = ["date", "symbol", "open_interest", "oi_change", "contract_expiry", "listing_date"]
    non_empty_parts = [x for x in enriched_parts if not x.empty]
    enrich_df = pd.concat(non_empty_parts, ignore_index=True) if non_empty_parts else pd.DataFrame(columns=expected_cols)
    for col in expected_cols:
        if col not in enrich_df.columns:
            enrich_df[col] = pd.NA

    merged = base.merge(
        enrich_df,
        on=["date", "symbol"],
        how="left",
        suffixes=("", "_new"),
    )

    for col in ["open_interest", "oi_change", "contract_expiry", "listing_date"]:
        merged[col] = merged[col].where(merged[col].notna(), merged[f"{col}_new"])
        merged = merged.drop(columns=[f"{col}_new"])

    merged = merged.sort_values(["symbol", "date"]).reset_index(drop=True)
    merged.to_csv(output_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(report_rows).to_csv(report_csv, index=False, encoding="utf-8-sig")

    before_oi = base["open_interest"].notna().sum()
    after_oi = merged["open_interest"].notna().sum()
    print(f"输入文件: {input_csv}")
    print(f"输出文件: {output_csv}")
    print(f"报告文件: {report_csv}")
    print(f"open_interest 非空: {before_oi} -> {after_oi} (新增 {after_oi - before_oi})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich futures missing fields from AkShare.")
    parser.add_argument("--input", type=str, default="futures_daily_main.csv")
    parser.add_argument("--output", type=str, default="futures_daily_main_enriched.csv")
    parser.add_argument("--report", type=str, default="enrich_report.csv")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    input_csv = script_dir / args.input
    output_csv = script_dir / args.output
    report_csv = script_dir / args.report

    enrich_missing_fields(input_csv=input_csv, output_csv=output_csv, report_csv=report_csv)


if __name__ == "__main__":
    main()