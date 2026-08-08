#!/usr/bin/env python3
"""Balanced fund correction-system research module.

This v1 uses the existing NAV cache and production universe to compare
balanced / multi-asset / income funds through a disturbance-recovery lens.
It deliberately separates measured NAV behavior from missing holdings/process
attribution, so the report does not overclaim manager skill.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_DB = "AI_Fund_Research/cache/fund_research.sqlite"
DEFAULT_UNIVERSE = "config/fund_universe.csv"
DEFAULT_OUTPUT_DIR = "AI_Fund_Research/correction_system"
SGD_CASH_CAGR_BENCHMARK = 1.8


EVENT_WINDOWS = [
    ("2018_Q4_Fed_Growth_Scare", "2018-09-20", "2019-04-30", "global_growth"),
    ("2020_Covid_Crash", "2020-01-15", "2020-12-31", "recession_liquidity"),
    ("2022_Inflation_Rate_Shock", "2021-12-01", "2023-12-31", "inflation_rates"),
    ("2023_Banking_Stress", "2023-02-01", "2023-07-31", "credit_confidence"),
    ("2024_2026_Recent_Window", "2024-01-01", "2026-12-31", "recent"),
]

BALANCED_KEYWORDS = [
    "balanced",
    "multi",
    "allocation",
    "income",
    "growth",
    "low volatility",
    "equity income",
    "dividend",
]


@dataclass(frozen=True)
class Paths:
    db: Path
    universe: Path
    output_dir: Path
    charts_dir: Path


def resolve_paths(db: str, universe: str, output_dir: str) -> Paths:
    out = Path(output_dir)
    charts = out / "charts"
    out.mkdir(parents=True, exist_ok=True)
    charts.mkdir(parents=True, exist_ok=True)
    return Paths(Path(db), Path(universe), out, charts)


def load_universe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    for col in ["retirement_candidate", "benchmark"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["true", "yes", "1", "y"])
    return df


def load_nav(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        nav = pd.read_sql_query(
            "SELECT symbol, date, adj_close FROM nav_history ORDER BY symbol, date",
            conn,
            parse_dates=["date"],
        )
    nav["symbol"] = nav["symbol"].astype(str).str.upper().str.strip()
    nav["adj_close"] = pd.to_numeric(nav["adj_close"], errors="coerce")
    return nav.dropna(subset=["date", "adj_close"])


def candidate_universe(universe: pd.DataFrame) -> pd.DataFrame:
    text = (
        universe.get("name", "").astype(str)
        + " "
        + universe.get("type", "").astype(str)
        + " "
        + universe.get("risk_group", "").astype(str)
        + " "
        + universe.get("notes", "").astype(str)
    ).str.lower()
    mask = universe.get("retirement_candidate", True).fillna(False)
    mask &= ~universe.get("benchmark", False).fillna(False)
    mask &= text.apply(lambda value: any(token in value for token in BALANCED_KEYWORDS))
    mask &= (
        universe.get("currency", "").astype(str).str.upper().eq("SGD")
        | universe.get("sgd_hedged", "").astype(str).str.lower().isin(["yes", "true", "1", "y"])
    )
    exclusion_text = text
    mask &= ~exclusion_text.str.contains(
        "money market|enhanced cash|cash plus|ultra-short|ultra short|short duration|short term|treasury|t-bill",
        regex=True,
        na=False,
    )
    return universe[mask].copy()


def add_static_benchmark(nav: pd.DataFrame, universe: pd.DataFrame, equity: str, bond: str, symbol: str, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    eq = nav[nav["symbol"].eq(equity)].set_index("date")["adj_close"].sort_index()
    bd = nav[nav["symbol"].eq(bond)].set_index("date")["adj_close"].sort_index()
    joined = pd.concat([eq.rename("equity"), bd.rename("bond")], axis=1, sort=False).dropna()
    if len(joined) < 60:
        return nav, universe
    rets = joined.pct_change().fillna(0.0)
    synthetic = (1.0 + (0.60 * rets["equity"] + 0.40 * rets["bond"])).cumprod() * 100.0
    bench_nav = pd.DataFrame({"symbol": symbol, "date": synthetic.index, "adj_close": synthetic.values})
    bench_row = {
        "symbol": symbol,
        "name": name,
        "type": "Static 60/40 Benchmark",
        "currency": "SGD proxy",
        "sgd_hedged": "Proxy",
        "fee_pct_seed": 0.0,
        "yield_pct_seed": np.nan,
        "risk_group": "benchmark_6040",
        "retirement_candidate": True,
        "benchmark": True,
        "notes": f"Synthetic daily 60/40 using {equity}/{bond}; proxy only, not a purchasable fund.",
    }
    nav = pd.concat([nav, bench_nav], ignore_index=True, sort=False)
    universe = pd.concat([universe, pd.DataFrame([bench_row])], ignore_index=True, sort=False)
    return nav, universe


def cagr(series: pd.Series) -> float:
    clean = series.dropna()
    if len(clean) < 2 or clean.iloc[0] <= 0:
        return np.nan
    years = max((clean.index[-1] - clean.index[0]).days / 365.25, 1 / 365.25)
    return ((clean.iloc[-1] / clean.iloc[0]) ** (1 / years) - 1) * 100.0


def drawdown(series: pd.Series) -> pd.Series:
    clean = series.dropna()
    return clean / clean.cummax() - 1.0


def max_underwater_days(series: pd.Series) -> int:
    dd = drawdown(series)
    max_run = run = 0
    for value in dd:
        if value < -1e-10:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return int(max_run)


def ulcer_index(series: pd.Series) -> float:
    dd_pct = drawdown(series) * 100.0
    return float(np.sqrt(np.nanmean(np.square(np.minimum(dd_pct, 0.0)))))


def sortino(series: pd.Series) -> float:
    returns = series.dropna().pct_change().dropna()
    if returns.empty:
        return np.nan
    downside = returns[returns < 0].std() * math.sqrt(252)
    ann = returns.mean() * 252
    if not downside or np.isnan(downside):
        return np.nan
    return float(ann / downside)


def recovery_metrics(series: pd.Series) -> dict[str, Any]:
    s = series.dropna()
    if len(s) < 2:
        return {}
    peak = s.cummax()
    dd = s / peak - 1.0
    trough_date = dd.idxmin()
    maxdd = float(dd.loc[trough_date] * 100.0)
    prior = s.loc[:trough_date]
    peak_date = prior.idxmax()
    prior_peak = float(s.loc[peak_date])
    after = s.loc[trough_date:]
    recovered = after[after >= prior_peak]
    recovery_date = pd.NaT if recovered.empty else recovered.index[0]
    trough_value = float(s.loc[trough_date])
    half_level = trough_value + 0.50 * (prior_peak - trough_value)
    eighty_level = trough_value + 0.80 * (prior_peak - trough_value)
    half_hit = after[after >= half_level]
    eighty_hit = after[after >= eighty_level]
    return {
        "peak_date": peak_date.date().isoformat(),
        "trough_date": trough_date.date().isoformat(),
        "recovery_date": "" if pd.isna(recovery_date) else recovery_date.date().isoformat(),
        "max_drawdown_pct": round(maxdd, 2),
        "peak_to_trough_days": int((trough_date - peak_date).days),
        "trough_to_recovery_days": "" if pd.isna(recovery_date) else int((recovery_date - trough_date).days),
        "peak_to_recovery_days": "" if pd.isna(recovery_date) else int((recovery_date - peak_date).days),
        "half_recovery_days": "" if half_hit.empty else int((half_hit.index[0] - trough_date).days),
        "eighty_recovery_days": "" if eighty_hit.empty else int((eighty_hit.index[0] - trough_date).days),
        "still_underwater": bool(recovered.empty),
    }


def forward_return(series: pd.Series, start_date: pd.Timestamp, calendar_days: int) -> float:
    s = series.dropna()
    if start_date not in s.index:
        return np.nan
    future = s[s.index >= start_date + pd.Timedelta(days=calendar_days)]
    if future.empty:
        return np.nan
    return float((future.iloc[0] / s.loc[start_date] - 1.0) * 100.0)


def long_term_metrics(symbol: str, name: str, info: pd.Series, series: pd.Series) -> dict[str, Any]:
    s = series.dropna()
    rec = recovery_metrics(s)
    returns = s.pct_change().dropna()
    annual_vol = returns.std() * math.sqrt(252) * 100.0 if not returns.empty else np.nan
    maxdd = rec.get("max_drawdown_pct", np.nan)
    return {
        "symbol": symbol,
        "fund": name,
        "type": info.get("type", ""),
        "currency": info.get("currency", ""),
        "sgd_hedged": info.get("sgd_hedged", ""),
        "fee_pct": info.get("fee_pct_seed", np.nan),
        "yield_pct": info.get("yield_pct_seed", np.nan),
        "first_date": s.index.min().date().isoformat() if not s.empty else "",
        "last_date": s.index.max().date().isoformat() if not s.empty else "",
        "history_years": round((s.index.max() - s.index.min()).days / 365.25, 2) if len(s) > 1 else np.nan,
        "cagr_pct": round(cagr(s), 2),
        "excess_vs_cash_pct": round(cagr(s) - SGD_CASH_CAGR_BENCHMARK, 2),
        "max_drawdown_pct": maxdd,
        "max_underwater_days": max_underwater_days(s),
        "ulcer_index": round(ulcer_index(s), 2),
        "sortino": round(sortino(s), 2) if not np.isnan(sortino(s)) else np.nan,
        "volatility_pct": round(float(annual_vol), 2) if not np.isnan(annual_vol) else np.nan,
        **rec,
    }


def event_metrics(symbol: str, name: str, series: pd.Series, benchmark: pd.Series | None) -> list[dict[str, Any]]:
    rows = []
    for event, start, end, shock_type in EVENT_WINDOWS:
        window = series.loc[pd.Timestamp(start) : pd.Timestamp(end)].dropna()
        if len(window) < 40:
            continue
        rec = recovery_metrics(window)
        if not rec:
            continue
        trough = pd.Timestamp(rec["trough_date"])
        maxdd = float(rec["max_drawdown_pct"])
        bench_dd = np.nan
        downside_capture = np.nan
        if benchmark is not None:
            b = benchmark.loc[window.index.min() : window.index.max()].dropna()
            if len(b) >= 40:
                bench_dd = float(recovery_metrics(b).get("max_drawdown_pct", np.nan))
                if bench_dd and not np.isnan(bench_dd):
                    downside_capture = maxdd / bench_dd * 100.0
        rows.append(
            {
                "symbol": symbol,
                "fund": name,
                "event": event,
                "shock_type": shock_type,
                **rec,
                "worst_1d_pct": round(window.pct_change().min() * 100.0, 2),
                "worst_5d_pct": round((window / window.shift(5) - 1).min() * 100.0, 2),
                "worst_20d_pct": round((window / window.shift(20) - 1).min() * 100.0, 2),
                "post_trough_1m_pct": round(forward_return(window, trough, 30), 2),
                "post_trough_3m_pct": round(forward_return(window, trough, 91), 2),
                "post_trough_6m_pct": round(forward_return(window, trough, 182), 2),
                "post_trough_12m_pct": round(forward_return(window, trough, 365), 2),
                "benchmark_maxdd_pct": round(bench_dd, 2) if not np.isnan(bench_dd) else np.nan,
                "downside_capture_vs_6040_pct": round(downside_capture, 2) if not np.isnan(downside_capture) else np.nan,
                "attribution_status": "NAV measured; holdings/process attribution not yet collected",
            }
        )
    return rows


def score_rank(metrics: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    out = metrics.copy()
    event_group = events.groupby("symbol") if not events.empty else None
    out["avg_event_maxdd_pct"] = out["symbol"].map(event_group["max_drawdown_pct"].mean()) if event_group is not None else np.nan
    out["avg_downside_capture_pct"] = out["symbol"].map(event_group["downside_capture_vs_6040_pct"].mean()) if event_group is not None else np.nan
    out["avg_post_trough_12m_pct"] = out["symbol"].map(event_group["post_trough_12m_pct"].mean()) if event_group is not None else np.nan

    def norm_low(value: pd.Series) -> pd.Series:
        v = value.astype(float)
        return ((v.max() - v) / (v.max() - v.min())).replace([np.inf, -np.inf], np.nan).fillna(0.5)

    def norm_high(value: pd.Series) -> pd.Series:
        v = value.astype(float)
        return ((v - v.min()) / (v.max() - v.min())).replace([np.inf, -np.inf], np.nan).fillna(0.5)

    dd_score = norm_high(out["max_drawdown_pct"]) * 15
    underwater_score = norm_low(out["max_underwater_days"]) * 15
    recovery_days = pd.to_numeric(out["peak_to_recovery_days"].replace("", np.nan), errors="coerce").fillna(out["max_underwater_days"].max() * 1.5)
    recovery_score = norm_low(recovery_days) * 15
    capture_score = norm_low(out["avg_downside_capture_pct"].fillna(out["avg_downside_capture_pct"].median())) * 10
    post_score = norm_high(out["avg_post_trough_12m_pct"].fillna(out["avg_post_trough_12m_pct"].median())) * 10
    stability_score = norm_low(out["ulcer_index"]) * 10
    real_score = norm_high(out["excess_vs_cash_pct"]) * 10
    evidence_score = 3.0  # v1 has measured NAV evidence but not full holdings/process attribution.
    fee_score = norm_low(pd.to_numeric(out["fee_pct"], errors="coerce").fillna(1.5)) * 5
    out["active_risk_control_evidence_score"] = evidence_score
    raw_score = (
        dd_score
        + underwater_score
        + recovery_score
        + capture_score
        + post_score
        + stability_score
        + real_score
        + evidence_score
        + fee_score
    )
    history_penalty = np.select(
        [out["history_years"] < 3, out["history_years"] < 5],
        [20, 10],
        default=0,
    )
    out["raw_correction_system_score"] = raw_score.round(0).astype(int)
    out["history_reliability_penalty"] = history_penalty
    out["correction_system_score"] = (raw_score - history_penalty).clip(lower=0).round(0).astype(int)
    out["data_quality_note"] = np.where(
        out["history_years"] < 5,
        "Short share-class history; do not treat as full-cycle evidence",
        "NAV history >=5Y where available; holdings/process attribution pending",
    )
    return out.sort_values("correction_system_score", ascending=False)


def write_placeholder_tables(paths: Paths, universe: pd.DataFrame) -> None:
    cols = [
        "symbol",
        "fund",
        "source_url",
        "publication_date",
        "evidence_type",
        "finding",
        "confidence",
        "missing_next_step",
    ]
    rows = []
    for _, row in universe.iterrows():
        rows.append(
            {
                "symbol": row["symbol"],
                "fund": row.get("name", row["symbol"]),
                "source_url": "",
                "publication_date": "",
                "evidence_type": "manager_process_holdings",
                "finding": "Not collected in v1. Use official factsheets/prospectus before attributing recovery to manager skill.",
                "confidence": "missing",
                "missing_next_step": "Collect official factsheet, prospectus, annual report, portfolio allocation history, fee/distribution policy.",
            }
        )
    pd.DataFrame(rows, columns=cols).to_csv(paths.output_dir / "manager_process_evidence.csv", index=False)
    pd.DataFrame(rows, columns=cols).to_csv(paths.output_dir / "portfolio_history.csv", index=False)
    pd.DataFrame(rows, columns=cols).to_csv(paths.output_dir / "platform_availability.csv", index=False)
    pd.DataFrame(rows, columns=cols).to_csv(paths.output_dir / "fees_distribution.csv", index=False)


def write_charts(paths: Paths, ranking: pd.DataFrame, nav_daily: pd.DataFrame) -> list[str]:
    if ranking.empty or nav_daily.empty:
        return []
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    chart_files: list[str] = []
    top_symbols = ranking.head(8)["symbol"].astype(str).tolist()
    nav = nav_daily[nav_daily["symbol"].isin(top_symbols)].copy()
    nav["date"] = pd.to_datetime(nav["date"])
    pivot = nav.pivot_table(index="date", columns="symbol", values="total_return_nav", aggfunc="last").sort_index()
    dd = pivot / pivot.cummax() - 1.0

    plt.figure(figsize=(12, 7))
    for symbol in dd.columns:
        plt.plot(dd.index, dd[symbol] * 100.0, label=symbol, linewidth=1.4)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Balanced / Multi-Asset Drawdown Comparison")
    plt.ylabel("Drawdown %")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out = paths.charts_dir / "drawdown_comparison.png"
    plt.savefig(out, dpi=160)
    plt.close()
    chart_files.append(str(out))

    window = dd.loc[pd.Timestamp("2021-12-01") : pd.Timestamp("2023-12-31")]
    if not window.empty:
        plt.figure(figsize=(12, 7))
        for symbol in window.columns:
            plt.plot(window.index, window[symbol] * 100.0, label=symbol, linewidth=1.4)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.title("2022 Inflation / Rate Shock Drawdown")
        plt.ylabel("Drawdown %")
        plt.legend(fontsize=8)
        plt.tight_layout()
        out = paths.charts_dir / "2022_event_drawdown.png"
        plt.savefig(out, dpi=160)
        plt.close()
        chart_files.append(str(out))
    return chart_files


def markdown_report(ranking: pd.DataFrame, events: pd.DataFrame, output_files: list[str], chart_files: list[str]) -> str:
    lines = [
        "# Balanced Fund Correction-System Report",
        "",
        "## Executive Summary",
        "",
        "This v1 is NAV-measured only. It compares disturbance size, underwater time, recovery speed, and post-trough rebound. Holdings/process attribution is explicitly marked missing until official documents are collected.",
        "",
    ]
    if ranking.empty:
        lines.append("No balanced / multi-asset candidates found in the NAV cache.")
    else:
        top = ranking.iloc[0]
        lines.extend(
            [
                f"- Best preliminary controller by NAV metrics: `{top['fund']}`.",
                "- Do not treat this as final manager-skill proof; v1 has not yet attributed returns to duration, credit, equity rotation, FX hedge, or cash positioning.",
                "- Short-history funds are allowed into the table but flagged in `data_quality_note`.",
                "",
                "## Main Table",
                "",
            ]
        )
        cols = [
            "fund",
            "symbol",
            "history_years",
            "cagr_pct",
            "max_drawdown_pct",
            "max_underwater_days",
            "peak_to_recovery_days",
            "ulcer_index",
            "avg_downside_capture_pct",
            "avg_post_trough_12m_pct",
            "fee_pct",
            "yield_pct",
            "correction_system_score",
            "data_quality_note",
        ]
        display = ranking[cols].head(15).where(pd.notna(ranking[cols].head(15)), "NA")
        lines.append("| " + " | ".join(display.columns) + " |")
        lines.append("| " + " | ".join("---" for _ in display.columns) + " |")
        for record in display.to_dict("records"):
            lines.append("| " + " | ".join(str(record[col]) for col in display.columns) + " |")
    if not events.empty:
        lines.extend(["", "## Shock Table", ""])
        shock = events.pivot_table(index=["fund", "symbol"], columns="event", values="max_drawdown_pct", aggfunc="min").reset_index()
        shock = shock.where(pd.notna(shock), "NA")
        lines.append("| " + " | ".join(shock.columns.astype(str)) + " |")
        lines.append("| " + " | ".join("---" for _ in shock.columns) + " |")
        for record in shock.to_dict("records"):
            lines.append("| " + " | ".join(str(record[col]) for col in shock.columns) + " |")
    lines.extend(
        [
            "",
            "## Controller Assessment",
            "",
            "- Sensors: not yet measured; official manager commentary and factsheets required.",
            "- Actuators: not yet measured; need equity weight, duration, credit quality, HY weight, cash, FX hedging history.",
            "- Feedback/feed-forward: not yet measured; use official investment process documents.",
            "- Current measured evidence: NAV drawdown, recovery, downside capture versus static 60/40 proxy, and post-trough returns.",
            "",
            "## Output Files",
            "",
        ]
    )
    for file in output_files:
        lines.append(f"- `{file}`")
    if chart_files:
        lines.extend(["", "## Charts", ""])
        for file in chart_files:
            lines.append(f"- `{file}`")
    lines.extend(
        [
            "",
            "## Next Data Tasks",
            "",
            "- Verify P-BIG 29/30 Nov 2023 strategy change using official PIMCO documents.",
            "- Separate predecessor stress evidence from current P-BIG live evidence.",
            "- Collect official HSBC Life Singapore Balanced allocation and underlying Schroder fund exposures.",
            "- Build total-return NAV with distributions reinvested for distributing classes.",
            "- Add official holdings/process evidence before assigning a high active-risk-control score.",
            "",
        ]
    )
    return "\n".join(lines)


def gpt_handoff(output_files: list[str]) -> str:
    return "\n".join(
        [
            "# GPT Read This: Correction-System Research",
            "",
            "Start with `balanced_fund_correction_system_report.md`, then inspect `correction_system_ranking.csv` and `drawdown_events.csv`.",
            "",
            "This is a v1 NAV-measured package. Treat manager skill attribution as incomplete unless `manager_process_evidence.csv` contains official evidence.",
            "",
            "Key rule: do not compare price NAV versus total-return NAV if distribution data is missing. The current v1 uses adjusted NAV from Yahoo cache.",
            "",
            "Files:",
            *[f"- `{file}`" for file in output_files],
            "",
            "Interpretation cautions:",
            "- Short share-class history is flagged in `data_quality_note`.",
            "- P-BIG strategy-change evidence must be verified before using pre-change history as current-strategy proof.",
            "- Static 60/40 is a proxy benchmark, not a perfect SGD-hedged passive product.",
            "- Good recovery is measured behavior first, not automatically manager skill.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Balanced fund correction-system research")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    paths = resolve_paths(args.db, args.universe, args.output_dir)
    universe = load_universe(paths.universe)
    nav = load_nav(paths.db)
    nav, universe = add_static_benchmark(nav, universe, "SPY", "A35.SI", "STATIC_60_40_SPY_A35", "Static 60/40 SPY + A35.SI Proxy")
    nav, universe = add_static_benchmark(nav, universe, "ES3.SI", "A35.SI", "STATIC_60_40_ES3_A35", "Static Singapore 60/40 ES3.SI + A35.SI Proxy")

    candidates = candidate_universe(universe)
    # Keep synthetic benchmarks in the comparison table.
    candidates = pd.concat([candidates, universe[universe["symbol"].astype(str).str.startswith("STATIC_60_40_")]], ignore_index=True, sort=False)
    series_by_symbol = {
        sym: df.set_index("date")["adj_close"].sort_index()
        for sym, df in nav[nav["symbol"].isin(candidates["symbol"])].groupby("symbol")
    }
    benchmark = series_by_symbol.get("STATIC_60_40_SPY_A35")

    master_rows = []
    event_rows = []
    nav_rows = []
    for _, info in candidates.iterrows():
        symbol = str(info["symbol"]).upper()
        if symbol not in series_by_symbol:
            continue
        series = series_by_symbol[symbol]
        name = str(info.get("name", symbol))
        master_rows.append(long_term_metrics(symbol, name, info, series))
        event_rows.extend(event_metrics(symbol, name, series, benchmark))
        nav_rows.extend(
            {
                "date": idx.date().isoformat(),
                "symbol": symbol,
                "nav": float(value),
                "distribution": np.nan,
                "total_return_nav": float(value),
                "currency": info.get("currency", ""),
                "source": "fund_research.sqlite/yahoo_adjusted_close",
                "source_date": "",
                "quality_flag": "adjusted_nav_proxy_distribution_detail_missing",
            }
            for idx, value in series.items()
        )

    fund_master = pd.DataFrame(master_rows)
    drawdown_events = pd.DataFrame(event_rows)
    ranking = score_rank(fund_master, drawdown_events)
    nav_daily = pd.DataFrame(nav_rows)

    output_files = [
        "fund_master.csv",
        "nav_daily.csv",
        "drawdown_events.csv",
        "recovery_metrics.csv",
        "rolling_returns.csv",
        "portfolio_history.csv",
        "fees_distribution.csv",
        "platform_availability.csv",
        "manager_process_evidence.csv",
        "correction_system_ranking.csv",
        "balanced_fund_correction_system_report.md",
        "GPT_READ_THIS_CORRECTION_SYSTEM_RESEARCH.md",
    ]
    fund_master.to_csv(paths.output_dir / "fund_master.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    nav_daily.to_csv(paths.output_dir / "nav_daily.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    drawdown_events.to_csv(paths.output_dir / "drawdown_events.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    fund_master.to_csv(paths.output_dir / "recovery_metrics.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    rolling = fund_master[["symbol", "fund", "history_years", "cagr_pct", "max_drawdown_pct", "ulcer_index", "sortino"]].copy()
    rolling["note"] = "v1 summary only; rolling 1Y/3Y/5Y time series to be expanded after total-return NAV validation"
    rolling.to_csv(paths.output_dir / "rolling_returns.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    ranking.to_csv(paths.output_dir / "correction_system_ranking.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    write_placeholder_tables(paths, candidates)
    chart_files = write_charts(paths, ranking, nav_daily)
    (paths.output_dir / "balanced_fund_correction_system_report.md").write_text(
        markdown_report(ranking, drawdown_events, output_files, chart_files),
        encoding="utf-8",
    )
    (paths.output_dir / "GPT_READ_THIS_CORRECTION_SYSTEM_RESEARCH.md").write_text(
        gpt_handoff(output_files),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "message": "correction_system_research_finished",
                "output_dir": str(paths.output_dir),
                "funds": int(len(fund_master)),
                "events": int(len(drawdown_events)),
                "top": "" if ranking.empty else ranking.iloc[0]["fund"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
