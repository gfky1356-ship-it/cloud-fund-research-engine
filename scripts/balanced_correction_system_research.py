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


def trading_days_between(series: pd.Series, start_date: pd.Timestamp, end_date: pd.Timestamp | str) -> float:
    if end_date == "" or pd.isna(end_date):
        return np.nan
    s = series.dropna()
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if start not in s.index or end not in s.index:
        return np.nan
    return float(s.index.get_loc(end) - s.index.get_loc(start))


def threshold_recovery_days(series: pd.Series, trough_date: pd.Timestamp, prior_peak: float, fraction: float) -> tuple[float, float]:
    s = series.dropna()
    if trough_date not in s.index:
        return np.nan, np.nan
    trough_value = float(s.loc[trough_date])
    threshold = trough_value + fraction * (prior_peak - trough_value)
    after = s.loc[trough_date:]
    hit = after[after >= threshold]
    if hit.empty:
        return np.nan, np.nan
    hit_date = hit.index[0]
    return float((hit_date - trough_date).days), trading_days_between(s, trough_date, hit_date)


def settling_metrics(series: pd.Series, recovery_date: str, prior_peak: float) -> dict[str, Any]:
    if not recovery_date:
        return {
            "post_recovery_relapse_30d_pct": np.nan,
            "post_recovery_relapse_60d_pct": np.nan,
            "post_recovery_relapse_90d_pct": np.nan,
            "post_recovery_vol_90d_pct": np.nan,
            "settled_20_trading_days": "",
        }
    s = series.dropna()
    rec = pd.Timestamp(recovery_date)
    if rec not in s.index:
        return {}
    out: dict[str, Any] = {}
    for days in [30, 60, 90]:
        post = s.loc[rec : rec + pd.Timedelta(days=days)]
        out[f"post_recovery_relapse_{days}d_pct"] = round(float((post / prior_peak - 1.0).min() * 100.0), 2) if not post.empty else np.nan
    post90 = s.loc[rec : rec + pd.Timedelta(days=90)]
    ret90 = post90.pct_change().dropna()
    out["post_recovery_vol_90d_pct"] = round(float(ret90.std() * math.sqrt(252) * 100.0), 2) if not ret90.empty else np.nan
    first20 = s.loc[rec:].head(20)
    out["settled_20_trading_days"] = "" if len(first20) < 20 else bool((first20 >= prior_peak * 0.98).all())
    return out


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
        peak = pd.Timestamp(rec["peak_date"])
        maxdd = float(rec["max_drawdown_pct"])
        prior_peak = float(window.loc[peak])
        days25, trading25 = threshold_recovery_days(window, trough, prior_peak, 0.25)
        days50, trading50 = threshold_recovery_days(window, trough, prior_peak, 0.50)
        days80, trading80 = threshold_recovery_days(window, trough, prior_peak, 0.80)
        full_days = rec.get("trough_to_recovery_days", "")
        full_days_num = pd.to_numeric(pd.Series([full_days]), errors="coerce").iloc[0]
        bench_dd = np.nan
        bench_recovery_days = np.nan
        benchmark_post_3m = np.nan
        benchmark_post_6m = np.nan
        benchmark_post_12m = np.nan
        downside_capture = np.nan
        if benchmark is not None:
            b = benchmark.loc[window.index.min() : window.index.max()].dropna()
            if len(b) >= 40:
                b_rec = recovery_metrics(b)
                bench_dd = float(b_rec.get("max_drawdown_pct", np.nan))
                bench_recovery_days = pd.to_numeric(pd.Series([b_rec.get("trough_to_recovery_days", np.nan)]), errors="coerce").iloc[0]
                if bench_dd and not np.isnan(bench_dd):
                    downside_capture = maxdd / bench_dd * 100.0
                benchmark_post_3m = forward_return(b, trough, 91)
                benchmark_post_6m = forward_return(b, trough, 182)
                benchmark_post_12m = forward_return(b, trough, 365)
        post3 = forward_return(window, trough, 91)
        post6 = forward_return(window, trough, 182)
        post12 = forward_return(window, trough, 365)
        recovery_date = rec.get("recovery_date", "")
        from_recovery_6m = forward_return(window, pd.Timestamp(recovery_date), 182) if recovery_date else np.nan
        from_recovery_12m = forward_return(window, pd.Timestamp(recovery_date), 365) if recovery_date else np.nan
        rows.append(
            {
                "symbol": symbol,
                "fund": name,
                "event": event,
                "shock_type": shock_type,
                **rec,
                "peak_to_trough_trading_days": trading_days_between(window, peak, trough),
                "trough_to_25pct_recovery_days": round(days25, 0) if not np.isnan(days25) else np.nan,
                "trough_to_25pct_recovery_trading_days": round(trading25, 0) if not np.isnan(trading25) else np.nan,
                "trough_to_50pct_recovery_days": round(days50, 0) if not np.isnan(days50) else np.nan,
                "trough_to_50pct_recovery_trading_days": round(trading50, 0) if not np.isnan(trading50) else np.nan,
                "trough_to_80pct_recovery_days": round(days80, 0) if not np.isnan(days80) else np.nan,
                "trough_to_80pct_recovery_trading_days": round(trading80, 0) if not np.isnan(trading80) else np.nan,
                "recovery_velocity_50_pp_per_day": round((0.50 * abs(maxdd)) / days50, 4) if days50 and not np.isnan(days50) else np.nan,
                "recovery_velocity_100_pp_per_day": round(abs(maxdd) / full_days_num, 4) if full_days_num and not np.isnan(full_days_num) else np.nan,
                "worst_1d_pct": round(window.pct_change().min() * 100.0, 2),
                "worst_5d_pct": round((window / window.shift(5) - 1).min() * 100.0, 2),
                "worst_20d_pct": round((window / window.shift(20) - 1).min() * 100.0, 2),
                "post_trough_1m_pct": round(forward_return(window, trough, 30), 2),
                "post_trough_3m_pct": round(post3, 2) if not np.isnan(post3) else np.nan,
                "post_trough_6m_pct": round(post6, 2) if not np.isnan(post6) else np.nan,
                "post_trough_12m_pct": round(post12, 2) if not np.isnan(post12) else np.nan,
                "post_trough_24m_pct": round(forward_return(window, trough, 730), 2),
                "post_full_recovery_6m_pct": round(from_recovery_6m, 2) if not np.isnan(from_recovery_6m) else np.nan,
                "post_full_recovery_12m_pct": round(from_recovery_12m, 2) if not np.isnan(from_recovery_12m) else np.nan,
                "benchmark_maxdd_pct": round(bench_dd, 2) if not np.isnan(bench_dd) else np.nan,
                "downside_capture_vs_6040_pct": round(downside_capture, 2) if not np.isnan(downside_capture) else np.nan,
                "benchmark_full_recovery_days": round(bench_recovery_days, 0) if not np.isnan(bench_recovery_days) else np.nan,
                "recovery_time_advantage_vs_6040_days": round(bench_recovery_days - full_days_num, 0) if not np.isnan(bench_recovery_days) and not np.isnan(full_days_num) else np.nan,
                "maxdd_advantage_vs_6040_pct": round(maxdd - bench_dd, 2) if not np.isnan(bench_dd) else np.nan,
                "benchmark_post_trough_3m_pct": round(benchmark_post_3m, 2) if not np.isnan(benchmark_post_3m) else np.nan,
                "benchmark_post_trough_6m_pct": round(benchmark_post_6m, 2) if not np.isnan(benchmark_post_6m) else np.nan,
                "benchmark_post_trough_12m_pct": round(benchmark_post_12m, 2) if not np.isnan(benchmark_post_12m) else np.nan,
                "recovery_alpha_3m_vs_6040_pct": round(post3 - benchmark_post_3m, 2) if not np.isnan(post3) and not np.isnan(benchmark_post_3m) else np.nan,
                "recovery_alpha_6m_vs_6040_pct": round(post6 - benchmark_post_6m, 2) if not np.isnan(post6) and not np.isnan(benchmark_post_6m) else np.nan,
                "recovery_alpha_12m_vs_6040_pct": round(post12 - benchmark_post_12m, 2) if not np.isnan(post12) and not np.isnan(benchmark_post_12m) else np.nan,
                **settling_metrics(window, recovery_date, prior_peak),
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


def score_throttle_events(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty:
        return events.copy(), pd.DataFrame()
    scored_parts = []

    def norm_low(series: pd.Series) -> pd.Series:
        v = pd.to_numeric(series, errors="coerce")
        if v.notna().sum() <= 1 or v.max() == v.min():
            return pd.Series(0.5, index=series.index)
        return ((v.max() - v) / (v.max() - v.min())).clip(0, 1).fillna(0.0)

    def norm_high(series: pd.Series) -> pd.Series:
        v = pd.to_numeric(series, errors="coerce")
        if v.notna().sum() <= 1 or v.max() == v.min():
            return pd.Series(0.5, index=series.index)
        return ((v - v.min()) / (v.max() - v.min())).clip(0, 1).fillna(0.0)

    for event, group in events.groupby("event", sort=False):
        g = group.copy()
        full_recovery = pd.to_numeric(g["trough_to_recovery_days"].replace("", np.nan), errors="coerce")
        unresolved_penalty = full_recovery.fillna(full_recovery.max() if full_recovery.notna().any() else 9999)
        if full_recovery.notna().any():
            unresolved_penalty = full_recovery.fillna(full_recovery.max() * 1.5)
        g["ThrottleScore_v2"] = (
            25 * norm_high(g["max_drawdown_pct"])
            + 15 * norm_low(g["trough_to_50pct_recovery_days"])
            + 10 * norm_low(g["trough_to_80pct_recovery_days"])
            + 20 * norm_low(unresolved_penalty)
            + 10 * norm_low(g["downside_capture_vs_6040_pct"])
            + 10 * norm_high(g["recovery_alpha_12m_vs_6040_pct"])
            + 5 * norm_high(g["post_recovery_relapse_90d_pct"])
            + 5 * norm_high(g["post_trough_12m_pct"])
        ).round(0)
        g["ThrottleScore_v2"] = g["ThrottleScore_v2"].fillna(0).astype(int)
        scored_parts.append(g)
    scored = pd.concat(scored_parts, ignore_index=True, sort=False)
    non_benchmark = scored[~scored["symbol"].astype(str).str.startswith("STATIC_60_40_")].copy()
    if non_benchmark.empty:
        non_benchmark = scored.copy()
    agg = (
        non_benchmark.groupby(["symbol", "fund"], as_index=False)
        .agg(
            average_throttle_score=("ThrottleScore_v2", "mean"),
            median_throttle_score=("ThrottleScore_v2", "median"),
            worst_shock_score=("ThrottleScore_v2", "min"),
            consistency_std=("ThrottleScore_v2", "std"),
            event_count=("event", "count"),
            average_maxdd_pct=("max_drawdown_pct", "mean"),
            average_recovery_alpha_12m_pct=("recovery_alpha_12m_vs_6040_pct", "mean"),
            average_recovery_time_advantage_days=("recovery_time_advantage_vs_6040_days", "mean"),
            average_downside_capture_pct=("downside_capture_vs_6040_pct", "mean"),
        )
    )
    agg["event_coverage_factor"] = (agg["event_count"] / 3.0).clip(upper=1.0)
    agg["reliability_adjusted_throttle_score"] = agg["average_throttle_score"] * agg["event_coverage_factor"]
    for col in [
        "average_throttle_score",
        "median_throttle_score",
        "worst_shock_score",
        "consistency_std",
        "event_coverage_factor",
        "reliability_adjusted_throttle_score",
        "average_maxdd_pct",
        "average_recovery_alpha_12m_pct",
        "average_recovery_time_advantage_days",
        "average_downside_capture_pct",
    ]:
        agg[col] = pd.to_numeric(agg[col], errors="coerce").round(2)
    return scored, agg.sort_values(["reliability_adjusted_throttle_score", "worst_shock_score"], ascending=False)


def audit_markdown() -> str:
    rows = [
        ("Max drawdown", "Yes", "recovery_metrics(), event_metrics()", "Adjusted NAV proxy", "Keep"),
        ("Peak date", "Yes", "recovery_metrics()", "Measured from window NAV", "Keep"),
        ("Trough date", "Yes", "recovery_metrics()", "Measured from window NAV", "Keep"),
        ("Peak -> trough days", "Yes", "recovery_metrics()", "Calendar days; v2 adds trading days", "Added trading-day field"),
        ("Trough -> 50% recovery", "Yes", "recovery_metrics(); v2 explicit threshold", "Calendar days", "Keep"),
        ("Trough -> 80% recovery", "Yes", "recovery_metrics(); v2 explicit threshold", "Calendar days", "Keep"),
        ("Trough -> 100% recovery", "Yes", "recovery_metrics()", "Blank if not recovered inside window", "Keep"),
        ("Peak -> full recovery", "Yes", "recovery_metrics()", "Calendar days", "Keep"),
        ("Underwater duration", "Yes", "max_underwater_days()", "Full-series metric; event-level still derived by recovery fields", "Keep"),
        ("Downside capture", "Yes", "event_metrics()", "Vs static 60/40 SPY+A35 proxy", "Keep; proxy caveat"),
        ("Upside / recovery capture", "Partial", "post_trough returns", "Absolute recovery exists; capture not explicit", "v2 adds recovery alpha fields"),
        ("Recovery alpha vs passive 60/40", "Missing in v1", "event_metrics()", "Benchmark aligned to fund trough date", "Added"),
        ("Recovery-time advantage vs 60/40", "Missing in v1", "event_metrics()", "Benchmark full-recovery days minus fund full-recovery days", "Added"),
        ("Post-recovery stability / relapse", "Missing in v1", "settling_metrics()", "30/60/90D relapse and 20-trading-day settled flag", "Added"),
        ("Post-trough 3M / 6M / 12M return", "Yes", "event_metrics()", "Measured when enough post-trough data exists", "Added 24M too"),
        ("Throttle score", "Partial", "correction_system_score", "v1 long-term score, not per-shock throttle", "Added ThrottleScore_v2 separately"),
    ]
    lines = [
        "# Throttle Research Audit",
        "",
        "Audit-first result: v1 already measured core NAV drawdown/recovery behavior, but lacked several explicit throttle-loop metrics. v2 extends v1 without replacing `correction_system_score`.",
        "",
        "| Metric | Existing? | File / function | Data quality | Action needed |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "## What Was Added In v2",
            "",
            "- `fund_throttle_event_metrics.csv`: one row per fund x shock with explicit recovery thresholds, velocity, benchmark-relative recovery alpha/time advantage, and settling/hunting fields.",
            "- `fund_throttle_ranking.csv`: average/median/worst-shock `ThrottleScore_v2` and consistency across available events.",
            "- `fund_throttle_summary.md`: compact answer table for GPT/user.",
            "- Additional charts for full-recovery days and downside-capture vs recovery-alpha scatter.",
            "",
            "## Caveats",
            "",
            "- Uses Yahoo adjusted NAV proxy from the existing cache; distribution total-return quality is not independently verified.",
            "- Static 60/40 is a proxy benchmark, not a perfect SGD-hedged passive product.",
            "- P-BIG current share class has short history; current-strategy evidence is mainly 2024-2026.",
            "- Holdings/process attribution remains missing; faster recovery is measured behavior, not automatically manager skill.",
            "",
        ]
    )
    return "\n".join(lines)


def throttle_summary_markdown(throttle_ranking: pd.DataFrame, throttle_events: pd.DataFrame) -> str:
    lines = [
        "# Fund Throttle / Correction-System Summary",
        "",
        "This is v2 throttle-speed output. It is event-based and does not overwrite the earlier long-term `correction_system_score`.",
        "",
    ]
    if not throttle_ranking.empty:
        cols = [
            "fund",
            "symbol",
            "average_throttle_score",
            "reliability_adjusted_throttle_score",
            "median_throttle_score",
            "worst_shock_score",
            "consistency_std",
            "event_count",
            "average_maxdd_pct",
            "average_recovery_alpha_12m_pct",
            "average_recovery_time_advantage_days",
        ]
        display = throttle_ranking[cols].head(12).where(pd.notna(throttle_ranking[cols].head(12)), "NA")
        lines.extend(["## Aggregate Ranking", ""])
        lines.append("| " + " | ".join(display.columns) + " |")
        lines.append("| " + " | ".join("---" for _ in display.columns) + " |")
        for record in display.to_dict("records"):
            lines.append("| " + " | ".join(str(record[col]) for col in display.columns) + " |")
    if not throttle_events.empty:
        event_best = throttle_events.sort_values("ThrottleScore_v2", ascending=False).groupby("event").head(3)
        display = event_best[["event", "fund", "symbol", "ThrottleScore_v2", "max_drawdown_pct", "trough_to_50pct_recovery_days", "trough_to_recovery_days", "recovery_alpha_12m_vs_6040_pct"]].where(pd.notna(event_best), "NA")
        lines.extend(["", "## Best Per Shock", ""])
        lines.append("| " + " | ".join(display.columns) + " |")
        lines.append("| " + " | ".join("---" for _ in display.columns) + " |")
        for record in display.to_dict("records"):
            lines.append("| " + " | ".join(str(record[col]) for col in display.columns) + " |")
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- `ThrottleScore_v2` is cross-sectional within each shock; compare funds inside the same event first.",
            "- A high score from only one event is weaker than consistent multi-event evidence.",
            "- P-BIG's current class mainly has recent-window evidence, so do not treat it as 2020/2022 proof.",
            "- Positive recovery alpha can come from beta rebound; manager attribution still needs holdings/process evidence.",
            "",
        ]
    )
    return "\n".join(lines)


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


def write_charts(paths: Paths, ranking: pd.DataFrame, nav_daily: pd.DataFrame, throttle_events: pd.DataFrame | None = None) -> list[str]:
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
    if throttle_events is not None and not throttle_events.empty:
        top = throttle_events.sort_values("ThrottleScore_v2", ascending=False).head(12)
        if not top.empty:
            plt.figure(figsize=(12, 7))
            labels = [f"{r['symbol']}\n{r['event'].replace('_', ' ')[:18]}" for _, r in top.iterrows()]
            values = pd.to_numeric(top["trough_to_recovery_days"].replace("", np.nan), errors="coerce")
            plt.bar(range(len(top)), values.fillna(0))
            plt.xticks(range(len(top)), labels, rotation=45, ha="right", fontsize=8)
            plt.ylabel("Trough to full recovery days")
            plt.title("Full Recovery Days: Top Throttle Events")
            plt.tight_layout()
            out = paths.charts_dir / "full_recovery_days_comparison.png"
            plt.savefig(out, dpi=160)
            plt.close()
            chart_files.append(str(out))

        scatter = throttle_events.dropna(subset=["downside_capture_vs_6040_pct", "recovery_alpha_12m_vs_6040_pct"])
        if not scatter.empty:
            plt.figure(figsize=(10, 7))
            plt.scatter(scatter["downside_capture_vs_6040_pct"], scatter["recovery_alpha_12m_vs_6040_pct"], alpha=0.75)
            for _, row in scatter.sort_values("ThrottleScore_v2", ascending=False).head(8).iterrows():
                plt.annotate(str(row["symbol"]), (row["downside_capture_vs_6040_pct"], row["recovery_alpha_12m_vs_6040_pct"]), fontsize=8)
            plt.axhline(0, color="black", linewidth=0.8)
            plt.axvline(100, color="gray", linewidth=0.8, linestyle="--")
            plt.xlabel("Downside capture vs 60/40 (%)")
            plt.ylabel("12M recovery alpha vs 60/40 (%)")
            plt.title("Downside Capture vs Recovery Alpha")
            plt.tight_layout()
            out = paths.charts_dir / "downside_capture_vs_recovery_alpha.png"
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
            "Start with `THROTTLE_RESEARCH_AUDIT.md`, then inspect `fund_throttle_summary.md`, `fund_throttle_ranking.csv`, and `fund_throttle_event_metrics.csv`.",
            "",
            "This package now contains v1 long-term correction-system metrics plus v2 event-based throttle-speed metrics. Treat manager skill attribution as incomplete unless `manager_process_evidence.csv` contains official evidence.",
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
            "- `ThrottleScore_v2` is event-based and separate from the older `correction_system_score`.",
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
    throttle_events, throttle_ranking = score_throttle_events(drawdown_events)
    ranking = score_rank(fund_master, drawdown_events)
    nav_daily = pd.DataFrame(nav_rows)

    output_files = [
        "THROTTLE_RESEARCH_AUDIT.md",
        "fund_throttle_event_metrics.csv",
        "fund_throttle_ranking.csv",
        "fund_throttle_summary.md",
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
    throttle_events.to_csv(paths.output_dir / "fund_throttle_event_metrics.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    throttle_ranking.to_csv(paths.output_dir / "fund_throttle_ranking.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    (paths.output_dir / "THROTTLE_RESEARCH_AUDIT.md").write_text(audit_markdown(), encoding="utf-8")
    (paths.output_dir / "fund_throttle_summary.md").write_text(throttle_summary_markdown(throttle_ranking, throttle_events), encoding="utf-8")
    fund_master.to_csv(paths.output_dir / "recovery_metrics.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    rolling = fund_master[["symbol", "fund", "history_years", "cagr_pct", "max_drawdown_pct", "ulcer_index", "sortino"]].copy()
    rolling["note"] = "v1 summary only; rolling 1Y/3Y/5Y time series to be expanded after total-return NAV validation"
    rolling.to_csv(paths.output_dir / "rolling_returns.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    ranking.to_csv(paths.output_dir / "correction_system_ranking.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    write_placeholder_tables(paths, candidates)
    chart_files = write_charts(paths, ranking, nav_daily, throttle_events)
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
                "throttle_events": int(len(throttle_events)),
                "top": "" if ranking.empty else ranking.iloc[0]["fund"],
                "top_throttle": "" if throttle_ranking.empty else throttle_ranking.iloc[0]["fund"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
