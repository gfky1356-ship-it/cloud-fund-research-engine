#!/usr/bin/env python3
"""Cloud-friendly fund research engine for conservative retirement screening.

The engine is designed for Google Colab first:
Python heavy work -> Google Drive persistent cache/output -> ChatGPT reads CSV/JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests


YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/AI_Fund_Research"
LOCAL_ROOT = "AI_Fund_Research"
TIER_A_MAX_DD = -5.0
TIER_B_MAX_DD = -10.0
TIER_C_MAX_DD = -15.0
USER_AGENT = "Mozilla/5.0 fund-research-engine/1.0"


@dataclass(frozen=True)
class Paths:
    project_dir: Path
    storage_root: Path
    cache_dir: Path
    output_dir: Path
    log_dir: Path
    db_path: Path
    universe_path: Path


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def is_colab() -> bool:
    return "google.colab" in sys.modules or Path("/content").exists()


def resolve_paths(storage_root: str | None, universe_path: str | None) -> Paths:
    project_dir = Path(__file__).resolve().parent
    if storage_root:
        root = Path(storage_root).expanduser()
    elif Path("/content/drive/MyDrive").exists():
        root = Path(DEFAULT_DRIVE_ROOT)
    else:
        root = project_dir / LOCAL_ROOT
    cache_dir = root / "cache"
    output_dir = root / "output"
    log_dir = root / "logs"
    for folder in (root, cache_dir, output_dir, log_dir):
        folder.mkdir(parents=True, exist_ok=True)
    return Paths(
        project_dir=project_dir,
        storage_root=root,
        cache_dir=cache_dir,
        output_dir=output_dir,
        log_dir=log_dir,
        db_path=cache_dir / "fund_research.sqlite",
        universe_path=Path(universe_path).expanduser() if universe_path else project_dir / "config" / "fund_universe.csv",
    )


def log_event(paths: Paths, message: str, **fields: Any) -> None:
    payload = {"ts": now_iso(), "message": message, **fields}
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    print(line)
    with (paths.log_dir / "fund_research_run.jsonl").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def init_db(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS nav_history (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                adj_close REAL NOT NULL,
                source TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                PRIMARY KEY (symbol, date)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fund_metadata (
                symbol TEXT PRIMARY KEY,
                metadata_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS source_status (
                symbol TEXT PRIMARY KEY,
                last_status TEXT NOT NULL,
                last_error TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def load_universe(universe_path: Path) -> pd.DataFrame:
    if not universe_path.exists():
        raise FileNotFoundError(f"Universe file not found: {universe_path}")
    df = pd.read_csv(universe_path)
    required = {"symbol", "name", "type", "currency", "retirement_candidate", "benchmark"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Universe missing required columns: {sorted(missing)}")
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    for col in ("retirement_candidate", "benchmark"):
        df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    return df


def existing_nav_bounds(db_path: Path, symbol: str) -> tuple[str | None, str | None, int]:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT MIN(date), MAX(date), COUNT(*) FROM nav_history WHERE symbol = ?",
            (symbol,),
        ).fetchone()
    return row[0], row[1], int(row[2] or 0)


def yahoo_download(symbol: str, start: date, end: date) -> pd.DataFrame:
    period1 = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp())
    period2 = int(datetime(end.year, end.month, end.day, tzinfo=timezone.utc).timestamp())
    params = urlencode({"period1": period1, "period2": period2, "interval": "1d", "events": "history", "includeAdjustedClose": "true"})
    url = f"{YAHOO_CHART_URL.format(symbol=symbol)}?{params}"
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Yahoo HTTP {response.status_code}: {response.text[:200]}")
    payload = response.json()
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(f"Yahoo error: {chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise RuntimeError("Yahoo returned no chart result")
    result = results[0]
    timestamps = result.get("timestamp") or []
    adjclose = (((result.get("indicators") or {}).get("adjclose") or [{}])[0]).get("adjclose") or []
    rows: list[dict[str, Any]] = []
    for ts, price in zip(timestamps, adjclose):
        if price is None or not math.isfinite(float(price)):
            continue
        rows.append(
            {
                "symbol": symbol,
                "date": datetime.fromtimestamp(int(ts), tz=timezone.utc).date().isoformat(),
                "adj_close": float(price),
                "source": "yahoo_chart",
                "fetched_at": now_iso(),
            }
        )
    if not rows:
        raise RuntimeError("Yahoo returned no usable adjusted close rows")
    return pd.DataFrame(rows).drop_duplicates(["symbol", "date"])


def upsert_nav(db_path: Path, nav: pd.DataFrame) -> int:
    if nav.empty:
        return 0
    records = nav[["symbol", "date", "adj_close", "source", "fetched_at"]].itertuples(index=False, name=None)
    with sqlite3.connect(db_path) as conn:
        before = conn.total_changes
        conn.executemany(
            """
            INSERT INTO nav_history(symbol, date, adj_close, source, fetched_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(symbol, date) DO UPDATE SET
                adj_close = excluded.adj_close,
                source = excluded.source,
                fetched_at = excluded.fetched_at
            """,
            list(records),
        )
        conn.commit()
        return conn.total_changes - before


def update_source_status(db_path: Path, symbol: str, status: str, error: str | None = None) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO source_status(symbol, last_status, last_error, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                last_status = excluded.last_status,
                last_error = excluded.last_error,
                updated_at = excluded.updated_at
            """,
            (symbol, status, error, now_iso()),
        )
        conn.commit()


def refresh_symbol(paths: Paths, symbol: str, years: int, force_full: bool, sleep_seconds: float) -> dict[str, Any]:
    today = date.today()
    earliest = today - timedelta(days=int(365.25 * years) + 14)
    _, max_date, row_count = existing_nav_bounds(paths.db_path, symbol)
    if force_full or not max_date:
        start = earliest
        reason = "full_refresh" if force_full else "empty_cache"
    else:
        start = max(datetime.fromisoformat(max_date).date() + timedelta(days=1), earliest)
        reason = "incremental"
    end = today + timedelta(days=1)
    if start >= end:
        update_source_status(paths.db_path, symbol, "cache_current", None)
        return {"symbol": symbol, "status": "cache_current", "reason": reason, "existing_rows": row_count, "new_rows": 0}
    try:
        nav = yahoo_download(symbol, start, end)
        changed = upsert_nav(paths.db_path, nav)
        update_source_status(paths.db_path, symbol, "ok", None)
        if sleep_seconds:
            time.sleep(sleep_seconds)
        return {"symbol": symbol, "status": "ok", "reason": reason, "existing_rows": row_count, "new_rows": int(len(nav)), "db_changes": changed}
    except Exception as exc:  # noqa: BLE001 - status log needs exact source failure
        incremental_no_data = (
            reason == "incremental"
            and row_count > 0
            and (
                "Data doesn't exist for startDate" in str(exc)
                or "Yahoo returned no usable adjusted close rows" in str(exc)
                or "Yahoo returned no chart result" in str(exc)
            )
        )
        if incremental_no_data:
            update_source_status(paths.db_path, symbol, "cache_current", None)
            return {
                "symbol": symbol,
                "status": "cache_current",
                "reason": "incremental_no_new_trading_data",
                "existing_rows": row_count,
                "new_rows": 0,
            }
        update_source_status(paths.db_path, symbol, "failed", str(exc))
        return {"symbol": symbol, "status": "failed", "reason": reason, "existing_rows": row_count, "new_rows": 0, "error": str(exc)}


def read_nav(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query("SELECT symbol, date, adj_close FROM nav_history ORDER BY symbol, date", conn, parse_dates=["date"])


def pct_from_seed(value: Any) -> float | None:
    if pd.isna(value) or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def max_drawdown(prices: pd.Series) -> float:
    if prices.empty:
        return float("nan")
    running_peak = prices.cummax()
    dd = prices / running_peak - 1.0
    return float(dd.min() * 100.0)


def max_recovery_days(prices: pd.Series, dates: pd.Series) -> float:
    if len(prices) < 2:
        return float("nan")
    peak = float(prices.iloc[0])
    peak_date = dates.iloc[0]
    underwater_start = None
    max_days = 0
    for price, current_date in zip(prices.iloc[1:], dates.iloc[1:]):
        price = float(price)
        if price >= peak:
            if underwater_start is not None:
                max_days = max(max_days, int((current_date - underwater_start).days))
                underwater_start = None
            peak = price
            peak_date = current_date
        elif underwater_start is None:
            underwater_start = peak_date
    if underwater_start is not None:
        max_days = max(max_days, int((dates.iloc[-1] - underwater_start).days))
    return float(max_days)


def cagr(prices: pd.Series, dates: pd.Series) -> float:
    if len(prices) < 2:
        return float("nan")
    days = max((dates.iloc[-1] - dates.iloc[0]).days, 1)
    years = days / 365.25
    if prices.iloc[0] <= 0 or years <= 0:
        return float("nan")
    return float(((prices.iloc[-1] / prices.iloc[0]) ** (1 / years) - 1) * 100.0)


def annual_volatility(prices: pd.Series) -> float:
    returns = prices.pct_change().dropna()
    if len(returns) < 20:
        return float("nan")
    return float(returns.std() * math.sqrt(252) * 100.0)


def sharpe_like(prices: pd.Series, risk_free_pct: float = 0.0) -> float:
    returns = prices.pct_change().dropna()
    if len(returns) < 20 or returns.std() == 0:
        return float("nan")
    annual_return = returns.mean() * 252 * 100.0
    annual_vol = returns.std() * math.sqrt(252) * 100.0
    return float((annual_return - risk_free_pct) / annual_vol)


def sortino_like(prices: pd.Series, risk_free_pct: float = 0.0) -> float:
    returns = prices.pct_change().dropna()
    downside = returns[returns < 0]
    if len(returns) < 20 or len(downside) < 2 or downside.std() == 0:
        return float("nan")
    annual_return = returns.mean() * 252 * 100.0
    downside_vol = downside.std() * math.sqrt(252) * 100.0
    return float((annual_return - risk_free_pct) / downside_vol)


def risk_tier(maxdd_pct: float) -> str:
    if pd.isna(maxdd_pct):
        return "Unknown"
    if maxdd_pct >= TIER_A_MAX_DD:
        return "A"
    if maxdd_pct >= TIER_B_MAX_DD:
        return "B"
    if maxdd_pct >= TIER_C_MAX_DD:
        return "C"
    return "Reject"


def calmar_ratio(cagr_pct: float, maxdd_pct: float) -> float:
    if pd.isna(cagr_pct) or pd.isna(maxdd_pct):
        return float("nan")
    if cagr_pct <= 0:
        return 0.0
    denominator = max(abs(float(maxdd_pct)), 1.5)
    return float(cagr_pct / denominator)


def compute_metrics(universe: pd.DataFrame, nav: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    today = pd.Timestamp(date.today())
    five_year_start = today - pd.Timedelta(days=int(365.25 * 5))
    one_year_start = today - pd.Timedelta(days=366)
    three_year_start = today - pd.Timedelta(days=int(365.25 * 3))
    for item in universe.to_dict("records"):
        symbol = item["symbol"]
        sdf = nav[nav["symbol"] == symbol].sort_values("date")
        if sdf.empty:
            rows.append({**item, "status": "NO_NAV", "fail_reason": "No NAV/price data"})
            continue
        windows = {
            "1y": sdf[sdf["date"] >= one_year_start],
            "3y": sdf[sdf["date"] >= three_year_start],
            "5y": sdf[sdf["date"] >= five_year_start],
        }
        lifetime = sdf
        primary = windows["5y"] if len(windows["5y"]) >= 60 else lifetime
        history_days = int((sdf["date"].iloc[-1] - sdf["date"].iloc[0]).days)
        history_years = history_days / 365.25
        history_less_than_5y = history_years < 4.85
        row = {
            **item,
            "status": "OK",
            "last_date": sdf["date"].iloc[-1].date().isoformat(),
            "history_years": round(history_years, 2),
            "history_less_than_5y": bool(history_less_than_5y),
            "cagr_1y_pct": cagr(windows["1y"]["adj_close"], windows["1y"]["date"]) if len(windows["1y"]) >= 2 else float("nan"),
            "cagr_3y_pct": cagr(windows["3y"]["adj_close"], windows["3y"]["date"]) if len(windows["3y"]) >= 2 else float("nan"),
            "cagr_5y_pct": cagr(windows["5y"]["adj_close"], windows["5y"]["date"]) if len(windows["5y"]) >= 2 else float("nan"),
            "maxdd_1y_pct": max_drawdown(windows["1y"]["adj_close"]) if len(windows["1y"]) >= 2 else float("nan"),
            "maxdd_3y_pct": max_drawdown(windows["3y"]["adj_close"]) if len(windows["3y"]) >= 2 else float("nan"),
            "maxdd_5y_or_life_pct": max_drawdown(primary["adj_close"]) if len(primary) >= 2 else float("nan"),
            "volatility_1y_pct": annual_volatility(windows["1y"]["adj_close"]),
            "recovery_days_5y_or_life": max_recovery_days(primary["adj_close"], primary["date"]) if len(primary) >= 2 else float("nan"),
            "sharpe_1y": sharpe_like(windows["1y"]["adj_close"]),
            "sortino_1y": sortino_like(windows["1y"]["adj_close"]),
            "yield_pct": pct_from_seed(item.get("yield_pct_seed")),
            "fee_pct": pct_from_seed(item.get("fee_pct_seed")),
            "duration_years": pct_from_seed(item.get("duration_years_seed")),
        }
        cagr_base = row["cagr_5y_pct"] if pd.notna(row["cagr_5y_pct"]) else row["cagr_3y_pct"]
        if pd.isna(cagr_base):
            cagr_base = row["cagr_1y_pct"]
        row["risk_tier"] = risk_tier(row["maxdd_5y_or_life_pct"])
        row["calmar_5y_or_life"] = calmar_ratio(cagr_base, row["maxdd_5y_or_life_pct"])
        row["retirement_pass"] = bool(
            row["retirement_candidate"]
            and not row["benchmark"]
            and pd.notna(row["maxdd_5y_or_life_pct"])
            and row["risk_tier"] in {"A", "B", "C"}
        )
        if row["benchmark"]:
            row["fail_reason"] = "Benchmark only"
        elif not row["retirement_candidate"]:
            row["fail_reason"] = "Not marked as retirement candidate"
        elif pd.isna(row["maxdd_5y_or_life_pct"]):
            row["fail_reason"] = "Insufficient drawdown data"
        elif row["risk_tier"] == "Reject":
            row["fail_reason"] = "MaxDD worse than 15% retirement limit"
        else:
            row["fail_reason"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def score_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    scored = metrics.copy()
    scored["score"] = np.nan
    pass_mask = scored["retirement_pass"].fillna(False)
    if not pass_mask.any():
        return scored
    cagr_base = scored["cagr_5y_pct"].where(pd.notna(scored["cagr_5y_pct"]), scored["cagr_3y_pct"]).where(
        pd.notna(scored["cagr_5y_pct"].where(pd.notna(scored["cagr_5y_pct"]), scored["cagr_3y_pct"])),
        scored["cagr_1y_pct"],
    )
    calmar = scored["calmar_5y_or_life"].fillna(0).clip(lower=0, upper=2.0)
    sortino = scored["sortino_1y"].fillna(0).clip(lower=0, upper=4.0)
    dd_score = ((scored["maxdd_5y_or_life_pct"].clip(lower=-15, upper=0) + 15) / 15).fillna(0)
    vol = scored["volatility_1y_pct"].fillna(scored["volatility_1y_pct"].median()).clip(lower=0, upper=12)
    yld = scored["yield_pct"].fillna(0).clip(lower=0, upper=7)
    fee = scored["fee_pct"].fillna(scored["fee_pct"].median()).clip(lower=0, upper=1.5)
    recovery = scored["recovery_days_5y_or_life"].fillna(scored["recovery_days_5y_or_life"].median()).clip(lower=0, upper=1095)
    score = (
        28 * (calmar / 2.0)
        + 22 * (sortino / 4.0)
        + 15 * dd_score
        + 14 * (cagr_base.fillna(0).clip(lower=0, upper=8) / 8)
        + 9 * (yld / 7)
        + 5 * (1 - (fee / 1.5))
        + 4 * (1 - (vol / 12))
        + 3 * (1 - (recovery / 1095))
    )
    scored.loc[pass_mask, "score"] = score[pass_mask].round(0)
    return scored


def scoped_for_display(scored: pd.DataFrame, display_currency: str, include_sgd_hedged: bool = True) -> pd.DataFrame:
    if display_currency.lower() == "all":
        return scored.copy()
    wanted = display_currency.upper()
    currency_match = scored["currency"].astype(str).str.upper() == wanted
    if wanted == "SGD" and include_sgd_hedged:
        hedge_match = scored["sgd_hedged"].astype(str).str.lower().isin(["yes", "true", "1", "y"])
        return scored[currency_match | hedge_match].copy()
    return scored[currency_match].copy()


def compact_latest(scored: pd.DataFrame, top_n: int) -> pd.DataFrame:
    passed = scored[scored["retirement_pass"].fillna(False)].copy()
    source = passed
    if source.empty:
        source = scored[
            scored["retirement_candidate"].fillna(False)
            & ~scored["benchmark"].fillna(False)
            & (scored["status"] == "OK")
        ].copy()
    if source.empty:
        return source
    cagr_base = source["cagr_5y_pct"].where(pd.notna(source["cagr_5y_pct"]), source["cagr_3y_pct"])
    source["cagr_display"] = cagr_base.where(pd.notna(cagr_base), source["cagr_1y_pct"])
    sort_score = source["score"].fillna(-1)
    source = source.assign(_sort_score=sort_score).sort_values(
        ["_sort_score", "maxdd_5y_or_life_pct", "yield_pct"],
        ascending=[False, False, False],
    ).head(top_n)
    columns = [
        "symbol",
        "name",
        "type",
        "currency",
        "sgd_hedged",
        "cagr_display",
        "maxdd_5y_or_life_pct",
        "calmar_5y_or_life",
        "yield_pct",
        "fee_pct",
        "volatility_1y_pct",
        "recovery_days_5y_or_life",
        "risk_tier",
        "score",
        "retirement_pass",
        "fail_reason",
        "history_less_than_5y",
        "last_date",
    ]
    out = source[columns].rename(
        columns={
            "symbol": "Fund",
            "name": "Name",
            "type": "Type",
            "currency": "Ccy",
            "sgd_hedged": "SGD_Hedged",
            "cagr_display": "CAGR",
            "maxdd_5y_or_life_pct": "MaxDD",
            "calmar_5y_or_life": "Calmar",
            "yield_pct": "Yield",
            "fee_pct": "Fee",
            "volatility_1y_pct": "Vol",
            "recovery_days_5y_or_life": "RecoveryDays",
            "risk_tier": "RiskTier",
            "score": "Score",
            "retirement_pass": "Retirement_Pass",
            "fail_reason": "Fail_Reason",
            "history_less_than_5y": "History_LT_5Y",
            "last_date": "Last_NAV",
        }
    )
    out.insert(0, "Code", out["Fund"])
    out["Fund"] = out["Name"]
    for col in ("CAGR", "MaxDD", "Calmar", "Yield", "Fee", "Vol"):
        out[col] = out[col].astype(float).round(2)
    out["RecoveryDays"] = out["RecoveryDays"].round(0).astype("Int64")
    out["Score"] = out["Score"].round(0).astype("Int64")
    out["Fail_Reason"] = out["Fail_Reason"].fillna("")
    return out


def universe_funnel(scored: pd.DataFrame, scoped: pd.DataFrame, latest: pd.DataFrame) -> dict[str, int]:
    eligible = scoped[
        scoped["retirement_candidate"].fillna(False)
        & ~scoped["benchmark"].fillna(False)
        & (scoped["status"] == "OK")
    ]
    return {
        "total_scanned": int(len(scored)),
        "sgd_or_sgd_hedged_eligible": int(len(eligible)),
        "history_gte_5y": int((eligible["history_less_than_5y"].fillna(True) == False).sum()),
        "tier_a": int((eligible["risk_tier"] == "A").sum()),
        "tier_b": int((eligible["risk_tier"] == "B").sum()),
        "tier_c": int((eligible["risk_tier"] == "C").sum()),
        "excluded_gt_15dd": int((eligible["risk_tier"] == "Reject").sum()),
        "final_top10": int(len(latest)),
    }


def cache_summary(statuses: list[dict[str, Any]], nav: pd.DataFrame) -> dict[str, Any]:
    full_downloads = sum(1 for item in statuses if item.get("reason") in {"empty_cache", "full_refresh"} and item.get("new_rows", 0) > 0)
    incremental_updates = sum(1 for item in statuses if item.get("reason") == "incremental" and item.get("new_rows", 0) > 0)
    no_new_data_hits = sum(1 for item in statuses if item.get("status") == "cache_current")
    failed_sources = sum(1 for item in statuses if item.get("status") == "failed")
    reused_rows = sum(int(item.get("existing_rows") or 0) for item in statuses)
    downloaded_rows = sum(int(item.get("new_rows") or 0) for item in statuses)
    funds_cached = int(nav["symbol"].nunique()) if not nav.empty else 0
    cached_nav_rows = int(len(nav))
    if failed_sources:
        state = "WARN"
    elif full_downloads and reused_rows:
        state = "MIXED"
    elif full_downloads:
        state = "MISS"
    else:
        state = "HIT"
    return {
        "state": state,
        "funds_cached": funds_cached,
        "cached_nav_rows": cached_nav_rows,
        "reused_rows": reused_rows,
        "downloaded_rows": downloaded_rows,
        "full_downloads": full_downloads,
        "incremental_updates": incremental_updates,
        "no_new_data_cache_hits": no_new_data_hits,
        "failed_sources": failed_sources,
    }


def markdown_summary(latest: pd.DataFrame, generated_at: str, mode: str, display_scope: str, cache: dict[str, Any], funnel: dict[str, int]) -> str:
    lines = [
        "# Fund Daily Summary",
        "",
        f"- Generated UTC: `{generated_at}`",
        f"- Mode: `{mode}`",
        f"- Display scope: `{display_scope}`",
        f"- Cache: `{cache['state']}`",
        "- Ranking: risk-adjusted retirement Top 10 using Calmar, Sortino, MaxDD, recovery time, CAGR, yield, fee, and volatility.",
        "- Risk tiers: `A <=5% DD`, `B >5% to 10% DD`, `C >10% to 15% DD`, `Reject >15% DD`.",
        "- SPY/ES3.SI are benchmark only and are not eligible for the retirement shortlist.",
        "",
    ]
    if latest.empty:
        lines.extend(
            [
                "## Top Retirement Candidates",
                "",
                "No SGD / SGD-hedged fund reached Tier A-C eligibility in this run.",
                "",
            ]
        )
    else:
        display = latest.copy()
        display["Fund"] = display["Fund"].astype(str)
        display["Type"] = display["Type"].astype(str)
        display = display[["Fund", "Code", "Type", "CAGR", "MaxDD", "Calmar", "Yield", "Fee", "RiskTier", "Score"]]
        display = display.where(pd.notna(display), "NA")
        header = "| " + " | ".join(display.columns) + " |"
        separator = "| " + " | ".join("---" for _ in display.columns) + " |"
        rows = []
        for record in display.to_dict("records"):
            rows.append("| " + " | ".join(str(record[col]) for col in display.columns) + " |")
        lines.extend(
            [
                "## Top Retirement Candidates",
                "",
                "\n".join([header, separator, *rows]),
                "",
            ]
        )
    lines.extend(
        [
            "## Universe Funnel",
            "",
            f"- Total scanned: `{funnel['total_scanned']}`",
            f"- SGD / SGD-hedged eligible: `{funnel['sgd_or_sgd_hedged_eligible']}`",
            f"- >=5Y history: `{funnel['history_gte_5y']}`",
            f"- Tier A: `{funnel['tier_a']}`",
            f"- Tier B: `{funnel['tier_b']}`",
            f"- Tier C: `{funnel['tier_c']}`",
            f"- >15% DD excluded: `{funnel['excluded_gt_15dd']}`",
            f"- Final Top 10 count: `{funnel['final_top10']}`",
            "",
            "## Cache Status",
            "",
            f"- Cache: `{cache['state']}`",
            f"- Funds cached: `{cache['funds_cached']}`",
            f"- Cached NAV rows: `{cache['cached_nav_rows']}`",
            f"- Reused rows: `{cache['reused_rows']}`",
            f"- Downloaded rows this run: `{cache['downloaded_rows']}`",
            f"- Full downloads: `{cache['full_downloads']}`",
            f"- Incremental updates: `{cache['incremental_updates']}`",
            f"- No-new-data cache hits: `{cache['no_new_data_cache_hits']}`",
            f"- Failed sources: `{cache['failed_sources']}`",
            "",
            "## GPT Reading Notes",
            "",
            "- Treat this file as the compact latest view.",
            "- Use CSV/JSON only when exact machine-readable fields are needed.",
            "- Do not treat USD-only unhedged products as SGD retirement candidates.",
            "- Tier B/C names are research candidates, not automatic buy recommendations.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(paths: Paths, scored: pd.DataFrame, scoped: pd.DataFrame, latest: pd.DataFrame, mode: str, display_scope: str, statuses: list[dict[str, Any]], nav: pd.DataFrame) -> dict[str, str]:
    run_date = date.today().isoformat()
    latest_csv = paths.output_dir / "fund_daily_summary.csv"
    latest_json = paths.output_dir / "fund_daily_summary.json"
    latest_md = paths.output_dir / "fund_daily_summary.md"
    full_csv = paths.output_dir / f"{run_date}_fund_full_ranking.csv"
    status_json = paths.output_dir / f"{run_date}_fund_run_status.json"
    latest.to_csv(latest_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    scored.to_csv(full_csv, index=False)
    generated_at = now_iso()
    cache = cache_summary(statuses, nav)
    funnel = universe_funnel(scored, scoped, latest)
    payload = {
        "generated_at": generated_at,
        "mode": mode,
        "display_scope": display_scope,
        "cache": cache,
        "universe_funnel": funnel,
        "ranking_logic": "Risk-adjusted Top 10. Tier A <=5% DD, B >5% to 10%, C >10% to 15%, Reject >15%. Score uses Calmar, Sortino, MaxDD, recovery days, CAGR, yield, fee, and volatility.",
        "storage_root": str(paths.storage_root),
        "latest_csv": str(latest_csv),
        "latest_json": str(latest_json),
        "latest_md": str(latest_md),
        "rows": latest.replace({np.nan: None}).to_dict("records"),
    }
    latest_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_md.write_text(markdown_summary(latest, generated_at, mode, display_scope, cache, funnel), encoding="utf-8")
    status_payload = {
        "generated_at": now_iso(),
        "mode": mode,
        "display_scope": display_scope,
        "cache": cache,
        "universe_funnel": funnel,
        "db_path": str(paths.db_path),
        "universe_path": str(paths.universe_path),
        "source_status": statuses,
        "pass_count": int(scored["retirement_pass"].fillna(False).sum()) if "retirement_pass" in scored else 0,
        "fail_count": int((~scored["retirement_pass"].fillna(False)).sum()) if "retirement_pass" in scored else int(len(scored)),
        "outputs": {
            "latest_csv": str(latest_csv),
            "latest_json": str(latest_json),
            "latest_md": str(latest_md),
            "full_ranking_csv": str(full_csv),
        },
    }
    status_json.write_text(json.dumps(status_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "latest_csv": str(latest_csv),
        "latest_json": str(latest_json),
        "latest_md": str(latest_md),
        "full_ranking_csv": str(full_csv),
        "status_json": str(status_json),
    }


def run_engine(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.storage_root, args.universe)
    init_db(paths.db_path)
    universe = load_universe(paths.universe_path)
    years = 6 if args.mode == "quick-daily" else 10
    force_full = args.force_full or args.mode == "deep-weekend"
    log_event(paths, "run_started", mode=args.mode, storage_root=str(paths.storage_root), universe_rows=int(len(universe)))
    statuses = []
    for symbol in universe["symbol"].tolist():
        status = refresh_symbol(paths, symbol, years=years, force_full=force_full, sleep_seconds=args.sleep)
        statuses.append(status)
        log_event(paths, "symbol_refresh", **status)
    nav = read_nav(paths.db_path)
    metrics = compute_metrics(universe, nav)
    scored = score_rows(metrics)
    scoped = scoped_for_display(scored, args.display_currency, include_sgd_hedged=not args.exclude_sgd_hedged)
    latest = compact_latest(scoped, args.top_n)
    display_scope = args.display_currency.upper()
    if display_scope == "SGD" and not args.exclude_sgd_hedged:
        display_scope = "SGD plus SGD-hedged"
    outputs = write_outputs(paths, scored, scoped, latest, args.mode, display_scope, statuses, nav)
    log_event(paths, "run_finished", outputs=outputs, latest_rows=int(len(latest)))
    print("\nLatest Top Funds")
    if latest.empty:
        print("No SGD / SGD-hedged fund reached Tier A-C eligibility. Check full ranking/status outputs.")
    else:
        print(latest.to_string(index=False))
    print("\nStable ChatGPT bridge outputs:")
    print(outputs["latest_csv"])
    print(outputs["latest_json"])
    print(outputs["latest_md"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cloud Python retirement fund research engine")
    parser.add_argument("--mode", choices=["quick-daily", "deep-weekend"], default="quick-daily")
    parser.add_argument("--storage-root", default=None, help="Persistent root. In Colab use /content/drive/MyDrive/AI_Fund_Research")
    parser.add_argument("--universe", default=None, help="CSV fund universe path")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--display-currency", default="SGD", help="Currency scope for latest output. Use ALL to show all currencies.")
    parser.add_argument("--exclude-sgd-hedged", action="store_true", help="For SGD display, exclude non-SGD funds marked SGD-hedged.")
    parser.add_argument("--force-full", action="store_true", help="Ignore cache bounds and refresh full history window")
    parser.add_argument("--sleep", type=float, default=0.2, help="Seconds between source requests")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_engine(args)


if __name__ == "__main__":
    raise SystemExit(main())
