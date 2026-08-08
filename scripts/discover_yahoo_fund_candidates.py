#!/usr/bin/env python3
"""Discover Yahoo-verifiable Singapore fund candidates for manual review.

This is a Level 1 discovery helper. It searches Yahoo Finance with broad
retirement-fund keywords, validates whether each symbol has usable historical
NAV/price data, and writes a review CSV. It does not edit the production
fund_universe.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import requests


SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"
CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
USER_AGENT = "Mozilla/5.0 fund-discovery/1.0"
DEFAULT_OUTPUT = "AI_Fund_Research/output/fund_candidate_discovery.csv"


DEFAULT_QUERIES = [
    "SGD money market fund",
    "SGD cash fund",
    "SGD enhanced cash fund",
    "SGD short duration bond fund",
    "Singapore short term bond fund SGD",
    "Singapore income fund SGD",
    "Singapore balanced fund SGD",
    "Singapore dividend fund SGD",
    "Singapore equity income fund SGD",
    "Singapore low volatility fund SGD",
    "SGD hedged income fund",
    "SGD hedged global income fund",
    "SGD hedged global allocation fund",
    "SGD hedged multi asset income fund",
    "Allianz Income and Growth SGD",
    "Amundi SGD hedged income fund",
    "BlackRock SGD hedged income fund",
    "Fidelity SGD hedged income fund",
    "First Sentier Bridge Fund SGD",
    "Franklin Income Fund SGD",
    "JPM Global Income SGD Hedged",
    "JPM Income SGD Hedged",
    "LionGlobal SGD bond fund",
    "Manulife SGD hedged income fund",
    "Nikko AM Shenton SGD fund",
    "PIMCO Income SGD Hedged",
    "Schroder Asian Income SGD",
    "Schroder SGD hedged income fund",
    "United SGD Fund",
]


@dataclass(frozen=True)
class Candidate:
    symbol: str
    name: str
    source_query: str
    exchange: str
    quote_type: str


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def search_yahoo(query: str, quotes_count: int, timeout: int) -> list[dict[str, Any]]:
    response = requests.get(
        SEARCH_URL,
        params={"q": query, "quotesCount": quotes_count, "newsCount": 0},
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json().get("quotes", [])


def chart_download(symbol: str, years: int, timeout: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    end = date.today() + timedelta(days=1)
    start = date.today() - timedelta(days=int(365.25 * years) + 14)
    period1 = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp())
    period2 = int(datetime(end.year, end.month, end.day, tzinfo=timezone.utc).timestamp())
    params = urlencode({"period1": period1, "period2": period2, "interval": "1d", "events": "history", "includeAdjustedClose": "true"})
    response = requests.get(f"{CHART_URL.format(symbol=symbol)}?{params}", headers={"User-Agent": USER_AGENT}, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f"Yahoo HTTP {response.status_code}: {response.text[:160]}")
    payload = response.json()
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(f"Yahoo error: {chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise RuntimeError("No chart result")
    result = results[0]
    meta = result.get("meta", {})
    timestamps = result.get("timestamp") or []
    adjclose = (((result.get("indicators") or {}).get("adjclose") or [{}])[0]).get("adjclose") or []
    rows = []
    for ts, price in zip(timestamps, adjclose):
        if price is None:
            continue
        price = float(price)
        if not math.isfinite(price):
            continue
        rows.append({"date": datetime.fromtimestamp(int(ts), tz=timezone.utc).date(), "adj_close": price})
    if len(rows) < 20:
        raise RuntimeError("Too few usable rows")
    return pd.DataFrame(rows), meta


def max_drawdown(prices: pd.Series) -> float:
    running_peak = prices.cummax()
    dd = prices / running_peak - 1.0
    return float(dd.min() * 100.0)


def cagr(prices: pd.Series, dates: pd.Series) -> float:
    if len(prices) < 2:
        return float("nan")
    days = max((dates.iloc[-1] - dates.iloc[0]).days, 1)
    years = days / 365.25
    if prices.iloc[0] <= 0 or years <= 0:
        return float("nan")
    return float(((prices.iloc[-1] / prices.iloc[0]) ** (1 / years) - 1) * 100.0)


def infer_review_flags(symbol: str, name: str, currency: str, instrument_type: str) -> dict[str, str]:
    text = f"{symbol} {name}".lower()
    likely_sgd = currency.upper() == "SGD" or "sgd" in text
    likely_hedged = any(token in text for token in ["hedged", "hdg", "h2-sgd", "sgd-h"])
    retirement_type = "review"
    if any(token in text for token in ["short duration", "short term", "money market", "cash", "liquidity"]):
        retirement_type = "cash_or_short_duration"
    elif any(token in text for token in ["income", "multi asset", "allocation", "balanced", "bridge", "dividend"]):
        retirement_type = "income_or_multi_asset"
    elif any(token in text for token in ["equity", "growth", "high yield"]):
        retirement_type = "higher_risk_review"
    eligible_hint = "yes_review" if likely_sgd or likely_hedged else "no_or_unknown"
    return {
        "eligible_hint": eligible_hint,
        "likely_sgd_or_hedged": str(bool(likely_sgd or likely_hedged)),
        "likely_hedged": str(bool(likely_hedged)),
        "retirement_type_hint": retirement_type,
        "instrument_type": instrument_type,
    }


def load_existing_symbols(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        return set(pd.read_csv(path)["symbol"].astype(str).str.upper())
    except Exception:
        return set()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Discover Yahoo-verifiable fund candidates")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--universe", default="config/fund_universe.csv")
    parser.add_argument("--queries-file", default=None)
    parser.add_argument("--quotes-count", type=int, default=20)
    parser.add_argument("--years", type=int, default=6)
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--include-existing", action="store_true")
    args = parser.parse_args(argv)

    queries = DEFAULT_QUERIES
    if args.queries_file:
        queries = [line.strip() for line in Path(args.queries_file).read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]

    seen: dict[str, Candidate] = {}
    for query in queries:
        try:
            quotes = search_yahoo(query, args.quotes_count, args.timeout)
        except Exception as exc:
            print(json.dumps({"message": "search_failed", "query": query, "error": str(exc)}), file=sys.stderr)
            continue
        for item in quotes:
            symbol = str(item.get("symbol") or "").upper().strip()
            if not symbol:
                continue
            is_singapore_like = symbol.endswith(".SI") or bool(re.match(r"0P[0-9A-Z]+(?:\\.SI)?$", symbol))
            is_fund_like = str(item.get("typeDisp") or "").lower() == "fund" or str(item.get("quoteType") or "").upper() in {"MUTUALFUND", "ETF"}
            if not (is_singapore_like or is_fund_like):
                continue
            if symbol not in seen:
                seen[symbol] = Candidate(
                    symbol=symbol,
                    name=str(item.get("shortname") or item.get("longname") or symbol),
                    source_query=query,
                    exchange=str(item.get("exchDisp") or item.get("exchange") or ""),
                    quote_type=str(item.get("quoteType") or item.get("typeDisp") or ""),
                )
        time.sleep(args.sleep)

    existing = load_existing_symbols(Path(args.universe))
    rows: list[dict[str, Any]] = []
    for candidate in seen.values():
        if candidate.symbol in existing and not args.include_existing:
            continue
        status = "ok"
        error = ""
        meta: dict[str, Any] = {}
        metrics: dict[str, Any] = {}
        try:
            nav, meta = chart_download(candidate.symbol, args.years, args.timeout)
            nav["date"] = pd.to_datetime(nav["date"])
            metrics = {
                "rows": int(len(nav)),
                "first_date": nav["date"].min().date().isoformat(),
                "last_date": nav["date"].max().date().isoformat(),
                "history_years": round(float((nav["date"].max() - nav["date"].min()).days / 365.25), 2),
                "cagr_pct": round(cagr(nav["adj_close"], nav["date"]), 2),
                "maxdd_pct": round(max_drawdown(nav["adj_close"]), 2),
            }
        except Exception as exc:
            status = "failed"
            error = str(exc)[:240]

        currency = str(meta.get("currency") or "")
        instrument_type = str(meta.get("instrumentType") or candidate.quote_type)
        flags = infer_review_flags(candidate.symbol, candidate.name, currency, instrument_type)
        rows.append(
            {
                "symbol": candidate.symbol,
                "name": candidate.name,
                "status": status,
                "error": error,
                "currency": currency,
                "exchange": str(meta.get("exchangeName") or candidate.exchange),
                "instrument_type": instrument_type,
                **metrics,
                **flags,
                "source_query": candidate.source_query,
                "already_in_universe": str(candidate.symbol in existing),
            }
        )
        time.sleep(args.sleep)

    out_path = Path(args.output)
    if out_path.parent:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.name == "fund_candidate_discovery.csv":
        out_path = out_path.with_name(f"fund_candidate_discovery_{now_stamp()}.csv")
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["status", "eligible_hint", "history_years", "maxdd_pct"], ascending=[True, True, False, False])
    else:
        df = pd.DataFrame(
            columns=[
                "symbol",
                "name",
                "status",
                "error",
                "currency",
                "exchange",
                "instrument_type",
                "eligible_hint",
                "likely_sgd_or_hedged",
                "likely_hedged",
                "retirement_type_hint",
                "source_query",
                "already_in_universe",
            ]
        )
    df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(json.dumps({"message": "discovery_finished", "output": str(out_path), "candidates": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
