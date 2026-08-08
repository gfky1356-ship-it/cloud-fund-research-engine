#!/usr/bin/env python3
"""Analyze the Level 1 discovery pool before doing more broad search.

The discovery pool is intentionally slower-moving than daily NAV/ranking.
This script merges discovery CSVs, deduplicates share classes into fund
families, estimates category coverage, and writes a compact gap report.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DISCOVERY_GLOB = "AI_Fund_Research/output/fund_candidate_discovery_*.csv"
DEFAULT_UNIVERSE = "config/fund_universe.csv"
DEFAULT_OUTPUT_DIR = "AI_Fund_Research/output"

CATEGORY_ORDER = [
    "Money Market",
    "Enhanced Cash",
    "Short Duration Bond",
    "Investment Grade Bond",
    "Global Bond SGD Hedged",
    "Income",
    "Conservative Multi-Asset",
    "Balanced",
    "Dividend / Equity Income",
    "Low Volatility Equity",
    "Other / Review",
]

MIN_FAMILY_TARGETS = {
    "Money Market": 5,
    "Enhanced Cash": 5,
    "Short Duration Bond": 8,
    "Investment Grade Bond": 6,
    "Global Bond SGD Hedged": 8,
    "Income": 10,
    "Conservative Multi-Asset": 5,
    "Balanced": 8,
    "Dividend / Equity Income": 8,
    "Low Volatility Equity": 5,
}


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def canonical_family(name: str, symbol: str) -> str:
    text = normalize_space(name).lower()
    if not text:
        return symbol.upper()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\b(class|cl|share class|acc|dist|dis|div|monthly|income|inc|mdis|idiv|irc|icdiv)\b", " ", text)
    text = re.sub(r"\b(a|b|c|d|e|i|inst|institutional|retail|sgd|usd|eur|hkd|aud|hedged|hdg|h2|h2-sgd)\b", " ", text)
    text = re.sub(r"\b[0-9]+(?:\.[0-9]+)?\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = normalize_space(text)
    return text or symbol.upper()


def classify_category(row: pd.Series) -> str:
    text = " ".join(
        normalize_space(row.get(col, ""))
        for col in ["symbol", "name", "type", "retirement_type_hint", "risk_group", "source_query", "notes"]
        if col in row.index
    ).lower()
    if any(token in text for token in ["money market", "liquidity"]):
        return "Money Market"
    if any(token in text for token in ["enhanced cash", "cash plus", "sgd fund", "cash_or_short_duration"]):
        return "Enhanced Cash"
    if any(token in text for token in ["short duration", "short term", "short-duration"]):
        return "Short Duration Bond"
    if any(token in text for token in ["investment grade", "ig bond", "corporate bond", "government bond"]):
        return "Investment Grade Bond"
    if "bond" in text and any(token in text for token in ["global", "pimco", "income fund"]) and any(token in text for token in ["sgd hedged", "hedged"]):
        return "Global Bond SGD Hedged"
    if any(token in text for token in ["low volatility", "minimum volatility", "min vol"]):
        return "Low Volatility Equity"
    if any(token in text for token in ["dividend", "equity income"]):
        return "Dividend / Equity Income"
    if any(token in text for token in ["conservative", "defensive allocation"]):
        return "Conservative Multi-Asset"
    if any(token in text for token in ["balanced", "bridge"]):
        return "Balanced"
    if any(token in text for token in ["income", "multi asset", "multi-asset", "allocation"]):
        return "Income"
    return "Other / Review"


def load_csvs(patterns: list[str]) -> pd.DataFrame:
    frames = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches and Path(pattern).exists():
            matches = [pattern]
        for path in matches:
            df = pd.read_csv(path)
            df["source_file"] = path
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def yesish(value: Any) -> bool:
    return str(value).strip().lower() in {"yes", "true", "1", "y", "yes_review"}


def build_pool(discovery: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if not discovery.empty:
        frames.append(discovery.copy())
    if not universe.empty:
        prod = universe.copy()
        prod["status"] = "production"
        prod["already_in_universe"] = "True"
        prod["eligible_hint"] = prod.apply(
            lambda r: "yes_review" if yesish(r.get("sgd_hedged")) or str(r.get("currency", "")).upper() == "SGD" else "no_or_unknown",
            axis=1,
        )
        prod["source_query"] = "production_universe"
        frames.append(prod)
    if not frames:
        return pd.DataFrame()
    pool = pd.concat(frames, ignore_index=True, sort=False)
    pool["symbol"] = pool["symbol"].astype(str).str.upper().str.strip()
    pool = pool[pool["symbol"].ne("")]
    pool["name"] = pool.get("name", pool["symbol"]).fillna(pool["symbol"]).map(normalize_space)
    pool["currency"] = pool.get("currency", "").fillna("").astype(str).str.upper()
    pool["likely_hedged_bool"] = pool.apply(
        lambda r: yesish(r.get("likely_hedged")) or "hedged" in f"{r.get('name', '')} {r.get('type', '')} {r.get('notes', '')}".lower(),
        axis=1,
    )
    pool["sgd_denom_bool"] = pool["currency"].eq("SGD") | pool["symbol"].str.endswith(".SI") | pool["name"].str.lower().str.contains(r"\bsgd\b", regex=True)
    pool["sgd_hedged_bool"] = pool["likely_hedged_bool"] | pool.get("sgd_hedged", "").fillna("").astype(str).str.lower().isin(["yes", "true"])
    pool["sg_retail_accessible_bool"] = pool["symbol"].str.endswith(".SI") | pool.get("already_in_universe", "").fillna("").astype(str).str.lower().eq("true")
    pool["category"] = pool.apply(classify_category, axis=1)
    pool["family_key"] = pool.apply(lambda r: canonical_family(r.get("name", ""), r.get("symbol", "")), axis=1)
    pool["has_nav_bool"] = pool.get("status", "").fillna("").astype(str).str.lower().isin(["ok", "production"])
    pool["history_years_num"] = pd.to_numeric(pool.get("history_years", pd.Series([None] * len(pool))), errors="coerce")
    pool = pool.sort_values(
        ["has_nav_bool", "sgd_denom_bool", "sgd_hedged_bool", "history_years_num"],
        ascending=[False, False, False, False],
    )
    return pool.drop_duplicates("symbol", keep="first").reset_index(drop=True)


def write_outputs(pool: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "fund_discovery_pool_normalized.csv"
    summary_path = output_dir / "fund_discovery_pool_summary.md"
    queries_path = output_dir / "fund_discovery_target_queries.txt"

    if pool.empty:
        summary_path.write_text("# Fund Discovery Pool Summary\n\nNo discovery rows found.\n", encoding="utf-8")
        return {"detail_csv": str(detail_path), "summary_md": str(summary_path), "total_rows": 0}

    pool.to_csv(detail_path, index=False, quoting=csv.QUOTE_MINIMAL)

    unique_families = pool.drop_duplicates("family_key")
    category_rows = []
    for category in CATEGORY_ORDER:
        sub = pool[pool["category"].eq(category)]
        fam_sub = unique_families[unique_families["category"].eq(category)]
        target = MIN_FAMILY_TARGETS.get(category)
        gap = "" if target is None else max(target - len(fam_sub), 0)
        category_rows.append(
            {
                "Category": category,
                "ShareClasses": int(len(sub)),
                "Families": int(len(fam_sub)),
                "SGD_or_Hedged": int((sub["sgd_denom_bool"] | sub["sgd_hedged_bool"]).sum()),
                "TargetFamilies": "" if target is None else target,
                "Gap": gap,
            }
        )

    gaps = [r for r in category_rows if isinstance(r["Gap"], int) and r["Gap"] > 0]
    target_queries = [
        f"{row['Category']} Singapore SGD fund" for row in gaps if row["Category"] != "Other / Review"
    ]
    if target_queries:
        queries_path.write_text("\n".join(target_queries) + "\n", encoding="utf-8")
    else:
        queries_path.write_text("# No targeted discovery query required from this audit.\n", encoding="utf-8")

    lines = [
        "# Fund Discovery Pool Summary",
        "",
        f"- Total discovery/pool rows after symbol dedup: `{len(pool)}`",
        f"- Unique fund families after share-class dedup: `{pool['family_key'].nunique()}`",
        f"- Unique share classes / symbols: `{pool['symbol'].nunique()}`",
        f"- SGD-denominated count: `{int(pool['sgd_denom_bool'].sum())}`",
        f"- SGD-hedged count: `{int(pool['sgd_hedged_bool'].sum())}`",
        f"- Singapore-retail-accessible hint count: `{int(pool['sg_retail_accessible_bool'].sum())}`",
        f"- Yahoo NAV / production-verified rows: `{int(pool['has_nav_bool'].sum())}`",
        "",
        "## Category Coverage",
        "",
        "| Category | ShareClasses | Families | SGD_or_Hedged | TargetFamilies | Gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in category_rows:
        lines.append(
            f"| {row['Category']} | {row['ShareClasses']} | {row['Families']} | {row['SGD_or_Hedged']} | {row['TargetFamilies']} | {row['Gap']} |"
        )
    lines.extend(
        [
            "",
            "## Category Gaps",
            "",
        ]
    )
    if gaps:
        for row in gaps:
            lines.append(f"- {row['Category']}: need about `{row['Gap']}` more unique families before broad coverage feels useful.")
    else:
        lines.append("- No material gaps versus the current rough target family counts.")
    lines.extend(
        [
            "",
            "## Targeted Discovery Queries",
            "",
        ]
    )
    if target_queries:
        for query in target_queries:
            lines.append(f"- `{query}`")
    else:
        lines.append("- No targeted discovery query required from this audit.")
    lines.extend(
        [
            "",
            "## Process Note",
            "",
            "- Daily production should update NAV/cache/metrics only.",
            "- Weekly discovery should review this coverage first, then use targeted search only for underrepresented categories.",
            "- Do not auto-promote Level 1 discovery rows into production without review.",
            "",
        ]
    )
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "detail_csv": str(detail_path),
        "summary_md": str(summary_path),
        "target_queries": str(queries_path),
        "total_rows": int(len(pool)),
        "unique_families": int(pool["family_key"].nunique()),
        "targeted_gap_categories": [row["Category"] for row in gaps],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze discovery pool coverage and gaps")
    parser.add_argument("--discovery", action="append", default=None, help="Discovery CSV file or glob. Can be repeated.")
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    discovery_patterns = args.discovery or [DEFAULT_DISCOVERY_GLOB]
    discovery = load_csvs(discovery_patterns)
    universe = pd.read_csv(args.universe) if Path(args.universe).exists() else pd.DataFrame()
    pool = build_pool(discovery, universe)
    result = write_outputs(pool, Path(args.output_dir))
    print(json.dumps({"message": "discovery_pool_analyzed", **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
