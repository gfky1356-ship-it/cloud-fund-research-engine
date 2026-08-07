import pandas as pd

from fund_research_engine import cagr, max_drawdown


def test_max_drawdown_uses_peak_to_trough():
    prices = pd.Series([100.0, 105.0, 101.0, 90.0, 96.0])
    assert round(max_drawdown(prices), 2) == -14.29


def test_cagr_one_year_double():
    prices = pd.Series([100.0, 200.0])
    dates = pd.Series([pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")])
    assert 99.0 < cagr(prices, dates) < 101.0
