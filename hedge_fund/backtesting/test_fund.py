"""backtest_fund transaction-cost tests — total_costs metric and CycleRecord fields."""

import pytest

from hedge_fund.backtesting.fund import backtest_fund, FundBacktestMetrics
from hedge_fund.data.models import Price
from hedge_fund.fund.spec import Fund, FundSpec
from hedge_fund.models import Signal


# ---------------------------------------------------------------------------
# Fakes (same pattern as existing test_fund.py)
# ---------------------------------------------------------------------------

class FakeDataClient:
    """Canned closes per ticker per date: {ticker: {date: close}}."""

    def __init__(self, series):
        self._series = series

    def get_prices(self, ticker, start_date, end_date, **kwargs):
        days = self._series.get(ticker, {})
        return [
            Price(open=close, close=close, high=close, low=close,
                  volume=1000, time=f"{day}T00:00:00Z")
            for day, close in sorted(days.items())
            if start_date <= day <= end_date
        ]


class FakeAnalyst:
    """Fixed conviction per ticker, on every date."""

    def __init__(self, name, views=None):
        self._name = name
        self._views = views or {}

    @property
    def name(self):
        return self._name

    def predict(self, ticker, date, data_client):
        return Signal(model_name=self._name, ticker=ticker, date=date,
                      value=self._views.get(ticker, 0.0))


def _spec(**overrides):
    base = dict(
        name="test-fund",
        strategies=[{"name": "solo", "models": [{"name": "a"}]}],
        risk={"max_position_pct": 1.0, "max_gross_exposure": 1.0},
        capital=100_000.0,
        rebalance="weekly",
    )
    return FundSpec(**{**base, **overrides})


FRIDAYS = ["2024-06-07", "2024-06-14", "2024-06-21"]

SERIES = {
    "SPY": {day: close for day, close in
            zip(FRIDAYS, [100.0, 102.0, 101.0])},
    "AAPL": {day: close for day, close in
             zip(FRIDAYS, [200.0, 210.0, 190.0])},
}


# ---------------------------------------------------------------------------
# total_costs metric
# ---------------------------------------------------------------------------

def test_total_costs_computed_correctly():
    spec = _spec()
    fund = Fund(spec, models={"solo": [FakeAnalyst("a", views={"AAPL": 1.0})]})

    commission_per_share = 0.01
    slippage_bps = 10.0

    result = backtest_fund(
        fund, "2024-06-03", "2024-06-21",
        FakeDataClient(SERIES), ["AAPL"],
        commission_per_share=commission_per_share,
        slippage_bps=slippage_bps,
    )

    # Week 1: buy 500 shares @ reference 200.
    #   fill_price = 200 * (1 + 10/10000) = 200.20
    #   commission = 0.01 * 500 = 5.0
    #   slippage_cost = |200.20 - 200.0| * 500 = 0.20 * 500 = 100.0
    # Weeks 2 and 3: the target remains 500 shares so no orders are placed.
    # total_costs = sum(commissions) + sum(slippage_costs) = 5.0 + 100.0 = 105.0

    assert result.metrics.total_costs == pytest.approx(105.0)


def test_total_costs_zero_with_no_costs():
    spec = _spec()
    fund = Fund(spec, models={"solo": [FakeAnalyst("a", views={"AAPL": 1.0})]})

    result = backtest_fund(
        fund, "2024-06-03", "2024-06-21",
        FakeDataClient(SERIES), ["AAPL"],
    )

    assert result.metrics.total_costs == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CycleRecord.total_commission
# ---------------------------------------------------------------------------

def test_cycle_record_total_commission():
    spec = _spec()
    fund = Fund(spec, models={"solo": [FakeAnalyst("a", views={"AAPL": 1.0})]})

    result = backtest_fund(
        fund, "2024-06-03", "2024-06-21",
        FakeDataClient(SERIES), ["AAPL"],
        commission_per_share=0.01,
        slippage_bps=0.0,
    )

    # Week 1 buys 500 shares: commission = 0.01 * 500 = 5.0
    assert result.records[0].total_commission == pytest.approx(5.0)
    # Weeks 2-3: no orders, so no commission
    assert result.records[1].total_commission == pytest.approx(0.0)
    assert result.records[2].total_commission == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# backtest_fund forwards cost params to SimBroker
# ---------------------------------------------------------------------------

def test_backtest_fund_accepts_cost_parameters():
    spec = _spec()
    fund = Fund(spec, models={"solo": [FakeAnalyst("a", views={"AAPL": 1.0})]})

    result = backtest_fund(
        fund, "2024-06-03", "2024-06-21",
        FakeDataClient(SERIES), ["AAPL"],
        commission_per_share=0.005,
        slippage_bps=5.0,
    )

    assert len(result.records) == 3
    assert result.records[0].fills[0].price != 200.0


def test_backtest_fund_default_costs_preserve_behavior():
    spec = _spec()
    fund = Fund(spec, models={"solo": [FakeAnalyst("a", views={"AAPL": 1.0})]})

    result = backtest_fund(
        fund, "2024-06-03", "2024-06-21",
        FakeDataClient(SERIES), ["AAPL"],
    )

    assert result.records[0].fills[0].price == pytest.approx(200.0)
    assert result.records[0].fills[0].commission == pytest.approx(0.0)
    assert result.nav == [100_000.0, 105_000.0, 95_000.0]


# ---------------------------------------------------------------------------
# FundBacktestMetrics has total_costs field
# ---------------------------------------------------------------------------

def test_fund_backtest_metrics_has_total_costs_field():
    spec = _spec()
    fund = Fund(spec, models={"solo": [FakeAnalyst("a", views={"AAPL": 1.0})]})

    result = backtest_fund(
        fund, "2024-06-03", "2024-06-21",
        FakeDataClient(SERIES), ["AAPL"],
        commission_per_share=0.01,
        slippage_bps=10.0,
    )

    assert hasattr(result.metrics, "total_costs")
    assert isinstance(result.metrics.total_costs, float)


def test_total_costs_with_only_commission():
    spec = _spec()
    fund = Fund(spec, models={"solo": [FakeAnalyst("a", views={"AAPL": 1.0})]})

    result = backtest_fund(
        fund, "2024-06-03", "2024-06-21",
        FakeDataClient(SERIES), ["AAPL"],
        commission_per_share=0.02,
        slippage_bps=0.0,
    )

    # Week 1: buy 500 shares, commission = 0.02 * 500 = 10.0
    # No slippage cost since bps = 0
    # total_costs = 10.0
    assert result.metrics.total_costs == pytest.approx(10.0)


def test_total_costs_with_only_slippage():
    spec = _spec()
    fund = Fund(spec, models={"solo": [FakeAnalyst("a", views={"AAPL": 1.0})]})

    result = backtest_fund(
        fund, "2024-06-03", "2024-06-21",
        FakeDataClient(SERIES), ["AAPL"],
        commission_per_share=0.0,
        slippage_bps=10.0,
    )

    # Week 1: buy 500 shares @ reference 200
    #   fill_price = 200 * (1 + 10/10000) = 200.20
    #   slippage_cost = |200.20 - 200.0| * 500 = 100.0
    # No commission
    # total_costs = 100.0
    assert result.metrics.total_costs == pytest.approx(100.0)
