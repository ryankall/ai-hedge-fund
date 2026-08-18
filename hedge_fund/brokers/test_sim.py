"""SimBroker transaction-cost tests — slippage, commission, and backward compat."""

import pytest

from hedge_fund.brokers.models import Order, Fill
from hedge_fund.brokers.sim import SimBroker


# ---------------------------------------------------------------------------
# Slippage
# ---------------------------------------------------------------------------

def test_buy_slippage_adjusts_fill_price_up():
    broker = SimBroker(cash=100_000.0, slippage_bps=50.0)
    order = Order(ticker="AAPL", side="buy", quantity=10, price=100.0)
    fill = broker.place_order(order)
    # 100 * (1 + 50/10000) = 100 * 1.005 = 100.50
    assert fill.price == pytest.approx(100.50)


def test_sell_slippage_adjusts_fill_price_down():
    broker = SimBroker(cash=100_000.0, slippage_bps=50.0)
    broker.place_order(Order(ticker="AAPL", side="buy", quantity=10, price=100.0))
    sell_fill = broker.place_order(Order(ticker="AAPL", side="sell", quantity=10, price=100.0))
    # 100 * (1 - 50/10000) = 100 * 0.995 = 99.50
    assert sell_fill.price == pytest.approx(99.50)


def test_slippage_affects_cash_on_buy():
    broker = SimBroker(cash=100_000.0, slippage_bps=50.0)
    broker.place_order(Order(ticker="AAPL", side="buy", quantity=10, price=100.0))
    # fill_price = 100.50, cash -= 100.50 * 10 = 1005.0
    assert broker.cash() == pytest.approx(100_000.0 - 1_005.0)


# ---------------------------------------------------------------------------
# Commission
# ---------------------------------------------------------------------------

def test_commission_deducted_from_cash_on_buy():
    broker = SimBroker(cash=100_000.0, commission_per_share=0.01)
    broker.place_order(Order(ticker="AAPL", side="buy", quantity=100, price=200.0))
    # cash -= (200 * 100 + 0.01 * 100) = 20_001.00
    assert broker.cash() == pytest.approx(100_000.0 - 20_001.0)


def test_commission_deducted_from_cash_on_sell():
    broker = SimBroker(cash=100_000.0, commission_per_share=0.01)
    broker.place_order(Order(ticker="AAPL", side="buy", quantity=100, price=200.0))
    broker.place_order(Order(ticker="AAPL", side="sell", quantity=100, price=200.0))
    # buy:  cash -= 200*100 + 0.01*100 = 20_001
    # sell: cash += 200*100 - 0.01*100 = 19_999
    assert broker.cash() == pytest.approx(100_000.0 - 20_001.0 + 19_999.0)


def test_commission_appears_on_fill():
    broker = SimBroker(cash=100_000.0, commission_per_share=0.005)
    fill = broker.place_order(Order(ticker="AAPL", side="buy", quantity=200, price=50.0))
    # commission = 0.005 * 200 = 1.0
    assert fill.commission == pytest.approx(1.0)


def test_commission_field_equals_per_share_times_quantity():
    broker = SimBroker(cash=50_000.0, commission_per_share=0.02)
    fill = broker.place_order(Order(ticker="MSFT", side="buy", quantity=75, price=300.0))
    assert fill.commission == pytest.approx(0.02 * 75)


# ---------------------------------------------------------------------------
# Combined costs — round-trip trade
# ---------------------------------------------------------------------------

def test_round_trip_costs_erode_cash():
    broker = SimBroker(cash=100_000.0, commission_per_share=0.01, slippage_bps=10.0)
    # Buy 100 @ reference 100 with 10 bps slippage and $0.01/share commission
    broker.place_order(Order(ticker="AAPL", side="buy", quantity=100, price=100.0))
    # fill_price_buy = 100 * (1 + 10/10000) = 100.10
    # cash -= 100.10 * 100 + 0.01 * 100 = 10_010 + 1 = 10_011
    # cash after buy = 100_000 - 10_011 = 89_989

    # Sell 100 @ reference 100 with 10 bps slippage and $0.01/share commission
    broker.place_order(Order(ticker="AAPL", side="sell", quantity=100, price=100.0))
    # fill_price_sell = 100 * (1 - 10/10000) = 99.90
    # cash += 99.90 * 100 - 0.01 * 100 = 9_990 - 1 = 9_989
    # cash after sell = 89_989 + 9_989 = 99_978

    assert broker.cash() < 100_000.0
    assert broker.cash() == pytest.approx(99_978.0)


def test_round_trip_no_position_left():
    broker = SimBroker(cash=100_000.0, commission_per_share=0.01, slippage_bps=10.0)
    broker.place_order(Order(ticker="AAPL", side="buy", quantity=100, price=100.0))
    broker.place_order(Order(ticker="AAPL", side="sell", quantity=100, price=100.0))
    assert broker.positions() == {}


# ---------------------------------------------------------------------------
# Zero-cost backward compatibility
# ---------------------------------------------------------------------------

def test_zero_cost_fill_price_equals_order_price():
    broker = SimBroker(cash=10_000.0)
    fill = broker.place_order(Order(ticker="AAPL", side="buy", quantity=10, price=100.0))
    assert fill.price == pytest.approx(100.0)
    assert fill.commission == pytest.approx(0.0)


def test_zero_cost_cash_changes_are_exact():
    broker = SimBroker(cash=10_000.0)
    broker.place_order(Order(ticker="AAPL", side="buy", quantity=10, price=100.0))
    assert broker.cash() == pytest.approx(9_000.0)
    broker.place_order(Order(ticker="AAPL", side="sell", quantity=10, price=110.0))
    assert broker.cash() == pytest.approx(9_000.0 + 1_100.0)


def test_zero_cost_preserves_existing_behavior():
    broker = SimBroker(cash=10_000.0)
    fill = broker.place_order(Order(ticker="AAPL", side="buy", quantity=5, price=100.0))
    assert broker.positions()["AAPL"].shares == 5
    assert broker.cash() == pytest.approx(9_500.0)
    assert fill.price == 100.0
    assert fill.commission == 0.0
    assert fill.quantity == 5


# ---------------------------------------------------------------------------
# Fill model: commission field exists and defaults to 0
# ---------------------------------------------------------------------------

def test_fill_model_commission_default():
    fill = Fill(ticker="X", side="buy", quantity=1, price=10.0)
    assert fill.commission == 0.0


def test_fill_model_commission_set():
    fill = Fill(ticker="X", side="buy", quantity=1, price=10.0, commission=2.5)
    assert fill.commission == 2.5


# ---------------------------------------------------------------------------
# SimBroker constructor accepts new parameters
# ---------------------------------------------------------------------------

def test_simbroker_accepts_commission_and_slippage():
    broker = SimBroker(cash=10_000.0, commission_per_share=0.005, slippage_bps=5.0)
    assert broker.cash() == pytest.approx(10_000.0)


def test_simbroker_defaults_preserve_signature():
    broker = SimBroker(cash=5_000.0)
    fill = broker.place_order(Order(ticker="MSFT", side="buy", quantity=1, price=300.0))
    assert fill.price == pytest.approx(300.0)
    assert fill.commission == pytest.approx(0.0)
    assert broker.cash() == pytest.approx(4_700.0)


# ---------------------------------------------------------------------------
# Slippage with different quantities
# ---------------------------------------------------------------------------

def test_buy_slippage_large_quantity():
    broker = SimBroker(cash=1_000_000.0, slippage_bps=25.0)
    fill = broker.place_order(Order(ticker="TSLA", side="buy", quantity=500, price=250.0))
    # 250 * (1 + 25/10000) = 250 * 1.0025 = 250.625
    assert fill.price == pytest.approx(250.625)
    # cash -= 250.625 * 500 = 125_312.50
    assert broker.cash() == pytest.approx(1_000_000.0 - 125_312.50)


def test_sell_slippage_reduces_proceeds():
    broker = SimBroker(cash=200_000.0, slippage_bps=20.0)
    broker.place_order(Order(ticker="GOOG", side="buy", quantity=50, price=150.0))
    sell_fill = broker.place_order(Order(ticker="GOOG", side="sell", quantity=50, price=150.0))
    # sell fill_price = 150 * (1 - 20/10000) = 150 * 0.998 = 149.70
    assert sell_fill.price == pytest.approx(149.70)


# ---------------------------------------------------------------------------
# Commission on sell side
# ---------------------------------------------------------------------------

def test_commission_on_sell_reduces_cash_proceeds():
    broker = SimBroker(cash=50_000.0, commission_per_share=0.05)
    broker.place_order(Order(ticker="XYZ", side="buy", quantity=200, price=100.0))
    # cash after buy = 50_000 - (100*200 + 0.05*200) = 50_000 - 20_010 = 29_990
    sell_fill = broker.place_order(Order(ticker="XYZ", side="sell", quantity=200, price=105.0))
    # cash += 105*200 - 0.05*200 = 21_000 - 10 = 20_990
    assert broker.cash() == pytest.approx(29_990.0 + 20_990.0)
    assert sell_fill.commission == pytest.approx(0.05 * 200)
