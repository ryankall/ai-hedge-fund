"""SimBroker — deterministic simulated broker for backtests.

Fills every order completely, exactly at the order's reference price. That
determinism is the point: given the same orders, a backtest replays to the
same book. Slippage/costs are a declared future addition inside place_order,
where they change fills without touching the pipeline.

Margin is not modeled: cash may go negative and stays visible. With an
unlevered mandate (gross_target <= 1), sells-before-buys ordering, and
floor-toward-zero sizing, a long book won't get there — but nothing here
pretends to enforce it.
"""

from __future__ import annotations

from hedge_fund.brokers.models import Fill, Order, Position


class SimBroker:
    """In-memory broker: signed positions plus a cash balance."""

    def __init__(
        self,
        cash: float,
        commission_per_share: float = 0.0,
        slippage_bps: float = 0.0,
    ) -> None:
        self._cash = cash
        self._commission_per_share = commission_per_share
        self._slippage_bps = slippage_bps
        self._shares: dict[str, int] = {}

    def positions(self) -> dict[str, Position]:
        return {
            t: Position(ticker=t, shares=s)
            for t, s in self._shares.items()
            if s != 0
        }

    def cash(self) -> float:
        return self._cash

    def place_order(self, order: Order) -> Fill:
        if order.price <= 0:
            raise ValueError(
                f"cannot fill {order.ticker} at price {order.price} — "
                "the caller must price every order"
            )

        if order.side == "buy":
            fill_price = order.price * (1 + self._slippage_bps / 10000)
        else:
            fill_price = order.price * (1 - self._slippage_bps / 10000)

        commission = self._commission_per_share * order.quantity

        if order.side == "buy":
            self._shares[order.ticker] = self._shares.get(order.ticker, 0) + order.quantity
            self._cash -= order.quantity * fill_price + commission
        else:
            self._shares[order.ticker] = self._shares.get(order.ticker, 0) - order.quantity
            self._cash += order.quantity * fill_price - commission

        if self._shares[order.ticker] == 0:
            del self._shares[order.ticker]

        return Fill(
            ticker=order.ticker,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
        )
