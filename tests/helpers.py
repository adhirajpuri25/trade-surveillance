"""Tiny DSL for hand-building order-event scenarios in tests."""
from __future__ import annotations

import itertools

import pandas as pd

T0 = 1_794_816_000_000  # an arbitrary session start (epoch ms)


class Scenario:
    def __init__(self):
        self.rows: list[dict] = []
        self._oid = itertools.count(1)
        self._tid = itertools.count(1)

    def _row(self, t, oid, acct, side, price, qty, etype, cpty=None, tid=None):
        self.rows.append({"timestamp": T0 + int(t * 1000), "order_id": oid, "account_id": acct,
                          "side": side, "price": price, "quantity": qty, "event_type": etype,
                          "counterparty_account_id": cpty, "trade_id": tid})

    def new(self, t: float, acct: str, side: str, price: float, qty: int) -> str:
        oid = f"O{next(self._oid):05d}"
        self._row(t, oid, acct, side, price, qty, "new")
        return oid

    def cancel(self, t: float, oid: str) -> None:
        o = next(r for r in self.rows if r["order_id"] == oid and r["event_type"] == "new")
        filled = sum(r["quantity"] for r in self.rows if r["order_id"] == oid and r["event_type"] == "fill")
        self._row(t, oid, o["account_id"], o["side"], o["price"], o["quantity"] - filled, "cancel")

    def trade(self, t: float, buyer: str, seller: str, price: float, qty: int,
              resting: str | None = None, aggressor_side: str = "buy") -> tuple[str, str]:
        """An aggressor order that trades `qty` against `resting` (an existing
        order id) or, if None, against a fresh resting order from the other party."""
        tid = f"T{next(self._tid):05d}"
        rest_side = "sell" if aggressor_side == "buy" else "buy"
        rest_acct = seller if aggressor_side == "buy" else buyer
        agg_acct = buyer if aggressor_side == "buy" else seller
        if resting is None:
            resting = self.new(t, rest_acct, rest_side, price, qty)
        agg = self.new(t, agg_acct, aggressor_side, price, qty)
        self._row(t, resting, rest_acct, rest_side, price, qty, "fill", agg_acct, tid)
        self._row(t, agg, agg_acct, aggressor_side, price, qty, "fill", rest_acct, tid)
        return resting, agg

    def background(self, acct: str, n: int, qty: int, side: str = "buy", price: float = 74.30,
                   start: float = 0.0) -> None:
        """n ordinary small orders, deep in the book, canceled after a minute.
        Gives the account a measurable 'normal' order size."""
        for i in range(n):
            oid = self.new(start + i * 0.5, acct, side, price, qty)
            self.cancel(start + i * 0.5 + 60, oid)

    def quote(self, acct: str = "MM", bid: float = 74.50, ask: float = 74.51, qty: int = 10) -> None:
        """Resting liquidity from another account so the book has a touch."""
        self.new(0, acct, "buy", bid, qty)
        self.new(0, acct, "sell", ask, qty)

    def events(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rows)
        df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
        df.insert(0, "event_id", range(len(df)))
        return df
