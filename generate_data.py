"""
Synthetic order-book event generator for one futures contract (ICE Brent-style).

Real order-level futures data with account IDs is confidential (only the
exchange and regulators see it), so we simulate a trading day:

* A small price-time-priority matching engine (a limit order book). Every fill
  comes out of actual order matching, so fills, cancels and price moves are
  consistent with each other.
* ~50 normal accounts: market makers quoting both sides, ordinary traders, and
  a few large hedgers.
* Injected, labeled manipulation: wash trades, spoofing and layering.

Outputs (in data/):
    events.csv         one row per order event (new / cancel / fill)
    labels.csv         ground truth: one row per injected manipulation incident
    account_links.csv  known beneficial-ownership links between accounts
                       (exchange reference data, used by the wash detector)

Usage:
    python generate_data.py [--seed 42] [--out data]
"""
from __future__ import annotations

import argparse
import heapq
import itertools
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from config import TICK_SIZE

# ----------------------------------------------------------------------------
# Simulation parameters (these shape the synthetic market, not the detectors)
# ----------------------------------------------------------------------------
SEED = 42
SESSION_START = pd.Timestamp("2026-11-16 08:00:00")  # one London trading day
SESSION_HOURS = 8
START_PRICE = 74.50  # $/bbl

N_MARKET_MAKERS = 5
N_TRADERS = 42
N_HEDGERS = 3  # 5 + 42 + 3 = 50 normal accounts

FAIR_VALUE_VOL_TICKS = 0.6  # random-walk std dev of "fair value" per second
AGGRESSOR_FAIR_SENSITIVITY = 0.8  # how strongly takers chase fair value
AGGRESSOR_IMBALANCE_SENSITIVITY = 3.0  # how strongly takers react to book imbalance
IMBALANCE_DEPTH_TICKS = 3  # levels counted when measuring book imbalance

N_SPOOF_INCIDENTS_PER_ACCOUNT = 15
N_LAYER_INCIDENTS_PER_ACCOUNT = 12
N_SELF_WASH_INCIDENTS = 8
N_LINKED_WASH_INCIDENTS = 6
N_HIDDEN_WASH_INCIDENTS = 5

EVENT_COLUMNS = [
    "timestamp", "order_id", "account_id", "side", "price", "quantity",
    "event_type", "counterparty_account_id", "trade_id",
]


@dataclass
class Order:
    order_id: str
    account: str
    side: str  # "buy" | "sell"
    price: int  # in ticks
    qty: int
    remaining: int
    t_new: int
    live: bool = True


@dataclass
class Account:
    account_id: str
    role: str
    median_size: float = 3.0
    rate_per_s: float = 1 / 60  # background order arrival rate
    p_aggressive: float = 0.3
    mean_life_s: float = 30.0
    p_large: float = 0.0  # chance an order is unusually large (innocent)
    direction: str | None = None  # hedgers trade one way only
    levels: int = 2  # market makers: quote levels per side
    half_spread: float = 1.2  # market makers: ticks from fair value
    requote_s: float = 15.0  # market makers: mean seconds between requotes
    requote_token: int = 0
    requote_pending: bool = False
    smp: bool = True  # uses the exchange's self-match prevention
    live_orders: set = field(default_factory=set)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class MarketSimulator:
    def __init__(self, seed: int = SEED):
        self.rng = np.random.default_rng(seed)
        self.t0 = int(SESSION_START.value // 1_000_000)  # epoch ms
        self.t_end = self.t0 + SESSION_HOURS * 3600 * 1000
        self.now = self.t0
        self.queue: list = []
        self.seq = itertools.count()
        self.order_ids = itertools.count(1)
        self.trade_ids = itertools.count(1)
        self.events: list[tuple] = []
        self.orders: dict[str, Order] = {}
        self.books = {"buy": {}, "sell": {}}  # side -> price -> deque[order_id]
        self.level_qty = {"buy": {}, "sell": {}}  # side -> price -> resting qty
        self.order_hooks: dict[str, callable] = {}  # order_id -> on_fill(order, qty)
        self.fair = START_PRICE / TICK_SIZE
        self.last_trade = int(round(self.fair))
        self.accounts: dict[str, Account] = {}
        self.links: list[tuple[str, str, str]] = []
        self.incidents: list[dict] = []

    # ------------------------------------------------------------- scheduling
    def schedule(self, t: float, fn, *args) -> None:
        heapq.heappush(self.queue, (int(t), next(self.seq), fn, args))

    def after(self, delay_ms: float, fn, *args) -> None:
        self.schedule(self.now + max(0, delay_ms), fn, *args)

    def run(self) -> None:
        while self.queue and self.queue[0][0] <= self.t_end:
            t, _, fn, args = heapq.heappop(self.queue)
            self.now = t
            fn(*args)

    # ------------------------------------------------------------- book state
    def best(self, side: str) -> int | None:
        book = self.books[side]
        if not book:
            return None
        return max(book) if side == "buy" else min(book)

    def touch(self) -> tuple[int, int]:
        """Best bid/ask in ticks, synthesised around the last trade if a side is empty."""
        bb, ba = self.best("buy"), self.best("sell")
        if bb is None and ba is None:
            bb, ba = self.last_trade - 1, self.last_trade + 1
        elif bb is None:
            bb = ba - 2
        elif ba is None:
            ba = bb + 2
        return bb, ba

    def imbalance(self) -> float:
        bb, ba = self.touch()
        bid = sum(q for p, q in self.level_qty["buy"].items() if p > bb - IMBALANCE_DEPTH_TICKS)
        ask = sum(q for p, q in self.level_qty["sell"].items() if p < ba + IMBALANCE_DEPTH_TICKS)
        return 0.0 if bid + ask == 0 else (bid - ask) / (bid + ask)

    # ----------------------------------------------------------- order entry
    def _emit(self, o: Order, etype: str, price: int, qty: int, cpty=None, trade_id=None):
        self.events.append((
            self.now, o.order_id, o.account, o.side, round(price * TICK_SIZE, 2), qty,
            etype, cpty, trade_id,
        ))

    def submit(self, account: str, side: str, price: int, qty: int, smp: bool = False) -> Order:
        """Enter a limit order. Marketable orders match immediately (price-time priority).

        smp=True turns on self-match prevention: matching stops rather than trade
        against the account's own resting order.
        """
        qty = int(max(1, qty))
        o = Order(f"O{next(self.order_ids):07d}", account, side, int(price), qty, qty, self.now)
        self.orders[o.order_id] = o
        self._emit(o, "new", o.price, qty)
        self._match(o, smp)
        if o.remaining > 0:
            self.books[side].setdefault(o.price, deque()).append(o.order_id)
            self.level_qty[side][o.price] = self.level_qty[side].get(o.price, 0) + o.remaining
            self.accounts[account].live_orders.add(o.order_id)
        else:
            o.live = False
        return o

    def _match(self, o: Order, smp: bool) -> None:
        opp = "sell" if o.side == "buy" else "buy"
        book = self.books[opp]
        while o.remaining > 0 and book:
            best = self.best(opp)
            if (o.side == "buy" and best > o.price) or (o.side == "sell" and best < o.price):
                break
            q = book[best]
            rest = self.orders[q[0]]
            if smp and rest.account == o.account:
                break
            fill = min(o.remaining, rest.remaining)
            tid = f"T{next(self.trade_ids):07d}"
            rest.remaining -= fill
            o.remaining -= fill
            self.level_qty[opp][best] -= fill
            self._emit(rest, "fill", best, fill, o.account, tid)
            self._emit(o, "fill", best, fill, rest.account, tid)
            self.last_trade = best
            if rest.remaining == 0:
                q.popleft()
                rest.live = False
                self.accounts[rest.account].live_orders.discard(rest.order_id)
                if not q:
                    del book[best]
                    del self.level_qty[opp][best]
            for filled in (rest, o):
                if filled.order_id in self.order_hooks:
                    self.after(0, self.order_hooks[filled.order_id], filled, fill)
                acct = self.accounts[filled.account]
                if acct.role == "market_maker" and not acct.requote_pending:
                    acct.requote_pending = True  # requote once per burst of fills
                    self.after(self.rng.uniform(200, 1500), self.mm_requote, acct.account_id, True)

    def cancel(self, order_id: str) -> bool:
        o = self.orders.get(order_id)
        if o is None or not o.live:
            return False
        q = self.books[o.side][o.price]
        q.remove(order_id)
        self.level_qty[o.side][o.price] -= o.remaining
        if not q:
            del self.books[o.side][o.price]
            del self.level_qty[o.side][o.price]
        o.live = False
        self.accounts[o.account].live_orders.discard(order_id)
        self._emit(o, "cancel", o.price, o.remaining)
        return True

    # --------------------------------------------------------- normal agents
    def draw_size(self, acct: Account) -> int:
        return max(1, int(round(self.rng.lognormal(math.log(acct.median_size), 0.5))))

    def fair_value_step(self) -> None:
        self.fair += self.rng.normal(0, FAIR_VALUE_VOL_TICKS)
        # keep fair value from drifting far from where the market actually trades
        self.fair += 0.02 * (self.last_trade - self.fair)
        self.after(1000, self.fair_value_step)

    def trader_action(self, account_id: str) -> None:
        acct = self.accounts[account_id]
        self.after(self.rng.exponential(1000 / acct.rate_per_s), self.trader_action, account_id)
        bb, ba = self.touch()
        mid = (bb + ba) / 2
        qty = self.draw_size(acct)
        large = self.rng.random() < acct.p_large
        if large:
            qty = int(qty * self.rng.uniform(4, 15))

        if self.rng.random() < acct.p_aggressive and not large:
            # Taker: buy/sell pressure depends on fair value vs. mid and on
            # visible book imbalance (this is what spoofers try to exploit).
            logit = (AGGRESSOR_FAIR_SENSITIVITY * (self.fair - mid)
                     + AGGRESSOR_IMBALANCE_SENSITIVITY * self.imbalance())
            side = acct.direction or ("buy" if self.rng.random() < sigmoid(logit) else "sell")
            price = ba if side == "buy" else bb
            o = self.submit(account_id, side, price, qty, smp=acct.smp)
            if o.live:  # immediate-or-cancel: pull any unfilled remainder
                self.after(self.rng.uniform(1, 30), self.cancel, o.order_id)
            return

        # Maker: rest a limit order at or near the touch.
        side = acct.direction or (
            "buy" if self.rng.random() < sigmoid(0.4 * (self.fair - mid)) else "sell")
        if ba - bb >= 2 and self.rng.random() < 0.15:
            price = bb + 1 if side == "buy" else ba - 1  # improve the price
        else:
            d = int(self.rng.geometric(0.45)) - 1  # 0 = join the touch
            price = bb - d if side == "buy" else ba + d
        o = self.submit(account_id, side, price, qty, smp=acct.smp)
        if o.live:
            if large and self.rng.random() < 0.5:
                life = self.rng.uniform(0.5, 5.0)  # nervous: big order pulled fast
            elif self.rng.random() < 0.25:
                life = self.rng.uniform(0.3, 5.0)  # quick cancel (re-pricing)
            else:
                life = self.rng.exponential(acct.mean_life_s)
            self.after(life * 1000, self.cancel, o.order_id)

    def mm_requote(self, account_id: str, from_fill: bool = False, token: int | None = None) -> None:
        acct = self.accounts[account_id]
        if from_fill:
            acct.requote_token += 1
            acct.requote_pending = False
        elif token is not None and token != acct.requote_token:
            return  # superseded by a fill-triggered requote
        for oid in sorted(acct.live_orders):
            self.cancel(oid)
        bb, ba = self.touch()
        center = self.fair + self.rng.normal(0, 0.5)
        bid0 = min(math.floor(center - acct.half_spread), (ba - 1) if self.best("sell") else 10**9)
        ask0 = max(math.ceil(center + acct.half_spread), (bb + 1) if self.best("buy") else 0)
        if ask0 <= bid0:
            ask0 = bid0 + 1
        for j in range(acct.levels):
            self.submit(account_id, "buy", bid0 - j, self.draw_size(acct), smp=True)
            self.submit(account_id, "sell", ask0 + j, self.draw_size(acct), smp=True)
        acct.requote_token += 1
        tok = acct.requote_token
        self.after(self.rng.exponential(acct.requote_s * 1000), self.mm_requote, account_id, False, tok)

    # ------------------------------------------------------ manipulation
    def _new_incident(self, pattern: str, accounts: list[str], description: str) -> dict:
        inc = {
            "incident_id": f"{pattern[:4].upper()}-{len(self.incidents) + 1:03d}",
            "pattern": pattern,
            "account_id": accounts[0],
            "related_accounts": ";".join(accounts),
            "order_ids": [],
            "description": description,
        }
        self.incidents.append(inc)
        return inc

    def _cross_spread(self, account_id: str, side: str, qty: int) -> Order:
        bb, ba = self.touch()
        o = self.submit(account_id, side, ba if side == "buy" else bb, qty, smp=True)
        if o.live:
            self.after(5, self.cancel, o.order_id)
        return o

    def inject_spoof(self, account_id: str) -> None:
        """Small genuine order on one side, big fake order(s) on the other,
        pulled as soon as the genuine order fills (Coscia / Sarao style)."""
        acct = self.accounts[account_id]
        gside = "buy" if self.rng.random() < 0.5 else "sell"
        sside = "sell" if gside == "buy" else "buy"
        inc = self._new_incident("spoofing", [account_id],
                                 f"genuine {gside}, spoof {sside}")
        st = {"done": False, "spoofs": [], "pulled": False}
        bb, ba = self.touch()
        gprice = bb if gside == "buy" else ba
        if ba - bb >= 2 and self.rng.random() < 0.5:
            gprice = bb + 1 if gside == "buy" else ba - 1
        g = self.submit(account_id, gside, gprice, self.draw_size(acct), smp=True)
        inc["order_ids"].append(g.order_id)

        def cancel_spoofs():
            st["pulled"] = True
            for i, oid in enumerate(st["spoofs"]):
                self.after(i * self.rng.uniform(1, 20), self.cancel, oid)

        def on_genuine_fill(order, qty):
            if not st["done"]:
                st["done"] = True
                self.cancel(g.order_id)  # don't want more than the planned size
                self.after(self.rng.uniform(150, 1500), cancel_spoofs)

        def place_spoofs():
            n = 1 if self.rng.random() < 0.7 else 2
            for _ in range(n):
                bb2, ba2 = self.touch()
                j = int(self.rng.choice([0, 1, 2], p=[0.2, 0.5, 0.3]))
                price = ba2 + j if sside == "sell" else bb2 - j
                qty = int(round(acct.median_size * self.rng.uniform(4, 20)))
                o = self.submit(account_id, sside, price, qty, smp=True)
                st["spoofs"].append(o.order_id)
                inc["order_ids"].append(o.order_id)
                if st["pulled"]:  # genuine order already filled: pull right away
                    self.after(self.rng.uniform(150, 1500), self.cancel, o.order_id)

        def fallback():
            if st["done"]:
                return
            st["done"] = True
            self.cancel(g.order_id)
            o = self._cross_spread(account_id, gside, g.qty)
            inc["order_ids"].append(o.order_id)
            self.after(self.rng.uniform(100, 800), cancel_spoofs)

        if g.live:
            self.order_hooks[g.order_id] = on_genuine_fill
        else:
            st["done"] = True
        self.after(self.rng.uniform(100, 400), place_spoofs)
        self.after(self.rng.uniform(2500, 4500), fallback)

    def inject_layering(self, account_id: str) -> None:
        """Small genuine order on one side, a stack of orders at several price
        levels on the other side, all canceled together after the genuine fill."""
        acct = self.accounts[account_id]
        gside = "buy" if self.rng.random() < 0.5 else "sell"
        lside = "sell" if gside == "buy" else "buy"
        k = int(self.rng.integers(3, 7))
        inc = self._new_incident("layering", [account_id],
                                 f"genuine {gside}, {k} layers on {lside}")
        st = {"done": False, "layers": [], "pulled": False}
        bb, ba = self.touch()
        g = self.submit(account_id, gside, bb if gside == "buy" else ba,
                        self.draw_size(acct), smp=True)
        inc["order_ids"].append(g.order_id)

        def cancel_layers():
            st["pulled"] = True
            for i, oid in enumerate(st["layers"]):
                self.after(i * self.rng.uniform(1, 15), self.cancel, oid)

        def on_genuine_fill(order, qty):
            if not st["done"]:
                st["done"] = True
                self.cancel(g.order_id)
                self.after(self.rng.uniform(100, 1500), cancel_layers)

        def place_layer(j):
            bb2, ba2 = self.touch()
            start = ba2 + 1 if lside == "sell" else bb2 - 1  # one tick behind the touch
            price = start + j if lside == "sell" else start - j
            qty = int(round(acct.median_size * self.rng.uniform(1.5, 4)))
            o = self.submit(account_id, lside, price, qty, smp=True)
            st["layers"].append(o.order_id)
            inc["order_ids"].append(o.order_id)
            if st["pulled"]:
                self.after(self.rng.uniform(100, 1500), self.cancel, o.order_id)

        def fallback():
            if st["done"]:
                return
            st["done"] = True
            self.cancel(g.order_id)
            o = self._cross_spread(account_id, gside, g.qty)
            inc["order_ids"].append(o.order_id)
            self.after(self.rng.uniform(100, 1200), cancel_layers)

        if g.live:
            self.order_hooks[g.order_id] = on_genuine_fill
        else:
            st["done"] = True
        t = self.rng.uniform(50, 200)
        for j in range(k):
            self.after(t, place_layer, j)
            t += self.rng.uniform(20, 150)
        self.after(t + self.rng.uniform(2500, 4500), fallback)

    def inject_wash(self, sellers: list[str], buyers: list[str], n_trades: int,
                    pattern_note: str, gap_ms: tuple[int, int]) -> None:
        """Execute n_trades crosses where trade i is sellers[i] -> buyers[i].

        Each cross is placed inside the spread so the two orders meet each
        other rather than other traders' resting orders.
        """
        accounts = list(dict.fromkeys(sellers + buyers))
        inc = self._new_incident("wash_trade", accounts, pattern_note)
        st = {"i": 0, "t_start": self.now, "price": None}
        qty = int(self.draw_size(self.accounts[sellers[0]]) * self.rng.uniform(1, 3))

        def attempt():
            if st["i"] >= n_trades or self.now - st["t_start"] > 90_000:
                return
            bb, ba = self.touch()
            if ba - bb < 2:
                self.after(250, attempt)  # wait for a gap in the spread
                return
            p = st["price"] if st["price"] is not None and bb < st["price"] < ba else bb + 1
            st["price"] = p
            i = st["i"]
            first_sells = self.rng.random() < 0.5
            a1, s1 = (sellers[i], "sell") if first_sells else (buyers[i], "buy")
            o1 = self.submit(a1, s1, p, qty)
            inc["order_ids"].append(o1.order_id)
            self.after(self.rng.uniform(5, 40), second, o1, p, first_sells, i)

        def second(o1, p, first_sells, i):
            a2, s2 = (buyers[i], "buy") if first_sells else (sellers[i], "sell")
            o2 = self.submit(a2, s2, p, qty)
            inc["order_ids"].append(o2.order_id)
            for o in (o1, o2):
                if o.live:
                    self.after(300, self.cancel, o.order_id)
            st["i"] += 1
            if st["i"] < n_trades:
                self.after(self.rng.uniform(*gap_ms), attempt)

        attempt()

    # ------------------------------------------------------------- setup
    def setup(self) -> None:
        rng = self.rng
        ids = rng.choice(np.arange(10000, 99999), size=59, replace=False)
        ids = [f"AC{i}" for i in ids]
        it = iter(ids)

        for _ in range(N_MARKET_MAKERS):
            a = Account(next(it), "market_maker", median_size=float(rng.uniform(3, 6)),
                        levels=int(rng.integers(2, 4)), half_spread=float(rng.uniform(0.8, 1.6)),
                        requote_s=float(rng.uniform(10, 25)))
            self.accounts[a.account_id] = a
        for _ in range(N_TRADERS):
            a = Account(next(it), "trader", median_size=float(rng.integers(1, 9)),
                        rate_per_s=float(rng.lognormal(math.log(1 / 75), 0.5)),
                        p_aggressive=float(rng.uniform(0.15, 0.45)),
                        mean_life_s=float(rng.uniform(10, 120)),
                        p_large=float(rng.choice([0.0, 0.01, 0.03], p=[0.5, 0.3, 0.2])),
                        smp=bool(rng.random() < 0.85))
            self.accounts[a.account_id] = a
        for _ in range(N_HEDGERS):
            a = Account(next(it), "hedger", median_size=float(rng.uniform(25, 50)),
                        rate_per_s=1 / 400, p_aggressive=0.2, mean_life_s=300.0,
                        direction=str(rng.choice(["buy", "sell"])))
            self.accounts[a.account_id] = a
        traders = [a for a in self.accounts.values() if a.role == "trader"]

        # Manipulators also trade normally all day -- that's what makes their
        # "normal" order size measurable, and it's how real bad actors look.
        roles = ["spoofer", "spoofer", "layerer", "layerer", "self_washer",
                 "linked_washer", "linked_washer", "hidden_washer", "hidden_washer"]
        manip = {}
        for role in roles:
            a = Account(next(it), role, median_size=float(rng.integers(2, 7)),
                        rate_per_s=1 / 60, p_aggressive=0.3,
                        mean_life_s=float(rng.uniform(20, 60)))
            self.accounts[a.account_id] = a
            manip.setdefault(role, []).append(a.account_id)

        # Exchange reference data: known common ownership. Two pairs of
        # ordinary traders are linked but trade independently (look-alikes).
        self.links = [
            (traders[0].account_id, traders[1].account_id, "same beneficial owner (independent desks)"),
            (traders[2].account_id, traders[3].account_id, "same beneficial owner (independent desks)"),
            (manip["linked_washer"][0], manip["linked_washer"][1], "same beneficial owner"),
        ]

        # Start background processes.
        self.after(0, self.fair_value_step)
        for a in self.accounts.values():
            if a.role == "market_maker":
                self.after(rng.uniform(0, 2000), self.mm_requote, a.account_id, True)
            else:
                self.after(rng.exponential(1000 / a.rate_per_s), self.trader_action, a.account_id)

        # Schedule manipulation, spread across the day (not in the first/last 10 min).
        lo, hi = self.t0 + 600_000, self.t_end - 600_000
        used: list[int] = []

        def pick_time() -> int:
            while True:
                t = int(rng.uniform(lo, hi))
                if all(abs(t - u) > 120_000 for u in used):
                    used.append(t)
                    return t

        for acc in manip["spoofer"]:
            for _ in range(N_SPOOF_INCIDENTS_PER_ACCOUNT):
                self.schedule(pick_time(), self.inject_spoof, acc)
        for acc in manip["layerer"]:
            for _ in range(N_LAYER_INCIDENTS_PER_ACCOUNT):
                self.schedule(pick_time(), self.inject_layering, acc)
        w = manip["self_washer"][0]
        for _ in range(N_SELF_WASH_INCIDENTS):
            n = int(rng.integers(2, 5))
            self.schedule(pick_time(), self.inject_wash, [w] * n, [w] * n, n,
                          f"self-match x{n}", (500, 4000))
        a, b = manip["linked_washer"]
        for _ in range(N_LINKED_WASH_INCIDENTS):
            n = int(rng.integers(2, 5))
            sellers = [a if i % 2 == 0 else b for i in range(n)]
            buyers = [b if i % 2 == 0 else a for i in range(n)]
            self.schedule(pick_time(), self.inject_wash, sellers, buyers, n,
                          f"linked-account round trip x{n}", (500, 4000))
        c, d = manip["hidden_washer"]
        for _ in range(N_HIDDEN_WASH_INCIDENTS):
            sellers, buyers = [c, d, c, d], [d, c, d, c]
            self.schedule(pick_time(), self.inject_wash, sellers, buyers, 4,
                          "unlinked-pair round trips x2", (300, 1500))

    # ------------------------------------------------------------- output
    def results(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        events = pd.DataFrame(self.events, columns=EVENT_COLUMNS)
        events.insert(0, "event_id", np.arange(len(events)))

        rows = []
        for inc in self.incidents:
            ev = events[events["order_id"].isin(inc["order_ids"])]
            if ev.empty:
                continue
            if inc["pattern"] == "wash_trade":
                # Only label wash incidents where at least one wash fill happened.
                accts = set(inc["related_accounts"].split(";"))
                f = ev[(ev["event_type"] == "fill") & ev["counterparty_account_id"].isin(accts)]
                if f.empty:
                    continue
            rows.append({
                "incident_id": inc["incident_id"],
                "pattern": inc["pattern"],
                "account_id": inc["account_id"],
                "related_accounts": inc["related_accounts"],
                "start_ms": int(ev["timestamp"].min()),
                "end_ms": int(ev["timestamp"].max()),
                "n_orders": len(set(inc["order_ids"])),
                "description": inc["description"],
                "order_ids": ";".join(dict.fromkeys(inc["order_ids"])),
            })
        labels = pd.DataFrame(rows)
        links = pd.DataFrame(self.links, columns=["account_a", "account_b", "relationship"])
        return events, labels, links


def generate(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sim = MarketSimulator(seed)
    sim.setup()
    sim.run()
    return sim.results()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()

    events, labels, links = generate(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.out / "events.csv", index=False, lineterminator="\n")
    labels.to_csv(args.out / "labels.csv", index=False, lineterminator="\n")
    links.to_csv(args.out / "account_links.csv", index=False, lineterminator="\n")
    counts = events["event_type"].value_counts().to_dict()
    print(f"Wrote {len(events):,} events {counts} to {args.out}/events.csv")
    print(f"Wrote {len(labels)} labeled incidents {labels['pattern'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
