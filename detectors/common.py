"""Shared helpers: alert schema, order-book replay, order lifecycle summaries."""
from __future__ import annotations

import heapq

import numpy as np
import pandas as pd

from config import TICK_SIZE

ALERT_COLUMNS = [
    "pattern", "account_id", "related_accounts", "start_ms", "end_ms",
    "score", "evidence", "order_ids",
]


def empty_alerts() -> pd.DataFrame:
    return pd.DataFrame(columns=ALERT_COLUMNS)


def to_ticks(price) -> np.ndarray | int:
    return np.rint(np.asarray(price, dtype=float) / TICK_SIZE).astype(np.int64)


def add_bbo(events: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `events` with `best_bid` / `best_ask` columns: the
    market's best prices immediately BEFORE each event.

    We rebuild the public order book by replaying every new/fill/cancel in
    order. This is what a surveillance system does to answer "where was the
    market when this order was entered?"
    """
    if {"best_bid", "best_ask"}.issubset(events.columns):
        return events
    ev = events.sort_values(["timestamp", "event_id"] if "event_id" in events else "timestamp",
                            kind="stable")
    n = len(ev)
    best_bid = np.full(n, np.nan)
    best_ask = np.full(n, np.nan)

    level = {"buy": {}, "sell": {}}  # side -> tick -> resting qty
    heaps = {"buy": [], "sell": []}  # max-heap via negation for bids
    remaining: dict[str, tuple[str, int, int]] = {}  # order_id -> (side, tick, qty)

    def top(side):
        h, lv = heaps[side], level[side]
        while h:
            p = -h[0] if side == "buy" else h[0]
            if lv.get(p, 0) > 0:
                return p
            heapq.heappop(h)
        return None

    def change(side, tick, delta):
        lv = level[side]
        new = lv.get(tick, 0) + delta
        if new > 0:
            if tick not in lv or lv[tick] <= 0:
                heapq.heappush(heaps[side], -tick if side == "buy" else tick)
            lv[tick] = new
        else:
            lv.pop(tick, None)

    ticks = to_ticks(ev["price"].to_numpy())
    cols = zip(ev["order_id"].to_numpy(), ev["side"].to_numpy(),
               ev["event_type"].to_numpy(), ev["quantity"].to_numpy())
    for i, (oid, side, etype, qty) in enumerate(cols):
        b, a = top("buy"), top("sell")
        best_bid[i] = np.nan if b is None else b * TICK_SIZE
        best_ask[i] = np.nan if a is None else a * TICK_SIZE
        if etype == "new":
            remaining[oid] = (side, ticks[i], int(qty))
            change(side, ticks[i], int(qty))
        elif oid in remaining:
            s, t, q = remaining[oid]
            take = min(q, int(qty))
            change(s, t, -take)
            q -= take
            if q <= 0 or etype == "cancel":
                if etype == "cancel" and q > 0:
                    change(s, t, -q)
                remaining.pop(oid)
            else:
                remaining[oid] = (s, t, q)

    out = ev.copy()
    out["best_bid"] = np.round(best_bid, 8)
    out["best_ask"] = np.round(best_ask, 8)
    return out.loc[events.index]


def order_summary(events: pd.DataFrame) -> pd.DataFrame:
    """One row per order: who, which side, size, when it was entered/ended,
    how much filled, whether and when it was canceled, and the market's best
    bid/ask at entry."""
    ev = add_bbo(events)
    new = ev[ev["event_type"] == "new"][
        ["order_id", "account_id", "side", "price", "quantity", "timestamp", "best_bid", "best_ask"]
    ].rename(columns={"quantity": "qty", "timestamp": "t_new"})
    new = new.drop_duplicates("order_id")

    fills = ev[ev["event_type"] == "fill"].groupby("order_id").agg(
        filled_qty=("quantity", "sum"), t_first_fill=("timestamp", "min"),
        t_last_fill=("timestamp", "max"))
    cancels = ev[ev["event_type"] == "cancel"].groupby("order_id").agg(
        t_cancel=("timestamp", "min"), canceled_qty=("quantity", "sum"))

    o = new.set_index("order_id").join(fills).join(cancels).reset_index()
    o["filled_qty"] = o["filled_qty"].fillna(0).astype(int)
    o["canceled_qty"] = o["canceled_qty"].fillna(0).astype(int)
    o["canceled"] = o["t_cancel"].notna()
    # When the order stopped resting: canceled, fully filled, or never (end of day).
    fully_filled = o["filled_qty"] >= o["qty"]
    o["t_end"] = np.where(o["canceled"], o["t_cancel"],
                          np.where(fully_filled, o["t_last_fill"], np.inf))

    # Distance (in ticks) behind the best price on the order's own side at entry.
    # 0 = at or better than the touch.
    same_best = np.where(o["side"] == "buy", o["best_bid"], o["best_ask"])
    behind = np.where(o["side"] == "buy", same_best - o["price"], o["price"] - same_best)
    o["ticks_from_touch"] = np.clip(np.rint(behind / TICK_SIZE), 0, None)
    return o


def merge_windows(df: pd.DataFrame, key: str, start: str, end: str, gap_ms: float) -> pd.Series:
    """Assign a cluster id so rows with the same `key` whose [start, end]
    windows lie within `gap_ms` of each other share an id."""
    df = df.sort_values([key, start])
    cluster = np.zeros(len(df), dtype=int)
    cid, last_key, last_end = -1, None, -np.inf
    for i, (k, s, e) in enumerate(zip(df[key], df[start], df[end])):
        if k != last_key or s - last_end > gap_ms:
            cid += 1
            last_end = e
        else:
            last_end = max(last_end, e)
        last_key = k
        cluster[i] = cid
    return pd.Series(cluster, index=df.index)


def fmt_time(ms: float) -> str:
    return pd.to_datetime(int(ms), unit="ms").strftime("%H:%M:%S.%f")[:-3]


def plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"
