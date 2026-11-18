"""
Wash-trade detector.

A wash trade is a trade where there is no real change in who owns the
position: the same account (or accounts with the same beneficial owner) is on
both sides. Three rules:

1. SELF-MATCH:   buyer account == seller account.
2. LINKED:       buyer and seller are different accounts that exchange
                 reference data says share a beneficial owner.
3. ROUND TRIP:   two accounts with NO known link repeatedly trade the same
                 size back and forth at ~the same price within seconds
                 (A sells to B, then B sells it back to A). This catches links
                 the exchange doesn't know about yet; it is the weakest rule,
                 so it gets a lower score and needs repetition.

Trades are grouped into episodes (clusters in time per account/pair) and an
episode is alerted only if it contains at least `min_trades` such trades.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import TICK_SIZE, WASH, WashTradeConfig
from detectors.common import ALERT_COLUMNS, empty_alerts, fmt_time, merge_windows, plural


def trades_from_events(events: pd.DataFrame) -> pd.DataFrame:
    """One row per trade. Every trade has exactly one buy-side fill row, whose
    counterparty is the seller."""
    f = events[(events["event_type"] == "fill") & (events["side"] == "buy")]
    t = pd.DataFrame({
        "timestamp": f["timestamp"].to_numpy(),
        "buyer": f["account_id"].to_numpy(),
        "seller": f["counterparty_account_id"].to_numpy(),
        "price": f["price"].to_numpy(),
        "qty": f["quantity"].to_numpy(),
        "buy_order_id": f["order_id"].to_numpy(),
    })
    if "trade_id" in f:
        t["trade_id"] = f["trade_id"].to_numpy()
    return t


def _link_set(account_links: pd.DataFrame | None) -> set[frozenset]:
    if account_links is None or account_links.empty:
        return set()
    return {frozenset((a, b)) for a, b in zip(account_links["account_a"], account_links["account_b"])}


def _cluster_alerts(t: pd.DataFrame, kind: str, cfg: WashTradeConfig) -> list[dict]:
    if t.empty:
        return []
    t = t.copy()
    t["pair"] = [";".join(sorted({b, s})) for b, s in zip(t["buyer"], t["seller"])]
    t["cluster"] = merge_windows(t, "pair", "timestamp", "timestamp", cfg.cluster_gap_s * 1000)
    alerts = []
    for _, g in t.groupby("cluster"):
        n = len(g)
        if n < cfg.min_trades:
            continue
        accounts = g["pair"].iloc[0].split(";")
        px_range = (g["price"].max() - g["price"].min()) / TICK_SIZE
        base = 70 if kind == "self-match" else 60
        score = min(100, base + 10 * (n - 1))
        who = accounts[0] if kind == "self-match" else " <-> ".join(accounts)
        alerts.append({
            "pattern": "wash_trade",
            "account_id": accounts[0],
            "related_accounts": ";".join(accounts),
            "start_ms": int(g["timestamp"].min()),
            "end_ms": int(g["timestamp"].max()),
            "score": score,
            "evidence": (f"{kind}: {n} trades between {who} from {fmt_time(g['timestamp'].min())} "
                         f"to {fmt_time(g['timestamp'].max())}, {int(g['qty'].sum())} lots total, "
                         f"price range {px_range:.0f} ticks; no change in beneficial ownership."),
            "order_ids": ";".join(g["buy_order_id"]),
        })
    return alerts


def _round_trip_alerts(t: pd.DataFrame, cfg: WashTradeConfig) -> list[dict]:
    """Unlinked pairs trading the same size back and forth at ~the same price."""
    t = t.sort_values("timestamp")
    t = t.assign(pair=[tuple(sorted((b, s))) for b, s in zip(t["buyer"], t["seller"])])
    window = cfg.round_trip_window_s * 1000
    max_px = cfg.round_trip_max_price_diff_ticks * TICK_SIZE + 1e-9
    legs = []  # (pair, t_first, t_second, qty, order ids)
    counts = t["pair"].value_counts()
    for pair, g in t[t["pair"].isin(counts[counts >= 2].index)].groupby("pair"):
        rows = g.to_dict("records")
        used = set()
        for i, r in enumerate(rows):
            if i in used:
                continue
            for j in range(i + 1, len(rows)):
                r2 = rows[j]
                if r2["timestamp"] - r["timestamp"] > window:
                    break
                if (j not in used and r2["buyer"] == r["seller"] and r2["seller"] == r["buyer"]
                        and r2["qty"] == r["qty"] and abs(r2["price"] - r["price"]) <= max_px):
                    used.update((i, j))
                    legs.append((";".join(pair), r["timestamp"], r2["timestamp"], r["qty"],
                                 f"{r['buy_order_id']};{r2['buy_order_id']}"))
                    break
    if not legs:
        return []
    rt = pd.DataFrame(legs, columns=["pair", "t1", "t2", "qty", "order_ids"])
    rt["cluster"] = merge_windows(rt, "pair", "t1", "t2", cfg.cluster_gap_s * 1000)
    alerts = []
    for _, g in rt.groupby("cluster"):
        n = len(g)
        if n < cfg.min_round_trips:
            continue
        a, b = g["pair"].iloc[0].split(";")
        alerts.append({
            "pattern": "wash_trade",
            "account_id": a,
            "related_accounts": f"{a};{b}",
            "start_ms": int(g["t1"].min()),
            "end_ms": int(g["t2"].max()),
            "score": min(100, 40 + 10 * (n - 1)),
            "evidence": (f"round trip (accounts NOT known to be linked): {a} and {b} traded "
                         f"{n} matched round trips ({int(g['qty'].sum())} lots each way) between "
                         f"{fmt_time(g['t1'].min())} and {fmt_time(g['t2'].max())}; net position "
                         f"change zero. Possible undisclosed common ownership."),
            "order_ids": ";".join(g["order_ids"]),
        })
    return alerts


def detect_wash_trades(events: pd.DataFrame, account_links: pd.DataFrame | None = None,
                       cfg: WashTradeConfig = WASH) -> pd.DataFrame:
    """Flag self-matches, linked-account trades and unexplained round trips.

    Parameters
    ----------
    events : order events (needs fill rows with counterparty_account_id)
    account_links : known beneficial-ownership links (columns account_a, account_b)
    """
    t = trades_from_events(events)
    if t.empty:
        return empty_alerts()
    links = _link_set(account_links)
    is_self = t["buyer"] == t["seller"]
    is_linked = np.array([frozenset((b, s)) in links for b, s in zip(t["buyer"], t["seller"])]) & ~is_self

    alerts = (_cluster_alerts(t[is_self], "self-match", cfg)
              + _cluster_alerts(t[is_linked], "linked accounts", cfg)
              + _round_trip_alerts(t[~is_self & ~is_linked], cfg))
    if not alerts:
        return empty_alerts()
    return pd.DataFrame(alerts, columns=ALERT_COLUMNS).sort_values("start_ms").reset_index(drop=True)
