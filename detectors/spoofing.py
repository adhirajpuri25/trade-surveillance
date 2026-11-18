"""
Spoofing detector.

Spoofing = bidding or offering with the intent to cancel before execution
(CEA Section 4c(a)(5)(C)). The classic footprint:

  1. a LARGE order (relative to how this account normally trades),
  2. placed NEAR THE TOUCH where other traders will see it,
  3. CANCELED within seconds, essentially UNFILLED,
  4. while the SAME account gets a fill on the OPPOSITE side -- the trade the
     fake order was meant to help.

An order must pass all four tests to be flagged. Flagged orders from the same
account close together in time become one alert.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import SPOOFING, SpoofingConfig
from detectors.common import ALERT_COLUMNS, empty_alerts, fmt_time, merge_windows, order_summary, plural


def spoof_candidates(events: pd.DataFrame, cfg: SpoofingConfig = SPOOFING,
                     orders: pd.DataFrame | None = None) -> pd.DataFrame:
    """Orders passing tests 1-3, with the size ratio used for test 1."""
    o = order_summary(events) if orders is None else orders
    market_median = o["qty"].median()
    stats = o.groupby("account_id")["qty"].agg(["median", "count"])
    baseline = np.where(stats["count"] >= cfg.min_orders_for_baseline, stats["median"], market_median)
    o = o.join(pd.Series(baseline, index=stats.index, name="baseline"), on="account_id")
    o["size_ratio"] = o["qty"] / o["baseline"]
    o["lifetime_s"] = (o["t_cancel"] - o["t_new"]) / 1000

    mask = (
        (o["size_ratio"] >= cfg.size_multiple)
        & (o["qty"] >= cfg.min_market_multiple * market_median)
        & o["canceled"]
        & (o["lifetime_s"] <= cfg.max_order_lifetime_s)
        & (o["filled_qty"] <= cfg.max_filled_fraction * o["qty"])
        & (o["ticks_from_touch"] <= cfg.max_ticks_from_touch)
    )
    return o[mask].copy()


def detect_spoofing(events: pd.DataFrame, cfg: SpoofingConfig = SPOOFING,
                    orders: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return one alert per spoofing episode.

    `orders` (the output of order_summary) can be passed in to avoid
    recomputing it, e.g. during a threshold sweep.
    """
    cand = spoof_candidates(events, cfg, orders)
    if cand.empty:
        return empty_alerts()

    # Test 4: same account, opposite side, filled inside the order's window.
    fills = events[events["event_type"] == "fill"][
        ["account_id", "side", "timestamp", "quantity", "price", "order_id"]
    ].rename(columns={"side": "fill_side", "timestamp": "t_fill", "quantity": "fill_qty",
                      "price": "fill_price", "order_id": "fill_order_id"})
    m = cand.merge(fills, on="account_id")
    m = m[(m["fill_side"] != m["side"])
          & (m["t_fill"] >= m["t_new"] - cfg.fill_window_before_s * 1000)
          & (m["t_fill"] <= m["t_cancel"] + cfg.fill_window_after_s * 1000)]
    if m.empty:
        return empty_alerts()

    flagged = cand[cand["order_id"].isin(m["order_id"])].copy()
    flagged["cluster"] = merge_windows(flagged, "account_id", "t_new", "t_cancel",
                                       cfg.alert_merge_gap_s * 1000)
    m = m.merge(flagged[["order_id", "cluster"]], on="order_id")

    alerts = []
    for cid, g in flagged.groupby("cluster"):
        gf = m[m["cluster"] == cid].drop_duplicates("fill_order_id")
        acct = g["account_id"].iloc[0]
        spoof_side = g["side"].iloc[0]
        spoof_qty = int(g["qty"].sum())
        fill_qty = int(gf["fill_qty"].sum())
        ratio = float(g["size_ratio"].max())
        life = float(g["lifetime_s"].max())
        # Score 0-100, each part 0-1:
        #   size: 0.5 at the threshold, 1.0 at twice the threshold
        #   speed: 1.0 for an instant cancel, 0 at the lifetime limit
        #   asymmetry: how much bigger the fake side was than the real fill
        size_c = min(ratio / (2 * cfg.size_multiple), 1.0)
        speed_c = max(0.0, 1 - life / cfg.max_order_lifetime_s)
        asym_c = max(0.0, 1 - fill_qty / max(spoof_qty, 1))
        score = round(100 * (0.4 * size_c + 0.3 * speed_c + 0.3 * asym_c))
        fill_side = gf["fill_side"].iloc[0]
        alerts.append({
            "pattern": "spoofing",
            "account_id": acct,
            "related_accounts": acct,
            "start_ms": int(min(g["t_new"].min(), gf["t_fill"].min())),
            "end_ms": int(max(g["t_cancel"].max(), gf["t_fill"].max())),
            "score": score,
            "evidence": (f"{plural(len(g), f'large {spoof_side} order')} totaling "
                         f"{plural(spoof_qty, 'lot')} ({ratio:.1f}x account's median size of "
                         f"{g['baseline'].iloc[0]:.0f}), <= {plural(int(g['ticks_from_touch'].max()), 'tick')} "
                         f"from the touch, entered {fmt_time(g['t_new'].min())} and canceled after "
                         f"{life:.1f}s with {int(g['filled_qty'].sum())} filled; meanwhile "
                         f"{'bought' if fill_side == 'buy' else 'sold'} {plural(fill_qty, 'lot')} "
                         f"at {gf['fill_price'].mean():.2f}."),
            "order_ids": ";".join(list(g["order_id"]) + list(gf["fill_order_id"])),
        })
    return pd.DataFrame(alerts, columns=ALERT_COLUMNS).sort_values("start_ms").reset_index(drop=True)
