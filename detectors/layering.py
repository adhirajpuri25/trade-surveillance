"""
Layering detector.

Layering is a form of spoofing: instead of one big fake order, the trader
stacks several orders at DIFFERENT price levels on one side of the book to
create a false impression of depth, gets a fill on the OTHER side, and then
pulls the whole stack.

For every fill an account receives, we look at that account's orders on the
opposite side and flag the fill if:

  1. the account had orders resting at >= `min_levels` distinct prices on the
     opposite side, entered within `lookback_s` before the fill;
  2. >= `min_cancel_fraction` of that stacked quantity was canceled within
     `cancel_window_s` after the fill;
  3. the account was one-sided: the stack dwarfs what it had resting on the
     fill side (this separates layering from two-sided market making).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import LAYERING, LayeringConfig
from detectors.common import ALERT_COLUMNS, empty_alerts, fmt_time, merge_windows, order_summary, plural


def layering_hits(events: pd.DataFrame, cfg: LayeringConfig = LAYERING,
                  orders: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (fill, stack) combination that passes all three tests."""
    o = order_summary(events) if orders is None else orders
    fills = events[events["event_type"] == "fill"]
    lookback, window = cfg.lookback_s * 1000, cfg.cancel_window_s * 1000
    hits = []
    for acct, af in fills.groupby("account_id"):
        ao = o[o["account_id"] == acct]
        side = ao["side"].to_numpy()
        t_new = ao["t_new"].to_numpy(dtype=float)
        t_end = ao["t_end"].to_numpy(dtype=float)
        t_cancel = ao["t_cancel"].to_numpy(dtype=float)
        t_last_fill = ao["t_last_fill"].to_numpy(dtype=float)
        price = ao["price"].to_numpy()
        qty = ao["qty"].to_numpy(dtype=float)
        oid = ao["order_id"].to_numpy()
        for t, fside, fqty, fpx, foid in zip(af["timestamp"], af["side"], af["quantity"],
                                             af["price"], af["order_id"]):
            # Resting at the moment of the fill: entered at/before it and either
            # still alive afterwards or filled in this very instant. (An order
            # canceled in the same millisecond was no longer resting.)
            live = (t_new <= t) & ((t_end > t) | (t_last_fill >= t))
            stack = live & (side != fside) & (t_new >= t - lookback)
            if not stack.any():
                continue
            levels = np.unique(price[stack])
            if len(levels) < cfg.min_levels:
                continue
            stack_qty = qty[stack].sum()
            pulled = stack & (t_cancel >= t) & (t_cancel <= t + window)
            cancel_frac = qty[pulled].sum() / stack_qty
            if cancel_frac < cfg.min_cancel_fraction:
                continue
            fill_side_qty = qty[live & (side == fside)].sum()
            one_sided = stack_qty / (stack_qty + fill_side_qty)
            if one_sided < cfg.min_one_sidedness:
                continue
            hits.append({
                "account_id": acct, "t_fill": t, "fill_side": fside, "fill_qty": fqty,
                "fill_price": fpx, "fill_order_id": foid, "n_levels": len(levels),
                "stack_qty": stack_qty, "cancel_frac": cancel_frac, "one_sided": one_sided,
                "t_stack_start": t_new[stack].min(),
                "t_last_cancel": np.nanmax(t_cancel[pulled]) if pulled.any() else t,
                "stack_order_ids": ";".join(oid[stack]),
            })
    return pd.DataFrame(hits)


def detect_layering(events: pd.DataFrame, cfg: LayeringConfig = LAYERING,
                    orders: pd.DataFrame | None = None) -> pd.DataFrame:
    h = layering_hits(events, cfg, orders)
    if h.empty:
        return empty_alerts()
    h["cluster"] = merge_windows(h, "account_id", "t_stack_start", "t_last_cancel",
                                 cfg.alert_merge_gap_s * 1000)
    alerts = []
    for _, g in h.groupby("cluster"):
        best = g.loc[g["n_levels"].idxmax()]
        acct = best["account_id"]
        secs_to_cancel = (g["t_last_cancel"].max() - g["t_fill"].min()) / 1000
        # Score 0-100: more levels, more of the stack pulled, faster, more one-sided.
        levels_c = min(best["n_levels"] / (2 * cfg.min_levels), 1.0)
        speed_c = max(0.0, 1 - secs_to_cancel / cfg.cancel_window_s)
        score = round(100 * (0.3 * levels_c + 0.25 * g["cancel_frac"].max()
                             + 0.2 * speed_c + 0.25 * g["one_sided"].max()))
        stack_side = "sell" if best["fill_side"] == "buy" else "buy"
        order_ids = list(dict.fromkeys(";".join(g["stack_order_ids"]).split(";")))
        alerts.append({
            "pattern": "layering",
            "account_id": acct,
            "related_accounts": acct,
            "start_ms": int(g["t_stack_start"].min()),
            "end_ms": int(max(g["t_last_cancel"].max(), g["t_fill"].max())),
            "score": score,
            "evidence": (f"{best['n_levels']} {stack_side} orders at distinct price levels "
                         f"({plural(int(best['stack_qty']), 'lot')}) stacked from "
                         f"{fmt_time(g['t_stack_start'].min())}; {best['fill_side']} "
                         f"{plural(int(g['fill_qty'].sum()), 'lot')} filled at {g['fill_price'].mean():.2f}, then "
                         f"{best['cancel_frac']:.0%} of the stack canceled within "
                         f"{secs_to_cancel:.1f}s; one-sidedness {best['one_sided']:.0%}."),
            "order_ids": ";".join(order_ids + list(g["fill_order_id"])),
        })
    return pd.DataFrame(alerts, columns=ALERT_COLUMNS).sort_values("start_ms").reset_index(drop=True)
