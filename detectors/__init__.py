"""Market-abuse detectors. Each is a pure function: events DataFrame in, alerts DataFrame out."""
from __future__ import annotations

import pandas as pd

from detectors.common import ALERT_COLUMNS, add_bbo, order_summary
from detectors.layering import detect_layering
from detectors.spoofing import detect_spoofing
from detectors.wash_trades import detect_wash_trades

__all__ = ["detect_wash_trades", "detect_spoofing", "detect_layering", "run_all_detectors"]


def run_all_detectors(events: pd.DataFrame, account_links: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run every detector and return one combined alerts table with alert IDs."""
    events = add_bbo(events)  # replay the book once, shared by all detectors
    orders = order_summary(events)
    parts = [
        detect_wash_trades(events, account_links),
        detect_spoofing(events, orders=orders),
        detect_layering(events, orders=orders),
    ]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=["alert_id"] + ALERT_COLUMNS)
    alerts = pd.concat(parts, ignore_index=True)
    alerts = alerts.sort_values(["start_ms", "pattern"]).reset_index(drop=True)
    alerts.insert(0, "alert_id", [f"AL{i + 1:04d}" for i in range(len(alerts))])
    return alerts
