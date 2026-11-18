"""
Score the detectors against the ground-truth labels.

* Precision = share of alerts that correspond to a real injected incident
              ("when we raise an alert, how often is it real?")
* Recall    = share of real incidents that got at least one alert
              ("how much of the manipulation did we catch?")
* F1        = harmonic mean of the two.

An alert MATCHES an incident when the pattern is the same, they share an
account, and the alert's time window overlaps the incident's window (padded by
MATCH_TOLERANCE_S).

Also runs a sensitivity sweep on the main spoofing threshold (size multiple).

Usage:
    python evaluate.py                     # uses data/events.csv, data/labels.csv
    python evaluate.py --seeds 1 2 3 4 5   # re-simulate other days, average metrics
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

import config
from detectors import run_all_detectors
from detectors.common import add_bbo, order_summary
from detectors.spoofing import detect_spoofing

PATTERNS = ["wash_trade", "spoofing", "layering"]


def _accounts(s: str) -> set[str]:
    return set(str(s).split(";"))


def match_alerts(alerts: pd.DataFrame, labels: pd.DataFrame,
                 tolerance_s: float = config.MATCH_TOLERANCE_S) -> pd.DataFrame:
    """Return all (alert_id, incident_id) pairs that match, ignoring pattern
    (pattern agreement is checked by the caller)."""
    tol = tolerance_s * 1000
    rows = []
    for a in alerts.itertuples():
        a_acc = _accounts(a.related_accounts) | {a.account_id}
        for inc in labels.itertuples():
            if a.end_ms < inc.start_ms - tol or a.start_ms > inc.end_ms + tol:
                continue
            if a_acc & (_accounts(inc.related_accounts) | {inc.account_id}):
                rows.append({"alert_id": a.alert_id, "alert_pattern": a.pattern,
                             "incident_id": inc.incident_id, "incident_pattern": inc.pattern})
    return pd.DataFrame(rows, columns=["alert_id", "alert_pattern", "incident_id", "incident_pattern"])


def score(alerts: pd.DataFrame, labels: pd.DataFrame, patterns=PATTERNS) -> pd.DataFrame:
    if "alert_id" not in alerts:
        alerts = alerts.assign(alert_id=[f"A{i}" for i in range(len(alerts))])
    m = match_alerts(alerts, labels)
    same = m[m["alert_pattern"] == m["incident_pattern"]]
    cross = m[m["alert_pattern"] != m["incident_pattern"]]
    rows = []
    for p in patterns:
        a_ids = set(alerts.loc[alerts["pattern"] == p, "alert_id"])
        i_ids = set(labels.loc[labels["pattern"] == p, "incident_id"])
        tp_alerts = a_ids & set(same["alert_id"])
        found = i_ids & set(same["incident_id"])
        # Alerts that missed their own pattern but landed on a different kind of
        # real manipulation: wrong label, but not innocent activity.
        other = (a_ids - tp_alerts) & set(cross["alert_id"])
        prec = len(tp_alerts) / len(a_ids) if a_ids else float("nan")
        rec = len(found) / len(i_ids) if i_ids else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if a_ids and i_ids and prec + rec > 0 else 0.0
        rows.append({
            "pattern": p, "incidents": len(i_ids), "alerts": len(a_ids),
            "true_pos_alerts": len(tp_alerts), "false_pos_alerts": len(a_ids) - len(tp_alerts),
            "hit_other_pattern": len(other), "incidents_found": len(found),
            "missed": len(i_ids) - len(found),
            "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
        })
    return pd.DataFrame(rows)


def spoofing_sweep(events: pd.DataFrame, labels: pd.DataFrame,
                   values=config.SPOOFING_SWEEP) -> pd.DataFrame:
    events = add_bbo(events)
    orders = order_summary(events)
    rows = []
    for v in values:
        cfg = replace(config.SPOOFING, size_multiple=v)
        alerts = detect_spoofing(events, cfg, orders=orders)
        r = score(alerts, labels, ["spoofing"]).iloc[0]
        rows.append({"size_multiple": v, "alerts": r["alerts"], "precision": r["precision"],
                     "recall": r["recall"], "f1": r["f1"]})
    return pd.DataFrame(rows)


def plot_sweep(sweep: pd.DataFrame, path: Path) -> None:
    """Static precision/recall-vs-threshold chart for the README."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange, ink, muted = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
    for col, color, label in (("precision", blue, "Precision"), ("recall", orange, "Recall")):
        ax.plot(sweep["size_multiple"], sweep[col], color=color, lw=2, marker="o", ms=5,
                markeredgecolor="white", markeredgewidth=1.5, label=label)
        last = sweep.dropna(subset=[col]).iloc[-1]
        ax.annotate(label, (last["size_multiple"], last[col]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=ink, fontsize=9)
    d = config.SPOOFING.size_multiple
    ax.axvline(d, color=muted, lw=1, ls="--")
    ax.annotate(f"default = {d:g}x", (d, 0.04), xytext=(4, 0), textcoords="offset points",
                color=muted, fontsize=8)
    ax.set_xlabel("Spoofing size threshold (order size / account's median size)", color=muted, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, sweep["size_multiple"].max() + 2.5)
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=muted, labelsize=8, length=0)
    ax.legend(frameon=False, fontsize=8, loc="lower right", ncols=2)
    ax.set_title("Spoofing detector: precision vs. recall as the size threshold moves",
                 loc="left", fontsize=10, color=ink)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def evaluate(events: pd.DataFrame, labels: pd.DataFrame, links: pd.DataFrame | None,
             out_dir: Path | None = None, alerts: pd.DataFrame | None = None, quiet: bool = False):
    if alerts is None:
        alerts = run_all_detectors(events, links)
    metrics = score(alerts, labels)
    sweep = spoofing_sweep(events, labels)
    if out_dir is not None:
        metrics.to_csv(out_dir / "metrics.csv", index=False)
        sweep.to_csv(out_dir / "spoofing_sweep.csv", index=False)
    if not quiet:
        cols = ["pattern", "incidents", "alerts", "true_pos_alerts", "false_pos_alerts",
                "hit_other_pattern", "missed", "precision", "recall", "f1"]
        print("\n=== Detection accuracy vs. ground truth ===")
        print(metrics[cols].to_string(index=False))
        print(f"\n=== Spoofing sensitivity sweep (size_multiple; default = "
              f"{config.SPOOFING.size_multiple}) ===")
        print(sweep.to_string(index=False))
    return metrics, sweep


def robustness(seeds: list[int]) -> pd.DataFrame:
    """Re-simulate several different days and average the metrics, so the
    thresholds aren't just tuned to one lucky dataset."""
    from generate_data import generate

    rows = []
    for seed in seeds:
        events, labels, links = generate(seed)
        m = score(run_all_detectors(events, links), labels)
        rows.append(m.assign(seed=seed))
    allm = pd.concat(rows)
    print(f"\n=== Robustness across seeds {seeds} ===")
    print(allm.pivot(index="seed", columns="pattern", values="f1").to_string())
    summary = allm.groupby("pattern")[["precision", "recall", "f1"]].agg(["mean", "min"]).round(3)
    print("\n" + summary.to_string())
    return allm


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate detectors against ground truth")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--seeds", type=int, nargs="*",
                    help="also re-simulate these seeds and report average metrics")
    args = ap.parse_args()
    if args.seeds:
        robustness(args.seeds)
        return
    events = pd.read_csv(args.data / "events.csv")
    labels = pd.read_csv(args.data / "labels.csv")
    links_path = args.data / "account_links.csv"
    links = pd.read_csv(links_path) if links_path.exists() else None
    alerts_path = args.data / "alerts.csv"
    alerts = pd.read_csv(alerts_path) if alerts_path.exists() else None
    evaluate(events, labels, links, out_dir=args.data, alerts=alerts)


if __name__ == "__main__":
    main()
