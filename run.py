"""
End-to-end pipeline: generate synthetic data -> run detectors -> evaluate.

    python run.py              # default seed (42)
    python run.py --seed 7     # a different simulated day

Writes to data/: events.csv, labels.csv, account_links.csv, alerts.csv,
metrics.csv, spoofing_sweep.csv; and docs/spoofing_sweep.png.
Then launch the dashboard with:  streamlit run app.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from detectors import run_all_detectors
from evaluate import evaluate, plot_sweep
from generate_data import SEED, generate


def main() -> None:
    ap = argparse.ArgumentParser(description="generate -> detect -> evaluate")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t = time.perf_counter()
    print(f"[1/3] Simulating one trading day (seed={args.seed})...")
    events, labels, links = generate(args.seed)
    events.to_csv(args.out / "events.csv", index=False)
    labels.to_csv(args.out / "labels.csv", index=False)
    links.to_csv(args.out / "account_links.csv", index=False)
    print(f"      {len(events):,} order events, {events['account_id'].nunique()} accounts, "
          f"{len(labels)} injected incidents {labels['pattern'].value_counts().to_dict()}")

    print("[2/3] Running detectors...")
    alerts = run_all_detectors(events, links)
    alerts.to_csv(args.out / "alerts.csv", index=False)
    print(f"      {len(alerts)} alerts {alerts['pattern'].value_counts().to_dict()}")

    print("[3/3] Evaluating against ground truth...")
    _, sweep = evaluate(events, labels, links, out_dir=args.out, alerts=alerts)
    plot_sweep(sweep, Path("docs") / "spoofing_sweep.png")
    print(f"\nDone in {time.perf_counter() - t:.1f}s. Dashboard: streamlit run app.py")


if __name__ == "__main__":
    main()
