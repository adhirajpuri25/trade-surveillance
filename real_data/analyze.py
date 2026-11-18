"""
Order-level spoofing screen on real Coinbase BTC-USD data.

Public data has NO account IDs, and the public book feed (level2_batch) has no
order IDs either: it only reports the total size resting at each price level.
So we infer "orders" from the book itself:

  * a LARGE ADD = the size at one price level, near the touch, jumps by an
    unusually large amount in a single 50 ms update;
  * it is FLAGGED if that extra size disappears again within a few seconds
    while (almost) no trades print at that price, i.e. it was canceled, not filled.

That is the same "large order, near the touch, canceled within seconds without
filling" footprint as the synthetic spoofing detector, minus the test we can't
do without accounts: did the SAME trader fill on the other side?

Usage:
    python -m real_data.analyze                      # newest file in real_data/raw/
    python -m real_data.analyze --raw path/to/file.jsonl.gz
"""
from __future__ import annotations

import argparse
import gzip
import heapq
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from config import REAL_DATA, RealDataConfig

HERE = Path(__file__).parent
RAW_DIR, OUT_DIR = HERE / "raw", HERE / "output"


# --------------------------------------------------------------------- loading
def load_messages(path: Path) -> list[dict]:
    """Read a gzipped JSON-lines capture (tolerates a truncated final block)."""
    msgs = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    msgs.append(json.loads(line))
                except json.JSONDecodeError:
                    break
    except (EOFError, zlib.error, OSError):
        pass
    return msgs


def _to_ms(times) -> np.ndarray:
    # Pin the unit: pandas may parse ISO strings at microsecond resolution.
    ts = pd.to_datetime(pd.Series(times), utc=True, format="ISO8601").dt.as_unit("ms")
    return ts.astype("int64").to_numpy().astype(float)


def replay(msgs: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild the price-level book and return

    changes: one row per price-level update (side, price, old/new size, delta,
             best bid/ask just before the update, `epoch` = book resets)
    trades:  one row per match (price, size, maker side, order IDs)
    """
    book = {"buy": {}, "sell": {}}
    heaps = {"buy": [], "sell": []}

    def push(side, px):
        heapq.heappush(heaps[side], -px if side == "buy" else px)

    def best(side):
        h = heaps[side]
        while h:
            px = -h[0] if side == "buy" else h[0]
            if book[side].get(px, 0) > 0:
                return px
            heapq.heappop(h)
        return np.nan

    rows, trades, epoch = [], [], 0
    for m in msgs:
        t = m.get("type")
        if t == "snapshot":
            epoch += 1
            book = {"buy": {}, "sell": {}}
            heaps = {"buy": [], "sell": []}
            for side, key in (("buy", "bids"), ("sell", "asks")):
                for px, sz in m[key]:
                    px, sz = float(px), float(sz)
                    if sz > 0:
                        book[side][px] = sz
                        push(side, px)
        elif t == "l2update" and epoch:
            bb, ba = best("buy"), best("sell")
            for side, px, sz in m["changes"]:
                px, sz = float(px), float(sz)
                old = book[side].get(px, 0.0)
                if sz > 0:
                    if old == 0:
                        push(side, px)
                    book[side][px] = sz
                else:
                    book[side].pop(px, None)
                rows.append((m["time"], epoch, side, px, old, sz, sz - old, bb, ba))
        elif t == "match":
            trades.append((m["time"], float(m["price"]), float(m["size"]), m["side"],
                           m.get("maker_order_id"), m.get("taker_order_id"), m.get("trade_id")))
        elif t == "_reconnect":
            epoch += 1  # ignore updates until the next snapshot
            book = {"buy": {}, "sell": {}}
            heaps = {"buy": [], "sell": []}

    changes = pd.DataFrame(rows, columns=["time", "epoch", "side", "price", "old_size",
                                          "new_size", "delta", "best_bid", "best_ask"])
    changes["t"] = _to_ms(changes["time"])
    tr = pd.DataFrame(trades, columns=["time", "price", "size", "maker_side", "maker_order_id",
                                       "taker_order_id", "trade_id"])
    tr["t"] = _to_ms(tr["time"]) if len(tr) else []
    return changes, tr


# ------------------------------------------------------------------- detection
def _near_touch_adds(changes: pd.DataFrame, cfg: RealDataConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    c = changes.copy()
    c["mid"] = (c["best_bid"] + c["best_ask"]) / 2
    behind = np.where(c["side"] == "buy", c["best_bid"] - c["price"], c["price"] - c["best_ask"])
    c["dist_bps"] = np.clip(behind, 0, None) / c["mid"] * 1e4
    adds = c[(c["delta"] > 0) & (c["dist_bps"] <= cfg.max_distance_bps)]
    return c, adds


def large_adds(changes: pd.DataFrame, trades: pd.DataFrame,
               cfg: RealDataConfig = REAL_DATA) -> pd.DataFrame:
    """Find large size jumps near the touch and follow what happened to them."""
    c, adds = _near_touch_adds(changes, cfg)
    median_add = adds["delta"].median()
    big = adds[(adds["delta"] >= cfg.size_multiple * median_add)
               & (adds["delta"] * adds["price"] >= cfg.min_notional_usd)]
    res = _follow(big, c, trades, cfg, median_add)
    res.attrs["median_add_btc"] = median_add
    res.attrs["n_adds_near_touch"] = len(adds)
    return res


def baseline_rate(changes: pd.DataFrame, trades: pd.DataFrame, cfg: RealDataConfig = REAL_DATA,
                  n: int = 3000, seed: int = 0) -> float:
    """Share of ORDINARY near-touch adds (random sample, any size) that would
    meet the same 'gone within max_lifetime_s, unfilled' test. If this is high,
    fast cancellation alone says little -- which is the point."""
    c, adds = _near_touch_adds(changes, cfg)
    sample = adds.sample(min(n, len(adds)), random_state=seed)
    res = _follow(sample, c, trades, cfg, adds["delta"].median())
    return float(res["flagged"].mean()) if len(res) else float("nan")


def _follow(cands: pd.DataFrame, c: pd.DataFrame, trades: pd.DataFrame,
            cfg: RealDataConfig, median_add: float) -> pd.DataFrame:
    # Index level histories and trades per (side, price) for the forward look.
    keys = set(zip(cands["epoch"], cands["side"], cands["price"]))
    sub = c[[k in keys for k in zip(c["epoch"], c["side"], c["price"])]]
    levels = {k: g for k, g in sub.groupby(["epoch", "side", "price"])}
    trades_at = {k: g for k, g in trades.groupby(["maker_side", "price"])} if len(trades) else {}
    horizon = cfg.follow_s * 1000

    out = []
    for r in cands.itertuples():
        hist = levels[(r.epoch, r.side, r.price)]
        later = hist[(hist["t"] > r.t) & (hist["t"] <= r.t + horizon)]
        gone = later[later["new_size"] <= r.new_size - cfg.removal_fraction * r.delta]
        t_end = gone["t"].iloc[0] if len(gone) else np.nan
        tr = trades_at.get((r.side, r.price))
        window_end = t_end if len(gone) else r.t + horizon
        filled = 0.0 if tr is None else tr.loc[(tr["t"] >= r.t) & (tr["t"] <= window_end), "size"].sum()
        fill_frac = min(filled / r.delta, 1.0)
        if not len(gone):
            outcome = "still resting"
        elif fill_frac <= cfg.max_filled_fraction:
            outcome = "canceled unfilled"
        elif fill_frac >= 0.9:
            outcome = "filled"
        else:
            outcome = "partly filled, rest canceled"
        out.append({
            "time": r.time, "t": r.t, "side": r.side, "price": r.price, "size_btc": r.delta,
            "notional_usd": r.delta * r.price, "size_x_median": r.delta / median_add,
            "dist_bps": r.dist_bps, "dist_usd": r.dist_bps * r.mid / 1e4,
            "best_bid": r.best_bid, "best_ask": r.best_ask, "level_size_after": r.new_size,
            "outcome": outcome, "lifetime_s": (t_end - r.t) / 1000 if len(gone) else np.nan,
            "filled_btc": filled, "t_end": t_end,
        })
    res = pd.DataFrame(out)
    if res.empty:
        return res.assign(flagged=pd.Series(dtype=bool))
    res["flagged"] = (res["outcome"] == "canceled unfilled") & (res["lifetime_s"] <= cfg.max_lifetime_s)
    return res


def add_context(flagged: pd.DataFrame, changes: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """What the rest of the market did while each flagged order was visible.
    (Descriptive only: without account IDs we can't say who benefited.)"""
    mids = changes.assign(mid=(changes["best_bid"] + changes["best_ask"]) / 2)[["t", "mid"]]
    rows = []
    for r in flagged.itertuples():
        before = mids[mids["t"] <= r.t]["mid"]
        after = mids[mids["t"] <= r.t_end]["mid"]
        w = trades[(trades["t"] >= r.t) & (trades["t"] <= r.t_end)]
        # Coinbase reports the MAKER's side; the taker did the opposite.
        rows.append({
            "mid_move_bps": (after.iloc[-1] / before.iloc[-1] - 1) * 1e4 if len(before) and len(after) else np.nan,
            "taker_buy_btc": w.loc[w["maker_side"] == "sell", "size"].sum(),
            "taker_sell_btc": w.loc[w["maker_side"] == "buy", "size"].sum(),
        })
    return pd.concat([flagged.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


# --------------------------------------------------------------------- output
def plot_example(ev: pd.Series, changes: pd.DataFrame, trades: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange, gray, ink, muted = "#2a78d6", "#eb6834", "#a3a29b", "#0b0b0b", "#52514e"
    t0, t1 = ev["t"] - 4000, ev["t_end"] + 4000
    w = changes[(changes["t"] >= t0) & (changes["t"] <= t1)]
    ts = lambda ms: pd.to_datetime(ms, unit="ms")  # noqa: E731
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=150)
    ax.step(ts(w["t"]), w["best_bid"], where="post", color=gray, lw=1.3, label="Best bid")
    ax.step(ts(w["t"]), w["best_ask"], where="post", color=gray, lw=1.3, ls=":", label="Best ask")
    color = blue if ev["side"] == "buy" else orange
    ax.plot([ts(ev["t"]), ts(ev["t_end"])], [ev["price"]] * 2, color=color, lw=5, solid_capstyle="round",
            label=f"Flagged {ev['side']} size ({ev['size_btc']:.2f} BTC)")
    ax.scatter([ts(ev["t_end"])], [ev["price"]], marker="x", s=60, color=color, zorder=5)
    tw = trades[(trades["t"] >= t0) & (trades["t"] <= t1)]
    ax.scatter(ts(tw["t"]), tw["price"], s=8 + 40 * np.sqrt(tw["size"]), color=muted, alpha=0.5,
               edgecolor="white", lw=0.5, label="Trades (dot size = BTC)", zorder=4)
    ax.annotate(f"{ev['size_btc']:.2f} BTC appears", (ts(ev["t"]), ev["price"]), xytext=(0, 10),
                textcoords="offset points", fontsize=8, color=ink, ha="left")
    ax.annotate(f"gone {ev['lifetime_s']:.1f}s later, {ev['filled_btc']:.3f} BTC traded there",
                (ts(ev["t_end"]), ev["price"]), xytext=(6, -14), textcoords="offset points",
                fontsize=8, color=ink)
    ax.set_title(f"Coinbase BTC-USD: large {ev['side']} order pulled without trading "
                 f"({ev['time'][11:19]} UTC)", loc="left", fontsize=10, color=ink)
    ax.set_ylabel("Price (USD)", color=muted, fontsize=9)
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=muted, labelsize=8, length=0)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M:%S"))
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, -0.1), ncols=4)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def summarize(res: pd.DataFrame, changes: pd.DataFrame, trades: pd.DataFrame, cfg: RealDataConfig) -> dict:
    span_min = (changes["t"].max() - changes["t"].min()) / 60000
    by_outcome = res["outcome"].value_counts().to_dict() if len(res) else {}
    return {
        "window_utc": f"{changes['time'].min()[:19]} to {changes['time'].max()[:19]}",
        "minutes": round(span_min, 1),
        "level_updates": len(changes),
        "trades": len(trades),
        "traded_btc": round(trades["size"].sum(), 2),
        "adds_near_touch": res.attrs.get("n_adds_near_touch", 0),
        "median_add_btc": res.attrs.get("median_add_btc", np.nan),
        "large_adds": len(res),
        "outcomes": by_outcome,
        "flagged": int(res["flagged"].sum()) if len(res) else 0,
        "thresholds": cfg.__dict__,
    }


def newest_raw() -> Path:
    files = sorted(RAW_DIR.glob("*.jsonl.gz"))
    if not files:
        raise SystemExit("No capture found. Run: python -m real_data.capture --minutes 20")
    return files[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=None)
    args = ap.parse_args()
    raw = args.raw or newest_raw()
    print(f"Replaying {raw} ...")
    changes, trades = replay(load_messages(raw))
    res = large_adds(changes, trades)
    summary = summarize(res, changes, trades, REAL_DATA)
    summary["share_of_large_adds_flagged"] = round(float(res["flagged"].mean()), 3) if len(res) else None
    summary["share_of_ordinary_adds_meeting_same_test"] = round(baseline_rate(changes, trades), 3)
    OUT_DIR.mkdir(exist_ok=True)
    flagged = res[res["flagged"]].sort_values("notional_usd", ascending=False) if len(res) else res
    if len(flagged):
        flagged = add_context(flagged, changes, trades)
    digits = {"price": 2, "best_bid": 2, "best_ask": 2, "size_btc": 8, "level_size_after": 8,
              "filled_btc": 8, "taker_buy_btc": 8, "taker_sell_btc": 8, "notional_usd": 0,
              "size_x_median": 1, "dist_bps": 3, "dist_usd": 2, "lifetime_s": 3, "mid_move_bps": 3}
    res.drop(columns=["t", "t_end"], errors="ignore").round(digits).to_csv(
        OUT_DIR / "large_adds.csv", index=False, lineterminator="\n")
    flagged.drop(columns=["t", "t_end", "flagged"], errors="ignore").round(digits).to_csv(
        OUT_DIR / "flagged_orders.csv", index=False, lineterminator="\n")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    if len(flagged):
        # Chart the largest flagged order that sat behind the touch and lived
        # long enough (>= 1 s) to be visible on a chart.
        visible = flagged[(flagged["lifetime_s"] >= 1) & (flagged["dist_usd"] > 0)]
        ex = (visible if len(visible) else flagged).iloc[0]
        plot_example(ex, changes, trades, HERE.parent / "docs" / "real_data_example.png")

    print(json.dumps(summary, indent=2, default=float))
    if len(flagged):
        cols = ["time", "side", "price", "size_btc", "notional_usd", "size_x_median", "dist_usd",
                "lifetime_s", "filled_btc", "mid_move_bps", "taker_buy_btc", "taker_sell_btc"]
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(flagged[cols].head(10).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
