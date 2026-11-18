"""
Record a sample of real BTC-USD market data from Coinbase's public WebSocket feed.

Coinbase's order-level channels ("full" / "level3", which carry an order ID for
every add and cancel) now require an authenticated API key. The best public
substitute is:

    level2_batch  every change to the total size resting at each price level,
                  batched every 50 ms (no order IDs)
    matches       every trade, with maker/taker order IDs, price, size, side

Messages are written untouched (plus a local receive timestamp) to a gzipped
JSON-lines file, so the analysis can be re-run later without reconnecting.

Usage:
    python -m real_data.capture --minutes 20
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import websockets

URL = "wss://ws-feed.exchange.coinbase.com"
CHANNELS = ["level2_batch", "matches"]
RAW_DIR = Path(__file__).parent / "raw"


async def record(product: str, minutes: float, out: Path) -> None:
    deadline = time.time() + minutes * 60
    counts: dict[str, int] = {}
    reconnects = 0
    with gzip.open(out, "wt", encoding="utf-8") as f:
        while time.time() < deadline and reconnects <= 5:
            try:
                async with websockets.connect(URL, max_size=2**25, ping_interval=20) as ws:
                    await ws.send(json.dumps({"type": "subscribe", "product_ids": [product],
                                              "channels": CHANNELS}))
                    while time.time() < deadline:
                        raw = await asyncio.wait_for(ws.recv(), timeout=30)
                        msg = json.loads(raw)
                        if msg.get("type") == "error":
                            raise RuntimeError(f"Coinbase error: {msg}")
                        msg["recv_ts"] = time.time()
                        f.write(json.dumps(msg, separators=(",", ":")) + "\n")
                        t = msg.get("type", "?")
                        counts[t] = counts.get(t, 0) + 1
                        if sum(counts.values()) % 5000 == 0:
                            left = (deadline - time.time()) / 60
                            print(f"  {counts}  ({left:.1f} min left)", flush=True)
            except (websockets.ConnectionClosed, asyncio.TimeoutError, OSError) as e:
                # A reconnect gets a fresh book snapshot; the gap is marked so
                # the analysis can reset its state.
                reconnects += 1
                f.write(json.dumps({"type": "_reconnect", "recv_ts": time.time(),
                                    "reason": repr(e)}) + "\n")
                print(f"  connection lost ({e!r}); reconnecting ({reconnects}/5)", flush=True)
                await asyncio.sleep(2)
    print(f"Saved {sum(counts.values()):,} messages {counts} to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--product", default="BTC-USD")
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    out = args.out or RAW_DIR / f"{args.product.lower().replace('-', '_')}_{stamp}.jsonl.gz"
    print(f"Recording {args.product} {CHANNELS} for {args.minutes} min -> {out}")
    asyncio.run(record(args.product, args.minutes, out))


if __name__ == "__main__":
    main()
