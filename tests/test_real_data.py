"""Order-level screen on Coinbase-style messages (snapshot, l2update, match)."""
import pandas as pd

from real_data.analyze import large_adds, replay

T0 = pd.Timestamp("2026-11-18T02:00:00Z")


def ts(s: float) -> str:
    return (T0 + pd.Timedelta(seconds=s)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def l2(t, side, price, size):
    return {"type": "l2update", "time": ts(t), "changes": [[side, f"{price:.2f}", f"{size:.8f}"]]}


def match(t, maker_side, price, size):
    return {"type": "match", "time": ts(t), "side": maker_side, "price": f"{price:.2f}",
            "size": f"{size:.8f}", "maker_order_id": "m", "taker_order_id": "t", "trade_id": 1}


def scenario(big_outcome: str) -> list[dict]:
    msgs = [{"type": "snapshot", "bids": [["64000.00", "0.5"]], "asks": [["64000.01", "0.5"]]}]
    # 40 ordinary small adds near the touch set the "normal" add size (0.01 BTC).
    for i in range(40):
        msgs.append(l2(1 + i * 0.1, "buy", 63999.99 - (i % 3) * 0.01, 0.01 * (1 + i // 3)))
    # A 3 BTC (~$192k) offer appears 1 cent behind the best ask...
    msgs.append(l2(10.0, "sell", 64000.02, 3.0))
    if big_outcome == "pulled":
        msgs.append(l2(12.0, "sell", 64000.02, 0.0))  # ...and vanishes with no trades
    elif big_outcome == "filled":
        msgs.append(match(11.5, "sell", 64000.02, 3.0))  # ...or is bought
        msgs.append(l2(11.55, "sell", 64000.02, 0.0))
    return msgs


def test_large_add_pulled_without_trading_is_flagged():
    changes, trades = replay(scenario("pulled"))
    res = large_adds(changes, trades)
    assert len(res) == 1
    r = res.iloc[0]
    assert r["outcome"] == "canceled unfilled" and r["flagged"]
    assert abs(r["lifetime_s"] - 2.0) < 1e-6


def test_large_add_that_trades_is_not_flagged():
    changes, trades = replay(scenario("filled"))
    res = large_adds(changes, trades)
    assert len(res) == 1
    assert res.iloc[0]["outcome"] == "filled"
    assert not res.iloc[0]["flagged"]


def test_large_add_that_stays_is_not_flagged():
    changes, trades = replay(scenario("stays"))
    res = large_adds(changes, trades)
    assert res.iloc[0]["outcome"] == "still resting"
    assert not res["flagged"].any()
