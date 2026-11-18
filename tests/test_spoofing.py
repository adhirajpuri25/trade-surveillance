from detectors.spoofing import detect_spoofing
from tests.helpers import Scenario


def base() -> Scenario:
    s = Scenario()
    s.quote()  # best bid 74.50 / best ask 74.51
    s.background("SPOOFER", n=30, qty=2)  # normal size: 2 lots
    s.background("OTHER", n=30, qty=2, side="sell", price=74.70)
    return s


def test_flags_large_quick_cancel_with_opposite_fill():
    s = base()
    genuine = s.new(100.0, "SPOOFER", "buy", 74.50, 2)
    spoof = s.new(100.2, "SPOOFER", "sell", 74.52, 40)  # 20x normal, 1 tick behind the ask
    s.trade(101.5, buyer="SPOOFER", seller="OTHER", price=74.50, qty=2,
            resting=genuine, aggressor_side="sell")
    s.cancel(102.0, spoof)  # pulled 1.8s after entry, never filled

    alerts = detect_spoofing(s.events())
    assert len(alerts) == 1
    a = alerts.iloc[0]
    assert a["account_id"] == "SPOOFER"
    assert spoof in a["order_ids"]
    assert a["score"] > 50


def test_large_quick_cancel_without_opposite_fill_is_not_flagged():
    # Someone who changes their mind about a big order isn't spoofing:
    # there's no trade on the other side that the order could have helped.
    s = base()
    big = s.new(100.2, "SPOOFER", "sell", 74.52, 40)
    s.cancel(102.0, big)
    assert detect_spoofing(s.events()).empty


def test_big_trader_at_normal_size_is_not_flagged():
    # A hedger who always trades ~40 lots: 40 lots isn't unusual FOR THEM.
    s = Scenario()
    s.quote()
    s.background("HEDGER", n=30, qty=40)
    s.background("OTHER", n=30, qty=2, side="sell", price=74.70)
    genuine = s.new(100.0, "HEDGER", "buy", 74.50, 2)
    big = s.new(100.2, "HEDGER", "sell", 74.52, 40)
    s.trade(101.5, buyer="HEDGER", seller="OTHER", price=74.50, qty=2,
            resting=genuine, aggressor_side="sell")
    s.cancel(102.0, big)
    assert detect_spoofing(s.events()).empty


def test_large_order_that_rests_a_long_time_is_not_flagged():
    s = base()
    genuine = s.new(100.0, "SPOOFER", "buy", 74.50, 2)
    big = s.new(100.2, "SPOOFER", "sell", 74.52, 40)
    s.trade(101.5, buyer="SPOOFER", seller="OTHER", price=74.50, qty=2,
            resting=genuine, aggressor_side="sell")
    s.cancel(160.0, big)  # rested for a minute: a real offer
    assert detect_spoofing(s.events()).empty


def test_large_order_deep_in_book_is_not_flagged():
    s = base()
    genuine = s.new(100.0, "SPOOFER", "buy", 74.50, 2)
    big = s.new(100.2, "SPOOFER", "sell", 74.71, 40)  # 20 ticks away: nobody reacts
    s.trade(101.5, buyer="SPOOFER", seller="OTHER", price=74.50, qty=2,
            resting=genuine, aggressor_side="sell")
    s.cancel(102.0, big)
    assert detect_spoofing(s.events()).empty
