from detectors.layering import detect_layering
from tests.helpers import Scenario


def test_flags_one_sided_stack_canceled_after_opposite_fill():
    s = Scenario()
    s.quote()
    genuine = s.new(100.0, "LAYERER", "sell", 74.51, 2)
    layers = [s.new(100.1 + i * 0.1, "LAYERER", "buy", p, 8)
              for i, p in enumerate([74.49, 74.48, 74.47, 74.46])]
    s.trade(102.0, buyer="OTHER", seller="LAYERER", price=74.51, qty=2, resting=genuine)
    for i, oid in enumerate(layers):
        s.cancel(102.5 + i * 0.01, oid)

    alerts = detect_layering(s.events())
    assert len(alerts) == 1
    a = alerts.iloc[0]
    assert a["account_id"] == "LAYERER"
    assert all(oid in a["order_ids"] for oid in layers)
    assert "4 buy orders" in a["evidence"]


def test_market_maker_requoting_both_sides_is_not_flagged():
    # A market maker quotes three levels on BOTH sides, gets lifted on the
    # offer, then cancels everything to requote. Same cancel burst as layering,
    # but the account was two-sided, so it's not flagged.
    s = Scenario()
    bids = [s.new(100.0, "MM", "buy", p, 5) for p in (74.50, 74.49, 74.48)]
    asks = [s.new(100.0, "MM", "sell", p, 5) for p in (74.51, 74.52, 74.53)]
    s.trade(102.0, buyer="OTHER", seller="MM", price=74.51, qty=5, resting=asks[0])
    for oid in bids + asks[1:]:
        s.cancel(102.3, oid)
    assert detect_layering(s.events()).empty


def test_stack_left_resting_after_fill_is_not_flagged():
    # Same stack, but the orders stay in the book: they were genuine bids.
    s = Scenario()
    s.quote()
    genuine = s.new(100.0, "TRADER", "sell", 74.51, 2)
    layers = [s.new(100.1, "TRADER", "buy", p, 8) for p in (74.49, 74.48, 74.47, 74.46)]
    s.trade(102.0, buyer="OTHER", seller="TRADER", price=74.51, qty=2, resting=genuine)
    for oid in layers:
        s.cancel(200.0, oid)
    assert detect_layering(s.events()).empty


def test_two_levels_is_not_enough():
    s = Scenario()
    s.quote()
    genuine = s.new(100.0, "TRADER", "sell", 74.51, 2)
    layers = [s.new(100.1, "TRADER", "buy", p, 8) for p in (74.49, 74.48)]
    s.trade(102.0, buyer="OTHER", seller="TRADER", price=74.51, qty=2, resting=genuine)
    for oid in layers:
        s.cancel(102.5, oid)
    assert detect_layering(s.events()).empty
