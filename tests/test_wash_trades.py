import pandas as pd

from detectors.wash_trades import detect_wash_trades
from tests.helpers import Scenario

LINKS = pd.DataFrame({"account_a": ["FIRM_A1"], "account_b": ["FIRM_A2"],
                      "relationship": ["same beneficial owner"]})


def test_flags_repeated_self_match():
    s = Scenario()
    s.trade(10.0, buyer="W", seller="W", price=74.50, qty=5)
    s.trade(12.0, buyer="W", seller="W", price=74.50, qty=5)
    alerts = detect_wash_trades(s.events())
    assert len(alerts) == 1
    assert alerts.iloc[0]["account_id"] == "W"
    assert "self-match" in alerts.iloc[0]["evidence"]


def test_single_accidental_self_match_is_not_flagged():
    # One self-match is usually two of a firm's own algos crossing by accident.
    s = Scenario()
    s.trade(10.0, buyer="W", seller="W", price=74.50, qty=5)
    s.trade(500.0, buyer="W", seller="X", price=74.52, qty=5)
    assert detect_wash_trades(s.events()).empty


def test_flags_linked_account_round_trip():
    s = Scenario()
    s.trade(10.0, buyer="FIRM_A2", seller="FIRM_A1", price=74.50, qty=7)
    s.trade(11.5, buyer="FIRM_A1", seller="FIRM_A2", price=74.50, qty=7)
    alerts = detect_wash_trades(s.events(), LINKS)
    assert len(alerts) == 1
    assert set(alerts.iloc[0]["related_accounts"].split(";")) == {"FIRM_A1", "FIRM_A2"}


def test_same_trades_between_unlinked_accounts_need_more_evidence():
    # Without ownership data, one back-and-forth isn't enough to alert...
    s = Scenario()
    s.trade(10.0, buyer="FIRM_A2", seller="FIRM_A1", price=74.50, qty=7)
    s.trade(11.5, buyer="FIRM_A1", seller="FIRM_A2", price=74.50, qty=7)
    assert detect_wash_trades(s.events(), account_links=None).empty


def test_flags_repeated_round_trips_between_unlinked_accounts():
    # ...but two matched round trips in seconds suggests a hidden link.
    s = Scenario()
    for t in (10.0, 13.0):
        s.trade(t, buyer="C", seller="D", price=74.50, qty=4)
        s.trade(t + 1.0, buyer="D", seller="C", price=74.50, qty=4)
    alerts = detect_wash_trades(s.events())
    assert len(alerts) == 1
    assert "NOT known to be linked" in alerts.iloc[0]["evidence"]
    assert alerts.iloc[0]["score"] < 70  # weaker evidence than a known link


def test_one_way_trading_between_unlinked_accounts_is_not_flagged():
    # C keeps buying from D: real risk transfer, not a wash.
    s = Scenario()
    for t in (10.0, 11.0, 12.0, 13.0):
        s.trade(t, buyer="C", seller="D", price=74.50, qty=4)
    assert detect_wash_trades(s.events()).empty
