"""Tests for the shared plumbing: book replay, data generator, evaluation."""
import pandas as pd

from detectors.common import add_bbo
from evaluate import score
from generate_data import generate
from tests.helpers import Scenario


def test_book_replay_tracks_best_bid_and_ask():
    s = Scenario()
    b1 = s.new(1, "A", "buy", 74.50, 5)
    s.new(2, "B", "sell", 74.53, 5)
    s.new(3, "C", "buy", 74.51, 1)  # improves the bid
    s.cancel(4, b1)
    s.new(5, "D", "sell", 74.60, 1)  # last row: shows the state after everything above
    ev = add_bbo(s.events())
    last = ev.iloc[-1]
    assert last["best_bid"] == 74.51
    assert last["best_ask"] == 74.53


def test_generator_is_reproducible_and_labeled():
    e1, l1, _ = generate(seed=7)
    e2, l2, _ = generate(seed=7)
    pd.testing.assert_frame_equal(e1, e2)
    assert set(l1["pattern"]) == {"wash_trade", "spoofing", "layering"}
    assert set(e1["event_type"]) == {"new", "cancel", "fill"}
    fills = e1[e1["event_type"] == "fill"]
    assert fills["counterparty_account_id"].notna().all()
    # every trade has exactly one buy row and one sell row
    assert (fills.groupby("trade_id")["side"].apply(sorted).map(tuple) == ("buy", "sell")).all()


def test_scoring_counts_matches_by_pattern_account_and_time():
    labels = pd.DataFrame([
        {"incident_id": "I1", "pattern": "spoofing", "account_id": "S",
         "related_accounts": "S", "start_ms": 1000, "end_ms": 5000},
        {"incident_id": "I2", "pattern": "spoofing", "account_id": "S",
         "related_accounts": "S", "start_ms": 900_000, "end_ms": 905_000},
    ])
    alerts = pd.DataFrame([
        {"alert_id": "A1", "pattern": "spoofing", "account_id": "S", "related_accounts": "S",
         "start_ms": 2000, "end_ms": 4000},  # matches I1
        {"alert_id": "A2", "pattern": "spoofing", "account_id": "X", "related_accounts": "X",
         "start_ms": 2000, "end_ms": 4000},  # wrong account -> false positive
    ])
    r = score(alerts, labels, ["spoofing"]).iloc[0]
    assert (r["precision"], r["recall"]) == (0.5, 0.5)
