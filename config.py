"""
Every detection threshold lives in this one file.

Each number below is a judgement call that trades off two kinds of error:

* set it LOOSER  -> more alerts, higher recall, but analysts waste time on
                    innocent activity (false positives);
* set it TIGHTER -> fewer, cleaner alerts, but real manipulation slips
                    through (false negatives).

A surveillance team tunes these against known cases and analyst capacity.
`evaluate.py` runs a sensitivity sweep on the main spoofing threshold so you can
see that tradeoff directly. DESIGN.md explains the reasoning in plain English.
"""
from dataclasses import dataclass

# Minimum price increment of the simulated contract. ICE Brent trades in
# $0.01/bbl ticks. Distances such as "near the touch" are measured in ticks.
TICK_SIZE = 0.01


@dataclass(frozen=True)
class WashTradeConfig:
    # Wash trades by the same account (or linked accounts) that happen within
    # this many seconds of each other are grouped into one alert.
    # Larger -> fewer, bigger alerts; may merge unrelated episodes.
    cluster_gap_s: float = 60.0

    # Minimum number of self/linked trades in a cluster before we alert.
    # 1 would flag every accidental self-match (common when an account runs
    # several algos without self-match prevention). 2+ requires a repeated
    # pattern, which is what wash trading looks like.
    min_trades: int = 2

    # --- "Hidden link" rule: accounts NOT known to be linked that trade back
    # and forth with each other (A sells to B, then B sells the same size back
    # to A at about the same price) -- a round trip with no real risk transfer.
    # Max seconds between the two legs of a round trip.
    # Larger -> catches slower round trips, but random two-way trading between
    # active accounts starts to look like round trips.
    round_trip_window_s: float = 30.0

    # Max price difference (ticks) between the two legs of a round trip.
    # A real round trip at a very different price is a genuine P&L event.
    round_trip_max_price_diff_ticks: int = 2

    # Round trips needed between the same unlinked pair before alerting.
    # This rule has no ownership data behind it, so we demand repetition to
    # keep false positives down.
    min_round_trips: int = 2


@dataclass(frozen=True)
class SpoofingConfig:
    # MAIN THRESHOLD (swept in evaluate.py).
    # An order is "large" if its size is >= this multiple of the account's own
    # median order size. Measuring against the account's own habits means a
    # big hedger who always trades 50 lots isn't flagged for trading 50 lots.
    # Lower -> catches smaller spoof orders, but ordinary size variation
    # starts producing alerts.
    size_multiple: float = 5.0

    # The order must ALSO be >= this multiple of the market-wide median order
    # size, so a tiny account whose median is 1 lot isn't flagged for a
    # 5-lot order that can't move anyone's view of the market.
    min_market_multiple: float = 2.0

    # Max seconds between order entry and cancellation. Spoof orders are meant
    # to be seen, not filled, so they're pulled quickly.
    # Longer -> more recall, but genuine traders who change their mind after
    # 10-20 seconds get swept in.
    max_order_lifetime_s: float = 5.0

    # Max distance (ticks) from the best price on the order's own side when it
    # was entered. Orders deep in the book barely affect other traders, so
    # spoofers place them at or near the touch.
    max_ticks_from_touch: int = 3

    # Spoof orders are canceled "without filling". We allow a small partial
    # fill (spoofers sometimes get clipped) but not a meaningful one.
    max_filled_fraction: float = 0.10

    # The same account must get a fill on the OPPOSITE side between
    # (order entry - before) and (cancel + after). That opposite-side fill is
    # the "genuine" order the spoof was meant to help -- it's the motive.
    fill_window_before_s: float = 2.0
    fill_window_after_s: float = 1.0

    # Accounts with fewer orders than this don't have a reliable "normal size";
    # we fall back to the market-wide median for them.
    min_orders_for_baseline: int = 20

    # Flagged orders from the same account within this many seconds are
    # combined into one alert (one episode = one case for an analyst).
    alert_merge_gap_s: float = 5.0


@dataclass(frozen=True)
class LayeringConfig:
    # Minimum number of DISTINCT price levels the account must have stacked on
    # one side. 2 levels is common for normal traders; 3+ on one side, all
    # pulled together, is the layering signature.
    min_levels: int = 3

    # The stacked orders must have been entered within this many seconds
    # before the opposite-side fill (they were placed to set it up).
    lookback_s: float = 10.0

    # ...and canceled within this many seconds AFTER the opposite-side fill.
    # Shorter -> stricter; longer -> catches slower layerers but also picks up
    # routine end-of-strategy cleanup.
    cancel_window_s: float = 5.0

    # Share of the stacked quantity that must be canceled in that window.
    # 0.8 allows one layer to get partially hit and still count.
    min_cancel_fraction: float = 0.8

    # One-sidedness: stacked-side quantity / (stacked side + the account's
    # resting quantity on the fill side, just before the fill).
    # Market makers stack several levels on BOTH sides and requote after every
    # fill, which otherwise looks just like layering. Requiring >= 0.8 means
    # the account was overwhelmingly leaning on one side.
    min_one_sidedness: float = 0.8

    # Fills from the same account within this many seconds are merged into a
    # single alert.
    alert_merge_gap_s: float = 10.0


@dataclass(frozen=True)
class RealDataConfig:
    """Order-level spoofing screen for real Coinbase data (real_data/).

    Public data has no account IDs and the public book feed has no order IDs,
    so "orders" are inferred from size jumps at a price level."""

    # Only look at size added within this distance of the best price on its
    # side, in basis points of the mid price (1 bp = 0.01%; ~$6.5 at $65k).
    # Wider -> includes deep orders that barely affect anyone.
    max_distance_bps: float = 2.0

    # "Large" = the size jump is >= this multiple of the median size added near
    # the touch... (no account history exists, so the baseline is market-wide)
    size_multiple: float = 10.0

    # ...and worth at least this many dollars. Filters out jumps that are big
    # relative to a tiny median but economically trivial.
    min_notional_usd: float = 100_000.0

    # The added size counts as "gone" once the level falls back by this share
    # of the jump.
    removal_fraction: float = 0.9

    # Flag if it disappears within this many seconds...
    max_lifetime_s: float = 5.0

    # ...with at most this share of it traded at that price in the meantime.
    max_filled_fraction: float = 0.10

    # How long to follow each large add before calling it "still resting".
    follow_s: float = 60.0


# Evaluation: an alert matches a ground-truth incident if it is for the same
# pattern, shares an account, and its time window overlaps the incident window
# padded by this many seconds on each side.
MATCH_TOLERANCE_S = 5.0

# Values of SpoofingConfig.size_multiple tried in the sensitivity sweep.
SPOOFING_SWEEP = [1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15]

WASH = WashTradeConfig()
SPOOFING = SpoofingConfig()
LAYERING = LayeringConfig()
REAL_DATA = RealDataConfig()
