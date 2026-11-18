"""
Streamlit dashboard: browse alerts and inspect a flagged account's order flow.

    streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from detectors import run_all_detectors
from detectors.common import add_bbo, order_summary
from evaluate import match_alerts, score, spoofing_sweep
from generate_data import generate

DATA = Path(__file__).parent / "data"
BUY, SELL = "#2a78d6", "#eb6834"  # blue / orange: identity colors, colorblind-safe pair
INK, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#e6e5e0", "rgba(82, 81, 78, 0.10)"
PATTERN_NAMES = {"spoofing": "Spoofing", "layering": "Layering", "wash_trade": "Wash trade"}

st.set_page_config(page_title="Trade Surveillance", page_icon=":mag:", layout="wide")


# ---------------------------------------------------------------- data
@st.cache_data(show_spinner="Loading order events...")
def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not (DATA / "events.csv").exists():
        DATA.mkdir(exist_ok=True)
        e, l, k = generate()
        e.to_csv(DATA / "events.csv", index=False, lineterminator="\n")
        l.to_csv(DATA / "labels.csv", index=False, lineterminator="\n")
        k.to_csv(DATA / "account_links.csv", index=False, lineterminator="\n")
    events = pd.read_csv(DATA / "events.csv")
    labels = pd.read_csv(DATA / "labels.csv")
    links = pd.read_csv(DATA / "account_links.csv")
    alerts_path = DATA / "alerts.csv"
    alerts = pd.read_csv(alerts_path) if alerts_path.exists() else run_all_detectors(events, links)
    return add_bbo(events), labels, links, alerts


@st.cache_data
def orders_table(events: pd.DataFrame) -> pd.DataFrame:
    return order_summary(events)


@st.cache_data
def sweep_table(events: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    return spoofing_sweep(events, labels)


events, labels, links, alerts = load()
orders = orders_table(events)

# Attach ground truth to each alert so the analyst view shows TP/FP.
m = match_alerts(alerts, labels)
same = m[m["alert_pattern"] == m["incident_pattern"]].groupby("alert_id")["incident_id"].agg(", ".join)
alerts = alerts.assign(
    ground_truth=alerts["alert_id"].map(same).fillna("none (false positive)"),
    start=pd.to_datetime(alerts["start_ms"], unit="ms").dt.strftime("%H:%M:%S"),
    duration_s=((alerts["end_ms"] - alerts["start_ms"]) / 1000).round(1),
)

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Filters")
patterns = st.sidebar.multiselect("Pattern", list(PATTERN_NAMES), default=list(PATTERN_NAMES),
                                  format_func=PATTERN_NAMES.get)
accts = ["All accounts"] + sorted(alerts["account_id"].unique())
account = st.sidebar.selectbox("Account", accts)
min_score = st.sidebar.slider("Minimum score", 0, 100, 0, step=5)
st.sidebar.caption(
    "Data: one simulated trading day of a Brent-style futures contract with injected, "
    "labeled manipulation. Ground truth is shown only because this is synthetic data -- "
    "a real analyst wouldn't have it.")

view = alerts[alerts["pattern"].isin(patterns) & (alerts["score"] >= min_score)]
if account != "All accounts":
    view = view[(view["account_id"] == account)
                | view["related_accounts"].str.contains(account, regex=False)]
view = view.sort_values("score", ascending=False)

# ---------------------------------------------------------------- header
st.title("Trade Surveillance")
st.caption("Market-abuse alerts for futures order flow: wash trades, spoofing and layering.")

c = st.columns(5)
c[0].metric("Order events", f"{len(events):,}")
c[1].metric("Accounts", events["account_id"].nunique())
for i, p in enumerate(PATTERN_NAMES, start=2):
    c[i].metric(f"{PATTERN_NAMES[p]} alerts", int((alerts["pattern"] == p).sum()))

tab_alerts, tab_perf = st.tabs(["Alerts", "Detector performance"])


# ---------------------------------------------------------------- timeline
def timeline(alert: pd.Series) -> go.Figure:
    # Pad the window by about the alert's own length (at least 4s) so the
    # orders inside the alert stay readable.
    pad_ms = max(4000, alert["end_ms"] - alert["start_ms"])
    t0, t1 = alert["start_ms"] - pad_ms, alert["end_ms"] + pad_ms
    accounts = set(str(alert["related_accounts"]).split(";"))
    flagged = set(str(alert["order_ids"]).split(";"))
    ts = lambda ms: pd.to_datetime(ms, unit="ms")  # noqa: E731

    fig = go.Figure()
    # Market context: best bid / ask, rebuilt by replaying the book.
    # (Fill rows are skipped: the book is momentarily crossed while an
    # incoming order is being matched.)
    mkt = events[(events["timestamp"] >= t0) & (events["timestamp"] <= t1)
                 & (events["event_type"] != "fill")]
    for col, name, dash in (("best_bid", "Best bid", "solid"), ("best_ask", "Best ask", "dot")):
        fig.add_trace(go.Scatter(x=ts(mkt["timestamp"]), y=mkt[col], name=name, mode="lines",
                                 line=dict(color="#a3a29b", width=1.5, dash=dash, shape="hv"),
                                 hovertemplate=f"{name} %{{y:.2f}}<extra></extra>"))

    o = orders[orders["account_id"].isin(accounts) & (orders["t_new"] <= t1)
               & (orders["t_end"] >= t0)].copy()
    o["is_flagged"] = o["order_id"].isin(flagged)
    o["t_stop"] = np.minimum(o["t_end"].replace(np.inf, t1), t1)

    for side, color in (("buy", BUY), ("sell", SELL)):
        so = o[o["side"] == side]
        for flag, opacity, width in ((True, 1.0, 4), (False, 0.35, 2)):
            seg = so[so["is_flagged"] == flag]
            if seg.empty:
                continue
            xs, ys = [], []
            for r in seg.itertuples():
                xs += [ts(r.t_new), ts(r.t_stop), None]
                ys += [r.price, r.price, None]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", opacity=opacity, line=dict(color=color, width=width),
                name=f"{side.title()} order{' (flagged)' if flag else ''}", hoverinfo="skip"))
        # entry markers sized by quantity
        fig.add_trace(go.Scatter(
            x=ts(so["t_new"]), y=so["price"], mode="markers", showlegend=False,
            marker=dict(color=color, size=8 + 2 * np.sqrt(so["qty"]),
                        line=dict(color="white", width=2),
                        opacity=np.where(so["is_flagged"], 1.0, 0.45)),
            customdata=np.stack([so["order_id"], so["account_id"], so["qty"]], axis=-1),
            hovertemplate=(f"<b>New {side}</b> %{{customdata[0]}}<br>%{{customdata[1]}}<br>"
                           "%{customdata[2]} lots @ %{y:.2f}<br>%{x|%H:%M:%S.%L}<extra></extra>")))

    ev = events[events["account_id"].isin(accounts) & (events["timestamp"] >= t0)
                & (events["timestamp"] <= t1)]
    for etype, symbol, name in (("cancel", "x-thin", "Cancel"), ("fill", "star", "Fill")):
        e = ev[ev["event_type"] == etype]
        if e.empty:
            continue
        fig.add_trace(go.Scatter(
            x=ts(e["timestamp"]), y=e["price"], mode="markers", name=name,
            marker=dict(symbol=symbol, size=12 if etype == "fill" else 10,
                        color=np.where(e["side"] == "buy", BUY, SELL),
                        line=dict(width=2, color=np.where(e["side"] == "buy", BUY, SELL))
                        if etype == "cancel" else dict(width=1.5, color="white")),
            customdata=np.stack([e["order_id"], e["side"], e["quantity"],
                                 e["counterparty_account_id"].fillna("")], axis=-1),
            hovertemplate=(f"<b>{name}</b> %{{customdata[1]}} %{{customdata[0]}}<br>"
                           "%{customdata[2]} lots @ %{y:.2f}<br>counterparty %{customdata[3]}"
                           "<br>%{x|%H:%M:%S.%L}<extra></extra>")))

    fig.add_vrect(x0=ts(alert["start_ms"]), x1=ts(alert["end_ms"]), fillcolor=BAND,
                  line_width=0, layer="below", annotation_text="alert window",
                  annotation_position="top left", annotation_font_color=MUTED)
    fig.update_layout(
        height=430, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
        hovermode="closest", font=dict(color=INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=MUTED)),
        xaxis=dict(showgrid=False, linecolor="#c3c2b7", tickformat="%H:%M:%S",
                   tickfont=dict(color=MUTED)),
        yaxis=dict(title=dict(text="Price ($/bbl)", font=dict(color=MUTED)), gridcolor=GRID,
                   tickformat=".2f", dtick=config.TICK_SIZE, zeroline=False,
                   tickfont=dict(color=MUTED)),
    )
    return fig


# ---------------------------------------------------------------- alerts tab
with tab_alerts:
    st.subheader(f"{len(view)} alerts")
    cols = ["alert_id", "pattern", "account_id", "related_accounts", "start", "duration_s",
            "score", "ground_truth", "evidence"]
    sel = st.dataframe(
        view[cols], hide_index=True, use_container_width=True, height=280,
        on_select="rerun", selection_mode="single-row",
        column_config={
            "pattern": st.column_config.TextColumn("Pattern"),
            "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
            "evidence": st.column_config.TextColumn("Evidence", width="large"),
            "ground_truth": st.column_config.TextColumn("Ground truth"),
        })

    if view.empty:
        st.info("No alerts match the filters.")
    else:
        rows = sel.selection.rows if sel and sel.selection else []
        linked = st.query_params.get("alert")  # e.g. ?alert=AL0012 opens that alert
        if rows:
            alert = view.iloc[rows[0]]
        elif linked in set(view["alert_id"]):
            alert = view[view["alert_id"] == linked].iloc[0]
        else:
            alert = view.iloc[0]
        st.subheader(f"{alert['alert_id']} · {PATTERN_NAMES[alert['pattern']]} · "
                     f"{alert['related_accounts'].replace(';', ' + ')}")
        st.markdown(f"**Evidence.** {alert['evidence']}")
        st.caption(f"Score {alert['score']} / 100 · ground truth: {alert['ground_truth']} · "
                   "Select a row above to inspect a different alert.")
        st.plotly_chart(timeline(alert), use_container_width=True)
        st.caption("Lines show how long each order rested (bold = orders cited by the alert). "
                   "Dot size = order quantity. Gray lines = market best bid/ask.")
        with st.expander("Raw events in this window"):
            pad = max(4000, alert["end_ms"] - alert["start_ms"])
            accts_ = set(str(alert["related_accounts"]).split(";"))
            raw = events[events["account_id"].isin(accts_)
                         & events["timestamp"].between(alert["start_ms"] - pad, alert["end_ms"] + pad)]
            raw = raw.assign(time=pd.to_datetime(raw["timestamp"], unit="ms").dt.strftime("%H:%M:%S.%f").str[:-3])
            st.dataframe(raw[["time", "order_id", "account_id", "side", "price", "quantity",
                              "event_type", "counterparty_account_id", "best_bid", "best_ask"]],
                         hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- performance tab
with tab_perf:
    st.subheader("Accuracy against injected ground truth")
    metrics = score(alerts, labels)
    st.dataframe(metrics.assign(pattern=metrics["pattern"].map(PATTERN_NAMES)), hide_index=True,
                 use_container_width=True)
    st.subheader("Spoofing threshold sensitivity")
    sw = sweep_table(events, labels)
    fig = go.Figure()
    for col, color in (("precision", BUY), ("recall", SELL)):
        fig.add_trace(go.Scatter(x=sw["size_multiple"], y=sw[col], name=col.title(),
                                 mode="lines+markers", line=dict(color=color, width=2),
                                 marker=dict(size=8, line=dict(color="white", width=2)),
                                 hovertemplate=f"{col.title()} %{{y:.2f}} at %{{x}}x<extra></extra>"))
    fig.add_vline(x=config.SPOOFING.size_multiple, line_dash="dash", line_color=MUTED,
                  annotation_text=f"default {config.SPOOFING.size_multiple:g}x",
                  annotation_font_color=MUTED)
    fig.update_layout(height=360, plot_bgcolor="white", hovermode="x unified",
                      margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.08, x=0),
                      xaxis=dict(title="Order size / account's median size", showgrid=False,
                                 linecolor="#c3c2b7"),
                      yaxis=dict(range=[0, 1.05], gridcolor=GRID, zeroline=False))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Lower thresholds catch more spoofing (recall) but flood analysts with innocent "
               "large orders (precision). Every threshold is in config.py.")
    st.dataframe(sw, hide_index=True, use_container_width=True)
