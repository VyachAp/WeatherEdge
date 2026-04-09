"""WeatherEdge Monitoring Dashboard.

Four-page Streamlit dashboard for monitoring the WeatherEdge trading system:
  1. Overview  — KPIs, bankroll curve, today's signals, open positions
  2. Signal Explorer — historical signal scatter, filterable, expandable details
  3. Model Calibration — reliability diagram, Brier scores, breakdown by variable
  4. Market Scanner — live active markets with edge highlighting
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# Database setup (synchronous engine for Streamlit)
# ---------------------------------------------------------------------------

_RAW_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://weather:weather@localhost:5432/weatheredge",
)
SYNC_URL = _RAW_URL.replace("+asyncpg", "")

INITIAL_BANKROLL = float(os.environ.get("INITIAL_BANKROLL", "750"))
MIN_EDGE = float(os.environ.get("MIN_EDGE", "0.10"))


@st.cache_resource
def get_engine():
    return create_engine(SYNC_URL)


# ---------------------------------------------------------------------------
# Cached query helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def fetch_current_bankroll() -> float:
    df = pd.read_sql(
        text("SELECT balance FROM bankroll_log ORDER BY timestamp DESC LIMIT 1"),
        get_engine(),
    )
    return float(df.iloc[0]["balance"]) if len(df) > 0 else INITIAL_BANKROLL


@st.cache_data(ttl=300)
def fetch_total_pnl() -> float:
    df = pd.read_sql(
        text(
            "SELECT COALESCE(SUM(pnl), 0) AS total "
            "FROM trades WHERE status IN ('won', 'lost')"
        ),
        get_engine(),
    )
    return float(df.iloc[0]["total"])


@st.cache_data(ttl=300)
def fetch_win_rate() -> float | None:
    df = pd.read_sql(
        text(
            "SELECT COUNT(*) FILTER (WHERE status = 'won') AS wins, "
            "       COUNT(*) FILTER (WHERE status IN ('won','lost')) AS total "
            "FROM trades"
        ),
        get_engine(),
    )
    total = int(df.iloc[0]["total"])
    if total == 0:
        return None
    return float(df.iloc[0]["wins"]) / total


@st.cache_data(ttl=300)
def fetch_bankroll_history() -> pd.DataFrame:
    return pd.read_sql(
        text("SELECT balance, peak, drawdown_pct, timestamp FROM bankroll_log ORDER BY timestamp"),
        get_engine(),
    )


@st.cache_data(ttl=300)
def fetch_open_positions() -> pd.DataFrame:
    df = pd.read_sql(
        text(
            "SELECT t.id, t.market_id, t.direction, t.stake_usd, t.entry_price, "
            "       t.opened_at, m.question, m.current_yes_price, m.end_date "
            "FROM trades t JOIN markets m ON t.market_id = m.id "
            "WHERE t.status = 'open'"
        ),
        get_engine(),
    )
    if df.empty:
        return df

    now = datetime.now(timezone.utc)

    def _unrealized_pnl(row):
        price = row["current_yes_price"] or 0.0
        entry = row["entry_price"] or 0.0
        if entry <= 0:
            return 0.0
        if row["direction"] == "BUY_YES":
            return row["stake_usd"] * (price / entry - 1)
        return row["stake_usd"] * ((1 - price) / entry - 1)

    df["unrealized_pnl"] = df.apply(_unrealized_pnl, axis=1)
    df["days_to_resolution"] = df["end_date"].apply(
        lambda d: max(0, (d - now).days) if pd.notna(d) else None
    )
    return df


@st.cache_data(ttl=300)
def fetch_todays_signals() -> pd.DataFrame:
    df = pd.read_sql(
        text(
            "SELECT s.id, s.market_id, s.edge, s.confidence, s.model_prob, "
            "       s.market_prob, s.direction, s.created_at, m.question, "
            "       EXISTS(SELECT 1 FROM trades t WHERE t.signal_id = s.id) AS has_trade "
            "FROM signals s JOIN markets m ON s.market_id = m.id "
            "WHERE DATE(s.created_at) = CURRENT_DATE "
            "ORDER BY s.created_at DESC"
        ),
        get_engine(),
    )
    if df.empty:
        return df

    now = datetime.now(timezone.utc)
    bankroll = fetch_current_bankroll()

    def _status(row):
        if row["has_trade"]:
            return "executed"
        created = row["created_at"]
        if hasattr(created, "tzinfo") and created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        if (now - created).total_seconds() < 3600:
            return "new"
        return "skipped"

    def _suggested_stake(row):
        mp = row["market_prob"] or 0.5
        edge = abs(row["edge"])
        odds = 1.0 / mp - 1.0 if mp > 0 else 0
        if odds <= 0:
            return 0.0
        kelly = edge / odds
        return round(min(kelly * 0.25 * bankroll, 0.05 * bankroll), 2)

    df["status"] = df.apply(_status, axis=1)
    df["suggested_stake"] = df.apply(_suggested_stake, axis=1)
    return df


@st.cache_data(ttl=300)
def fetch_active_signals_count() -> int:
    df = pd.read_sql(
        text("SELECT COUNT(*) AS cnt FROM signals WHERE DATE(created_at) = CURRENT_DATE"),
        get_engine(),
    )
    return int(df.iloc[0]["cnt"])


@st.cache_data(ttl=300)
def fetch_all_signals(
    date_from: date, date_to: date, min_edge: float, variable_type: str | None
) -> pd.DataFrame:
    query = (
        "SELECT s.id, s.market_id, s.edge, s.confidence, s.model_prob, "
        "       s.market_prob, s.gfs_prob, s.ecmwf_prob, s.direction, "
        "       s.created_at, m.question, m.parsed_variable, m.parsed_location, "
        "       m.current_yes_price, m.end_date, "
        "       CASE "
        "         WHEN EXISTS(SELECT 1 FROM trades t WHERE t.signal_id = s.id AND t.status = 'won') THEN 'won' "
        "         WHEN EXISTS(SELECT 1 FROM trades t WHERE t.signal_id = s.id AND t.status = 'lost') THEN 'lost' "
        "         ELSE 'pending' "
        "       END AS outcome "
        "FROM signals s JOIN markets m ON s.market_id = m.id "
        "WHERE s.created_at >= :date_from AND s.created_at <= :date_to "
        "  AND ABS(s.edge) >= :min_edge "
    )
    params: dict = {
        "date_from": datetime.combine(date_from, datetime.min.time()),
        "date_to": datetime.combine(date_to, datetime.max.time()),
        "min_edge": min_edge,
    }
    if variable_type:
        query += "  AND m.parsed_variable = :variable_type "
        params["variable_type"] = variable_type

    query += "ORDER BY s.created_at DESC"
    return pd.read_sql(text(query), get_engine(), params=params)


@st.cache_data(ttl=300)
def fetch_market_price_history(market_id: str) -> pd.DataFrame:
    return pd.read_sql(
        text(
            "SELECT yes_price, no_price, volume, liquidity, timestamp "
            "FROM market_snapshots WHERE market_id = :mid ORDER BY timestamp"
        ),
        get_engine(),
        params={"mid": market_id},
    )


@st.cache_data(ttl=300)
def fetch_calibration_data() -> pd.DataFrame:
    return pd.read_sql(
        text(
            "SELECT s.model_prob, s.gfs_prob, s.ecmwf_prob, s.direction, "
            "       s.created_at, m.parsed_variable, "
            "       EXTRACT(DAY FROM m.end_date - s.created_at) AS forecast_horizon, "
            "       t.status AS trade_status "
            "FROM signals s "
            "JOIN markets m ON s.market_id = m.id "
            "JOIN trades t ON t.signal_id = s.id "
            "WHERE t.status IN ('won', 'lost')"
        ),
        get_engine(),
    )


@st.cache_data(ttl=300)
def fetch_active_markets() -> pd.DataFrame:
    df = pd.read_sql(
        text(
            "SELECT m.id, m.question, m.current_yes_price, m.volume, m.liquidity, "
            "       m.end_date, m.parsed_variable, m.parsed_location, "
            "       s.model_prob, s.edge, s.confidence, s.direction "
            "FROM markets m "
            "LEFT JOIN LATERAL ( "
            "  SELECT * FROM signals WHERE market_id = m.id "
            "  ORDER BY created_at DESC LIMIT 1 "
            ") s ON true "
            "WHERE m.end_date > NOW() "
            "  AND m.parsed_variable IS NOT NULL "
            "ORDER BY ABS(s.edge) DESC NULLS LAST"
        ),
        get_engine(),
    )
    if df.empty:
        return df

    now = datetime.now(timezone.utc)
    df["market_prob"] = df["current_yes_price"]
    df["days_to_resolution"] = df["end_date"].apply(
        lambda d: max(0, (d - now).days) if pd.notna(d) else None
    )
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_reliability(
    predicted: np.ndarray, actual: np.ndarray, n_bins: int = 10
) -> tuple[list[float], list[float], list[int]]:
    """Bin predictions and compute observed frequency per bin."""
    bins = np.linspace(0, 1, n_bins + 1)
    centers: list[float] = []
    freqs: list[float] = []
    counts: list[int] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (predicted >= lo) & (predicted < hi)
        n = int(mask.sum())
        if n > 0:
            centers.append(float(predicted[mask].mean()))
            freqs.append(float(actual[mask].mean()))
            counts.append(n)
    return centers, freqs, counts


def _actual_outcome(row) -> float:
    """Determine if YES actually occurred (1.0) or not (0.0).

    A trade that WON on BUY_YES means YES occurred.
    A trade that LOST on BUY_YES means YES did NOT occur.
    Reverse for BUY_NO.
    """
    if row["trade_status"] == "won" and row["direction"] == "BUY_YES":
        return 1.0
    if row["trade_status"] == "lost" and row["direction"] == "BUY_YES":
        return 0.0
    if row["trade_status"] == "won" and row["direction"] == "BUY_NO":
        return 0.0
    # lost + BUY_NO -> YES occurred
    return 1.0


# ---------------------------------------------------------------------------
# Page 1 — Overview
# ---------------------------------------------------------------------------


def page_overview():
    st.header("Overview")

    bankroll = fetch_current_bankroll()
    total_pnl = fetch_total_pnl()
    win_rate = fetch_win_rate()
    positions = fetch_open_positions()
    signals_count = fetch_active_signals_count()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Bankroll", f"${bankroll:,.2f}")
    c2.metric("Total P&L", f"${total_pnl:+,.2f}")
    c3.metric("Win Rate", f"{win_rate:.1%}" if win_rate is not None else "N/A")
    c4.metric("Open Positions", len(positions))
    c5.metric("Today's Signals", signals_count)

    # --- Bankroll curve ---
    st.subheader("Bankroll Curve")
    hist = fetch_bankroll_history()
    if hist.empty:
        st.info("No bankroll history yet.")
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=hist["timestamp"],
                y=hist["peak"],
                mode="lines",
                line=dict(color="rgba(100,100,100,0.4)", dash="dot"),
                name="Peak",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=hist["timestamp"],
                y=hist["balance"],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(255,80,80,0.15)",
                line=dict(color="#1f77b4", width=2),
                name="Balance",
            )
        )
        fig.update_layout(
            yaxis_title="USD",
            xaxis_title="",
            template="plotly_dark",
            height=350,
            margin=dict(l=40, r=20, t=10, b=30),
            legend=dict(orientation="h", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Today's signals ---
    st.subheader("Today's Signals")
    signals = fetch_todays_signals()
    if signals.empty:
        st.info("No signals generated today.")
    else:
        display = signals[
            ["question", "edge", "confidence", "suggested_stake", "status"]
        ].copy()
        display.columns = ["Market", "Edge", "Confidence", "Suggested Stake ($)", "Status"]
        display["Edge"] = display["Edge"].map("{:+.1%}".format)
        display["Confidence"] = display["Confidence"].map("{:.1%}".format)
        st.dataframe(display, use_container_width=True, hide_index=True)

    # --- Open positions ---
    st.subheader("Open Positions")
    if positions.empty:
        st.info("No open positions.")
    else:
        display = positions[
            ["question", "entry_price", "current_yes_price", "unrealized_pnl", "days_to_resolution"]
        ].copy()
        display.columns = ["Market", "Entry Price", "Current Price", "Unrealized P&L", "Days Left"]
        display["Entry Price"] = display["Entry Price"].map("${:.3f}".format)
        display["Current Price"] = display["Current Price"].map("${:.3f}".format)
        display["Unrealized P&L"] = display["Unrealized P&L"].map("${:+,.2f}".format)
        st.dataframe(display, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page 2 — Signal Explorer
# ---------------------------------------------------------------------------


def page_signal_explorer():
    st.header("Signal Explorer")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        date_range = st.date_input(
            "Date range",
            value=(date.today() - timedelta(days=30), date.today()),
            max_value=date.today(),
        )
    with col2:
        min_edge_filter = st.slider("Min edge", 0.0, 0.5, 0.0, 0.01)
    with col3:
        var_options = ["All", "temperature", "precipitation", "wind_speed"]
        var_choice = st.selectbox("Variable type", var_options)

    if isinstance(date_range, tuple) and len(date_range) == 2:
        d_from, d_to = date_range
    else:
        d_from = d_to = date_range

    variable = var_choice if var_choice != "All" else None

    df = fetch_all_signals(d_from, d_to, min_edge_filter, variable)

    if df.empty:
        st.info("No signals match the current filters.")
        return

    # Scatter plot
    color_map = {"won": "#2ca02c", "lost": "#d62728", "pending": "#7f7f7f"}
    fig = px.scatter(
        df,
        x="edge",
        y="confidence",
        color="outcome",
        color_discrete_map=color_map,
        hover_data=["question"],
        labels={"edge": "Edge", "confidence": "Confidence", "outcome": "Outcome"},
    )
    fig.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Expandable signal details
    st.subheader("Signal Details")
    for _, row in df.iterrows():
        label = f"[{row['outcome'].upper()}] {row['question'][:80]}"
        with st.expander(label):
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("GFS Prob", f"{row['gfs_prob']:.1%}" if pd.notna(row["gfs_prob"]) else "N/A")
            mc2.metric("ECMWF Prob", f"{row['ecmwf_prob']:.1%}" if pd.notna(row["ecmwf_prob"]) else "N/A")
            mc3.metric("Consensus", f"{row['model_prob']:.1%}")

            history = fetch_market_price_history(row["market_id"])
            if not history.empty:
                hfig = px.line(
                    history,
                    x="timestamp",
                    y="yes_price",
                    labels={"yes_price": "YES Price", "timestamp": ""},
                )
                hfig.update_layout(template="plotly_dark", height=250, margin=dict(l=30, r=10, t=10, b=20))
                st.plotly_chart(hfig, use_container_width=True)
            else:
                st.caption("No price history available.")


# ---------------------------------------------------------------------------
# Page 3 — Model Calibration
# ---------------------------------------------------------------------------


def page_model_calibration():
    st.header("Model Calibration")

    df = fetch_calibration_data()
    if df.empty:
        st.info("Not enough resolved trades for calibration analysis.")
        return

    df["actual"] = df.apply(_actual_outcome, axis=1)

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        var_filter = st.selectbox("Variable", ["All", "temperature", "precipitation", "wind_speed"], key="cal_var")
    with col2:
        horizon_filter = st.radio("Forecast horizon", ["All", "1-3 days", "4-7 days"], horizontal=True)

    filtered = df.copy()
    if var_filter != "All":
        filtered = filtered[filtered["parsed_variable"] == var_filter]
    if horizon_filter == "1-3 days":
        filtered = filtered[filtered["forecast_horizon"].between(1, 3)]
    elif horizon_filter == "4-7 days":
        filtered = filtered[filtered["forecast_horizon"].between(4, 7)]

    if filtered.empty:
        st.warning("No data for the selected filters.")
        return

    # --- Reliability diagram ---
    st.subheader("Reliability Diagram")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name="Perfect",
            showlegend=True,
        )
    )

    prob_columns = [
        ("Consensus", "model_prob"),
        ("GFS", "gfs_prob"),
        ("ECMWF", "ecmwf_prob"),
    ]
    colors = {"Consensus": "#1f77b4", "GFS": "#ff7f0e", "ECMWF": "#2ca02c"}

    for name, col in prob_columns:
        sub = filtered.dropna(subset=[col])
        if sub.empty:
            continue
        preds = sub[col].values
        actual_arr = sub["actual"].values
        centers, freqs, counts = compute_reliability(preds, actual_arr)
        fig.add_trace(
            go.Scatter(
                x=centers, y=freqs,
                mode="lines+markers",
                name=f"{name} (n={sum(counts)})",
                line=dict(color=colors[name]),
                marker=dict(size=8),
            )
        )

    fig.update_layout(
        xaxis_title="Predicted Probability",
        yaxis_title="Observed Frequency",
        template="plotly_dark",
        height=400,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Brier score over time ---
    st.subheader("Brier Score Over Time")
    filtered["week"] = pd.to_datetime(filtered["created_at"]).dt.to_period("W").dt.start_time

    brier_rows = []
    for name, col in [("Consensus", "model_prob"), ("GFS", "gfs_prob"), ("ECMWF", "ecmwf_prob")]:
        sub = filtered.dropna(subset=[col])
        if sub.empty:
            continue
        grouped = sub.groupby("week").apply(
            lambda g: ((g[col] - g["actual"]) ** 2).mean(), include_groups=False
        ).reset_index()
        grouped.columns = ["week", "brier"]
        grouped["model"] = name
        brier_rows.append(grouped)

    if brier_rows:
        brier_df = pd.concat(brier_rows)
        bfig = px.line(
            brier_df, x="week", y="brier", color="model",
            labels={"brier": "Brier Score", "week": ""},
            color_discrete_map=colors,
        )
        bfig.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(bfig, use_container_width=True)
    else:
        st.info("Not enough data for Brier score chart.")


# ---------------------------------------------------------------------------
# Page 4 — Market Scanner
# ---------------------------------------------------------------------------


def page_market_scanner():
    st.header("Market Scanner")

    df = fetch_active_markets()
    if df.empty:
        st.info("No active weather markets found.")
        return

    display = df[[
        "question", "model_prob", "market_prob", "edge", "volume",
        "liquidity", "days_to_resolution",
    ]].copy()
    display.columns = [
        "Market", "Model Prob", "Market Prob", "Edge",
        "Volume", "Liquidity", "Days Left",
    ]

    # Highlight rows with edge above threshold
    def _highlight_edge(row):
        edge_val = row.get("Edge")
        if pd.notna(edge_val) and abs(edge_val) >= MIN_EDGE:
            return ["background-color: rgba(31,119,180,0.2)"] * len(row)
        return [""] * len(row)

    styled = display.style.apply(_highlight_edge, axis=1).format(
        {
            "Model Prob": "{:.1%}",
            "Market Prob": "{:.1%}",
            "Edge": "{:+.1%}",
            "Volume": "${:,.0f}",
            "Liquidity": "${:,.0f}",
        },
        na_rep="—",
    )
    st.dataframe(styled, use_container_width=True, hide_index=True, height=600)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="WeatherEdge Monitor",
        layout="wide",
        page_icon="\u2600",
    )

    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Signal Explorer", "Model Calibration", "Market Scanner"],
    )

    if page == "Overview":
        page_overview()
    elif page == "Signal Explorer":
        page_signal_explorer()
    elif page == "Model Calibration":
        page_model_calibration()
    elif page == "Market Scanner":
        page_market_scanner()

    # Auto-refresh every 5 minutes
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = datetime.now(timezone.utc)

    elapsed = (datetime.now(timezone.utc) - st.session_state.last_refresh).total_seconds()
    if elapsed > 300:
        st.session_state.last_refresh = datetime.now(timezone.utc)
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()
