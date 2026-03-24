import streamlit as st
from datetime import date, timedelta

from config.settings import BEAR_LABELS
from src.data.db import init_db
from src.data.event_repository import events_count_for_trap, leaderboard_for_trap


def start_date_for_window(window: str):
    if window == "Last 7 days":
        return date.today() - timedelta(days=7)
    if window == "Last 30 days":
        return date.today() - timedelta(days=30)
    return None


def render_trap_leaderboard(bear_label: str, start_date):
    total_events = events_count_for_trap(bear_label=bear_label, start_date=start_date)
    df = leaderboard_for_trap(bear_label=bear_label, start_date=start_date)

    if df.empty:
        st.info("No damage data for this time window yet.")
        return

    if total_events > 0:
        df["attendance_rate"] = (df["events_attended"] / total_events * 100).round(1)
    else:
        df["attendance_rate"] = 0.0

    df["avg_damage_when_present"] = df["avg_damage_when_present"].round(0).astype("int64")
    df["rank"] = df["total_damage"].rank(method="dense", ascending=False).astype("int64")
    df["total_damage_display"] = df["total_damage"].map(lambda value: f"{int(value):,}")
    df["avg_damage_when_present_display"] = df["avg_damage_when_present"].map(lambda value: f"{int(value):,}")

    st.caption(f"Total events in window: {total_events}")
    st.dataframe(
        df[["rank", "name", "total_damage_display", "events_attended", "attendance_rate", "avg_damage_when_present_display"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "rank": st.column_config.NumberColumn("Rank"),
            "name": st.column_config.TextColumn("Player"),
            "total_damage_display": st.column_config.TextColumn("Total Damage"),
            "events_attended": st.column_config.NumberColumn("Events Attended"),
            "attendance_rate": st.column_config.NumberColumn("Attendance %"),
            "avg_damage_when_present_display": st.column_config.TextColumn("Avg Damage (Present)"),
        },
        height="content",
    )


init_db()

st.title("Leaderboards")

window = st.selectbox("Time window", ["All-time", "Last 7 days", "Last 30 days"], index=0)
window_start = start_date_for_window(window)

left_col, right_col = st.columns(2)

with left_col:
    st.subheader(f"Leaderboard for {BEAR_LABELS[0]}")
    render_trap_leaderboard(BEAR_LABELS[0], window_start)

with right_col:
    st.subheader(f"Leaderboard for {BEAR_LABELS[1]}")
    render_trap_leaderboard(BEAR_LABELS[1], window_start)
