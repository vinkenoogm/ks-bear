import streamlit as st
import pandas as pd
from datetime import date, timedelta
from sqlalchemy import text

from config.settings import BEAR_LABELS
from db import engine, init_db


def start_date_for_window(window: str):
    if window == "Last 7 days":
        return date.today() - timedelta(days=7)
    if window == "Last 30 days":
        return date.today() - timedelta(days=30)
    return None


def events_count_for_trap(bear_label: str, start_date):
    with engine.begin() as conn:
        return int(
            conn.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM events
                    WHERE bear_label = :bear_label
                      AND (:start_date IS NULL OR event_date >= :start_date);
                    """
                ),
                {"bear_label": bear_label, "start_date": start_date},
            ).scalar_one()
        )


def leaderboard_for_trap(bear_label: str, start_date) -> pd.DataFrame:
    query = text(
        """
        SELECT
            p.name,
            COALESCE(SUM(CASE WHEN e.id IS NOT NULL THEN d.dmg ELSE 0 END), 0) AS total_damage,
            COALESCE(SUM(CASE WHEN e.id IS NOT NULL AND d.dmg > 0 THEN 1 ELSE 0 END), 0) AS events_attended,
            COALESCE(AVG(CASE WHEN e.id IS NOT NULL AND d.dmg > 0 THEN d.dmg END), 0) AS avg_damage_when_present
        FROM players p
        LEFT JOIN damage d
          ON d.player_id = p.id
        LEFT JOIN events e
          ON e.id = d.event_id
         AND e.bear_label = :bear_label
         AND (:start_date IS NULL OR e.event_date >= :start_date)
        GROUP BY p.id, p.name
        HAVING COALESCE(SUM(CASE WHEN e.id IS NOT NULL THEN d.dmg ELSE 0 END), 0) > 0
        ORDER BY total_damage DESC, p.name ASC;
        """
    )
    return pd.read_sql(query, engine, params={"bear_label": bear_label, "start_date": start_date})


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

    st.caption(f"Total events in window: {total_events}")
    st.dataframe(
        df[["rank", "name", "total_damage", "events_attended", "attendance_rate", "avg_damage_when_present"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "rank": st.column_config.NumberColumn("Rank"),
            "name": st.column_config.TextColumn("Player"),
            "total_damage": st.column_config.NumberColumn("Total Damage"),
            "events_attended": st.column_config.NumberColumn("Events Attended"),
            "attendance_rate": st.column_config.NumberColumn("Attendance %"),
            "avg_damage_when_present": st.column_config.NumberColumn("Avg Damage (Present)"),
        },
    )


init_db()

st.title("Leaderboards")

window = st.selectbox("Time window", ["All-time", "Last 7 days", "Last 30 days"], index=0)
window_start = start_date_for_window(window)

left_tab, right_tab = st.tabs(BEAR_LABELS)

with left_tab:
    render_trap_leaderboard(BEAR_LABELS[0], window_start)

with right_tab:
    render_trap_leaderboard(BEAR_LABELS[1], window_start)
