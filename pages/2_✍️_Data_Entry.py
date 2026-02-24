import streamlit as st
import pandas as pd
from datetime import date
from sqlalchemy import text

from auth import require_admin
from config.settings import BEAR_LABELS
from db import engine, init_db


def add_players(single_name: str, bulk_names: str) -> int:
    names = []
    if single_name.strip():
        names.append(single_name.strip())

    for line in bulk_names.splitlines():
        cleaned = line.strip()
        if cleaned:
            names.append(cleaned)

    unique_names = sorted(set(names))
    if not unique_names:
        return 0

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO players (name)
                VALUES (:name)
                ON CONFLICT (name) DO NOTHING;
                """
            ),
            [{"name": n} for n in unique_names],
        )

    return len(unique_names)


def get_or_create_event_id(event_date: date, bear_label: str) -> int:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO events (event_date, bear_label)
                VALUES (:event_date, :bear_label)
                ON CONFLICT (event_date, bear_label) DO NOTHING;
                """
            ),
            {"event_date": event_date, "bear_label": bear_label},
        )

        event_id = conn.execute(
            text(
                """
                SELECT id
                FROM events
                WHERE event_date = :event_date AND bear_label = :bear_label;
                """
            ),
            {"event_date": event_date, "bear_label": bear_label},
        ).scalar_one()

    return int(event_id)


def load_damage_sheet(event_id: int) -> pd.DataFrame:
    return pd.read_sql(
        text(
            """
            SELECT p.id AS player_id, p.name, COALESCE(d.dmg, 0) AS damage
            FROM players p
            LEFT JOIN damage d
              ON d.player_id = p.id
             AND d.event_id = :event_id
            ORDER BY p.name;
            """
        ),
        engine,
        params={"event_id": event_id},
    )


def save_damage(event_id: int, edited_df: pd.DataFrame) -> int:
    save_df = edited_df.copy()
    save_df["damage"] = pd.to_numeric(save_df["damage"], errors="coerce").fillna(0).clip(lower=0)
    save_df["damage"] = save_df["damage"].astype("int64")

    rows = [
        {"event_id": event_id, "player_id": int(r.player_id), "dmg": int(r.damage)}
        for r in save_df.itertuples(index=False)
    ]

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO damage (event_id, player_id, dmg)
                VALUES (:event_id, :player_id, :dmg)
                ON CONFLICT (event_id, player_id)
                DO UPDATE SET dmg = EXCLUDED.dmg;
                """
            ),
            rows,
        )

    attendance = int((save_df["damage"] > 0).sum())
    return attendance


init_db()

st.title("Data Entry")

require_admin()

st.success("You are in admin mode.")

with st.expander("Add players", expanded=False):
    single_name = st.text_input("Add one player")
    bulk_names = st.text_area("Bulk add (one name per line)", height=140)
    if st.button("Save players"):
        inserted_count = add_players(single_name=single_name, bulk_names=bulk_names)
        if inserted_count == 0:
            st.warning("No player names provided.")
        else:
            st.success(f"Processed {inserted_count} unique player names.")
            st.rerun()

players_count = pd.read_sql(text("SELECT COUNT(*) AS count FROM players;"), engine).iloc[0]["count"]
if int(players_count) == 0:
    st.warning("Add players first to start entering event damage.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    selected_date = st.date_input("Event date", value=date.today())
with col2:
    selected_trap = st.selectbox("Bear trap", options=BEAR_LABELS)

event_id = get_or_create_event_id(event_date=selected_date, bear_label=selected_trap)
st.caption(f"Editing event #{event_id} for {selected_trap} on {selected_date}")

damage_df = load_damage_sheet(event_id=event_id)
edited_df = st.data_editor(
    damage_df,
    use_container_width=True,
    hide_index=True,
    disabled=["player_id", "name"],
    column_config={
        "player_id": st.column_config.NumberColumn("Player ID"),
        "name": st.column_config.TextColumn("Player"),
        "damage": st.column_config.NumberColumn("Damage", min_value=0, step=1),
    },
)

if st.button("Save damage", type="primary"):
    attendance = save_damage(event_id=event_id, edited_df=edited_df)
    st.success(f"Saved damage for {len(edited_df)} players. Attendance: {attendance}.")
