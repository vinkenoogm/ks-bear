import pandas as pd
import streamlit as st
from auth import is_admin_logged_in, render_admin_login

from src.data.player_repository import add_missing_players, get_players_df
from src.data.event_repository import (
    load_all_events,
    load_scores,
    save_event_changes,
    update_event_scores,
)


st.set_page_config(page_title="Event Scores", layout="wide")

st.title("View Bear Event Scores")
st.caption("Manage events and view saved scores.")
st.info(
    "Use Manage Events to edit event date, trap number, and number of rallies. "
    "Total damage and player count are calculated automatically and cannot be edited. "
    "Use Choose Event filters to find an event, then click a row to view its results below. "
    "In the results section, damage values are editable so you can fix saved score mistakes."
)


def prepare_score_editor_df(df):
    editor_df = df.copy()
    if "player" in editor_df.columns:
        editor_df["player_name"] = editor_df["player"].fillna("").astype(str)
    if "damage" in editor_df.columns:
        editor_df["damage"] = pd.to_numeric(editor_df["damage"], errors="coerce").fillna(0).astype("int64")
        editor_df["damage_display"] = editor_df["damage"].map(lambda value: f"{int(value):,}")
    return editor_df


def parse_score_editor_df(edited_scores_df):
    parsed_df = edited_scores_df.copy()
    parsed_df["player_name"] = parsed_df.get("player_name", pd.Series(dtype=str)).fillna("").astype(str).str.strip()
    blank_players = parsed_df["player_name"].eq("")
    if blank_players.any():
        invalid_rows = ", ".join(str(index + 1) for index in parsed_df.index[blank_players])
        raise ValueError(f"Player name is required. Invalid row(s): {invalid_rows}")

    duplicate_players = parsed_df["player_name"].str.casefold().duplicated(keep=False)
    if duplicate_players.any():
        invalid_rows = ", ".join(str(index + 1) for index in parsed_df.index[duplicate_players])
        raise ValueError(f"Player names must be unique within the event. Duplicate row(s): {invalid_rows}")

    raw_damage = (
        parsed_df.get("damage_display", pd.Series(dtype=str))
        .fillna("0")
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    invalid_damage = raw_damage.eq("") | ~raw_damage.str.fullmatch(r"\d+")
    if invalid_damage.any():
        invalid_rows = ", ".join(str(index + 1) for index in parsed_df.index[invalid_damage])
        raise ValueError(f"Damage must be a whole number. Invalid row(s): {invalid_rows}")

    parsed_df["damage"] = raw_damage.astype("int64")
    return parsed_df.drop(columns=["damage_display", "player"], errors="ignore")


def resolve_score_editor_players(parsed_scores_df):
    resolved_df = parsed_scores_df.copy()
    existing_players = get_players_df()[["id", "name"]]
    existing_names = set(existing_players["name"].tolist())
    new_names = sorted(name for name in resolved_df["player_name"].tolist() if name not in existing_names)
    if new_names:
        add_missing_players(new_names)
        existing_players = get_players_df()[["id", "name"]]

    name_to_id = dict(zip(existing_players["name"], existing_players["id"]))
    resolved_df["player_id"] = resolved_df["player_name"].map(name_to_id)
    unresolved = resolved_df["player_id"].isna()
    if unresolved.any():
        invalid_rows = ", ".join(str(index + 1) for index in resolved_df.index[unresolved])
        raise ValueError(f"Unable to resolve player names. Invalid row(s): {invalid_rows}")

    resolved_df["player_id"] = resolved_df["player_id"].astype(int)
    return resolved_df[["player_id", "damage"]]

st.subheader("Manage Events")
orig_events_df = load_all_events()
if is_admin_logged_in():
    manage_events_df = orig_events_df.copy()
    manage_events_df["total_damage_display"] = manage_events_df["total_damage"].map(lambda value: f"{int(value):,}")

    with st.form("manage_events_form"):
        edited_events_df = st.data_editor(
            manage_events_df.assign(delete=False),
            width="content",
            hide_index=True,
            column_config={
                "id": st.column_config.NumberColumn("ID", format="%d", help="Event ID", width="small"),
                "event_date": st.column_config.DateColumn("Date", help="Event date (YYYY-MM-DD)", width="medium"),
                "bear_label": st.column_config.SelectboxColumn("Trap", options=["Bear Trap 1", "Bear Trap 2"], help="Trap type", width="medium"),
                "total_damage_display": st.column_config.TextColumn("Total Damage", width="medium"),
                "participant_count": st.column_config.NumberColumn("Players", format="%d", width="small"),
                "rallies": st.column_config.NumberColumn("Rallies", format="%d", width="small"),
                "delete": st.column_config.CheckboxColumn("Delete", help="Mark to delete this event and its scores", width="small"),
                "total_damage": None,
            },
            disabled=["id", "total_damage_display", "participant_count"],
            num_rows="fixed",
        )
        manage_submit = st.form_submit_button("Save Event Changes", width="stretch", type="primary")

    if manage_submit:
        try:
            updated, deleted, skipped_dupes = save_event_changes(orig_events_df, edited_events_df)
            msg = f"Updated {updated} | Deleted {deleted}"
            if skipped_dupes:
                msg += f" | Skipped {skipped_dupes} duplicate change(s)"
            st.success(msg)
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to save changes: {exc}")
else:
    render_admin_login("manage_events")

if orig_events_df.empty:
    st.info("No events found. Add data on the Data Entry page first.")
    st.stop()

st.subheader("Choose Event")
table_col, filter_col = st.columns([3, 1])

available_dates = sorted(orig_events_df["event_date"].dropna().unique(), reverse=True)
with filter_col:
    selected_date_filter = st.selectbox(
        "Event Date",
        options=["All"] + list(available_dates),
        index=0,
        format_func=lambda value: value if value == "All" else str(value),
    )
    trap_filter = st.selectbox("Trap Filter", ["All", "Bear Trap 1", "Bear Trap 2"], index=0)
    participants_filter = st.selectbox("Participants", ["All", "Has participants", "No participants"], index=0)

selectable_events_df = orig_events_df.copy()
if selected_date_filter != "All":
    selectable_events_df = selectable_events_df[selectable_events_df["event_date"] == selected_date_filter]
if trap_filter != "All":
    selectable_events_df = selectable_events_df[selectable_events_df["bear_label"] == trap_filter]
if participants_filter == "Has participants":
    selectable_events_df = selectable_events_df[selectable_events_df["participant_count"] > 0]
elif participants_filter == "No participants":
    selectable_events_df = selectable_events_df[selectable_events_df["participant_count"] == 0]

if selectable_events_df.empty:
    st.info("No events match the current filters.")
    st.stop()

selectable_events_df = selectable_events_df.copy()
selectable_events_df["total_damage_display"] = selectable_events_df["total_damage"].map(lambda value: f"{int(value):,}")
with table_col:
    event_selection = st.dataframe(
        selectable_events_df,
        width="content",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "id": st.column_config.NumberColumn("ID", format="%d", width="small"),
            "event_date": st.column_config.DateColumn("Date", width="medium"),
            "bear_label": st.column_config.TextColumn("Trap", width="medium"),
            "total_damage_display": st.column_config.TextColumn("Total Damage", width="medium"),
            "participant_count": st.column_config.NumberColumn("Players", format="%d", width="small"),
            "rallies": st.column_config.NumberColumn("Rallies", format="%d", width="small"),
            "total_damage": None,
        },
        key="event_selector",
    )

selected_rows = event_selection.selection.rows
selected_index = selected_rows[0] if selected_rows else 0
selected_event = selectable_events_df.iloc[selected_index]
selected_event_id = int(selected_event["id"])
selected_date = selected_event["event_date"]
selected_trap_label = selected_event["bear_label"]

df = load_scores(selected_event_id)


st.subheader(f"Results: {selected_trap_label} on {selected_date}")

if df.empty:
    st.info("No scores for the selected event.")
else:
    editor_df = prepare_score_editor_df(df)
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"event_scores_{selected_date}_{'trap1' if selected_trap_label.endswith('1') else 'trap2'}.csv",
        mime="text/csv",
        width="content",
    )

    if is_admin_logged_in():
        st.caption("Edit rows below to correct mistakes or add players for the selected event, then save your changes.")
        with st.form(f"event_scores_form_{selected_event_id}"):
            edited_scores_df = st.data_editor(
                editor_df,
                width="content",
                column_config={
                    "rank": st.column_config.NumberColumn("Rank", format="%d", help="Position in leaderboard", width="small"),
                    "player_id": None,
                    "player": None,
                    "player_name": st.column_config.TextColumn(
                        "Player",
                        width="medium",
                        help="Editable. Existing player names are recommended; a new player will be created if needed.",
                    ),
                    "damage_display": st.column_config.TextColumn("Damage", width="medium", help="Editable. Use digits; commas are optional."),
                    "damage": None,
                },
                disabled=["rank", "player_id"],
                hide_index=True,
                num_rows="dynamic",
                key=f"event_scores_editor_{selected_event_id}",
            )
            save_scores_clicked = st.form_submit_button("Save Score Changes", type="primary")

        if save_scores_clicked:
            try:
                parsed_scores_df = parse_score_editor_df(edited_scores_df)
                resolved_scores_df = resolve_score_editor_players(parsed_scores_df)
                update_event_scores(selected_event_id, resolved_scores_df)
                st.success("Saved updated scores for this event.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    else:
        st.caption("Log in as admin to edit saved scores.")
        st.dataframe(
            editor_df.drop(columns=["damage"], errors="ignore"),
            width="content",
            hide_index=True,
            column_config={
                "rank": st.column_config.NumberColumn("Rank", format="%d", help="Position in leaderboard", width="small"),
                "player_id": None,
                "player": st.column_config.TextColumn("Player", width="medium"),
                "damage_display": st.column_config.TextColumn("Damage", width="medium"),
            },
        )
