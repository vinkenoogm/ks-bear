import streamlit as st

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

st.subheader("Manage Events")
orig_events_df = load_all_events()
manage_events_df = orig_events_df.copy()
manage_events_df["total_damage_display"] = manage_events_df["total_damage"].map(lambda value: f"{int(value):,}")

with st.form("manage_events_form"):
    edited_events_df = st.data_editor(
        manage_events_df.assign(delete=False),
        use_container_width=False,
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
    manage_submit = st.form_submit_button("Save Event Changes", use_container_width=True, type="primary")

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
        use_container_width=False,
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
    st.caption("Edit damage values below to correct mistakes for the selected event, then save your changes.")
    with st.form(f"event_scores_form_{selected_event_id}"):
        edited_scores_df = st.data_editor(
            df,
            use_container_width=False,
            width="content",
            column_config={
                "rank": st.column_config.NumberColumn("Rank", format="%d", help="Position in leaderboard", width="small"),
                "player_id": None,
                "player": st.column_config.TextColumn("Player", width="medium"),
                "damage": st.column_config.NumberColumn("Damage", format="%,d", min_value=0, width="medium"),
            },
            disabled=["rank", "player"],
            hide_index=True,
            num_rows="fixed",
            key=f"event_scores_editor_{selected_event_id}",
        )
        save_scores_clicked = st.form_submit_button("Save Score Changes", type="primary")

    if save_scores_clicked:
        update_event_scores(selected_event_id, edited_scores_df)
        st.success("Saved updated scores for this event.")
        st.rerun()

    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"event_scores_{selected_date}_{'trap1' if selected_trap_label.endswith('1') else 'trap2'}.csv",
        mime="text/csv",
        use_container_width=False,
    )
