import os
import tempfile
from datetime import date

import pandas as pd
import streamlit as st
from PIL import Image

from auth import require_admin
from src.data.db import init_db
from src.data.event_repository import (
    get_existing_damage_map,
    get_or_create_event_id,
    upsert_damage_rows,
)
from src.data.player_repository import (
    add_missing_players,
    get_all_players_and_aliases,
    get_player_id_to_name,
    get_players_df,
    get_primary_player_names,
)
from src.services.bear_ocr import extract_bear_scores
from src.services.matching import (
    aggregate_damage_updates,
    build_damage_updates,
    collect_entered_names,
    diff_damage_updates,
    group_ocr_results,
)

st.set_page_config(page_title="Bear Data Entry", layout="wide")


def extract_uploaded_results(uploaded_files) -> dict:
    all_ocr_results = {}
    with st.spinner("Processing images..."):
        for uploaded_file in uploaded_files:
            suffix = os.path.splitext(uploaded_file.name)[1] or ".png"
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(uploaded_file.getbuffer())
                    temp_path = temp_file.name

                result = extract_bear_scores(temp_path)
                trap = result.get("trap", "Unknown")
                all_ocr_results.setdefault(trap, [])
                for score in result["scores"]:
                    all_ocr_results[trap].append(
                        {
                            "ocr_name": score["username"],
                            "damage": score["damage"],
                            "source_image": uploaded_file.name,
                        }
                    )
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
    return all_ocr_results


def render_image_preview(df: pd.DataFrame, trap_type: str, uploaded_files):
    if "source_image" not in df.columns:
        return

    unique_images = df["source_image"].unique()
    num_images = len(unique_images)
    if num_images == 0:
        return

    index_key = f"img_idx_{trap_type}"
    if index_key not in st.session_state:
        st.session_state[index_key] = 0
    if st.session_state[index_key] >= num_images:
        st.session_state[index_key] = 0

    current_idx = st.session_state[index_key]
    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col1:
        if st.button("<", key=f"prev_{trap_type}"):
            st.session_state[index_key] = (current_idx - 1) % num_images
            st.rerun()
    with btn_col2:
        st.write(f"Image {current_idx + 1} of {num_images}")
    with btn_col3:
        if st.button(">", key=f"next_{trap_type}"):
            st.session_state[index_key] = (current_idx + 1) % num_images
            st.rerun()

    img_name = unique_images[st.session_state[index_key]]
    orig_file = next((item for item in uploaded_files if item.name == img_name), None)
    if orig_file:
        orig_file.seek(0)
        preview_image = Image.open(orig_file)
        preview_image = preview_image.copy()
        preview_image.thumbnail((900, 700))
        st.image(preview_image, caption=img_name)


init_db()

st.title("Data Entry")

require_admin()

st.success("You are in admin mode.")

st.subheader("OCR Upload (Bear Trap Screenshots)")
uploaded_files = st.file_uploader("Upload Trap screenshots", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files and st.button("Extract from images"):
    raw_results = extract_uploaded_results(uploaded_files)
    if raw_results:
        grouped_results, total_dupes_removed = group_ocr_results(
            raw_results=raw_results,
            id_to_name=get_player_id_to_name(),
            combined_df=get_all_players_and_aliases(),
        )
        st.session_state.ocr_results_grouped = grouped_results
        if total_dupes_removed > 0:
            st.success(f"Extracted entries from {len(uploaded_files)} images. Skipped {total_dupes_removed} duplicate row(s) before review.")
        else:
            st.success(f"Extracted entries from {len(uploaded_files)} images.")
        st.rerun()

if "ocr_results_grouped" in st.session_state and st.session_state.ocr_results_grouped:
    st.write("### Extracted Scores - Verification")
    primary_names = get_primary_player_names()
    member_options = ["SELECT_MANUALLY"] + primary_names
    trap_map = {"Trap 1": "Bear Trap 1", "Trap 2": "Bear Trap 2"}

    for trap_type, df in st.session_state.ocr_results_grouped.items():
        with st.expander(f"Results for {trap_type}", expanded=True):
            col1, col2 = st.columns([3, 2])

            with col1:
                trap_date = st.date_input(f"Event date for {trap_type}", value=date.today(), key=f"date_{trap_type}")
                db_trap_label = trap_map.get(trap_type, "Bear Trap 1")

                display_df = df.copy()
                if "rank" in display_df.columns:
                    display_df["rank"] = pd.to_numeric(display_df["rank"], errors="coerce").fillna(0).astype(int)
                display_df["damage_display"] = display_df["damage"].fillna(0).map(lambda value: f"{int(value):,}")
                if "new_member_name" not in display_df.columns:
                    display_df["new_member_name"] = ""

                edited_ocr_df = st.data_editor(
                    display_df,
                    column_config={
                        "rank": st.column_config.NumberColumn("Rank", width="small", format="%d"),
                        "ocr_name": st.column_config.TextColumn("OCR Name (from image)", width="medium"),
                        "matched_name": st.column_config.SelectboxColumn(
                            "Existing Player",
                            width="medium",
                            help="Choose an existing alliance member, or leave SELECT_MANUALLY and type a new member name in the next column.",
                            options=member_options,
                        ),
                        "new_member_name": st.column_config.TextColumn(
                            "New Member Name",
                            width="medium",
                            help="Only used when Existing Player is SELECT_MANUALLY.",
                        ),
                        "damage_display": st.column_config.TextColumn("Damage", width="medium"),
                        "damage": None,
                        "source_image": None,
                        "_name_key": None,
                    },
                    hide_index=True,
                    width='stretch',
                    height="content",
                    num_rows="dynamic",
                    key=f"editor_{trap_type}",
                )

            with col2:
                render_image_preview(df, trap_type, uploaded_files)

            if st.button(f"Save {trap_type} Data", key=f"save_{trap_type}", width="stretch"):
                event_id = get_or_create_event_id(event_date=trap_date, bear_label=db_trap_label)

                entered_names = collect_entered_names(edited_ocr_df)
                new_names = sorted(name for name in entered_names if name not in primary_names)
                if new_names:
                    add_missing_players(new_names)
                    primary_names = get_primary_player_names()

                current_players = get_players_df()[["id", "name"]]
                name_to_id = dict(zip(current_players["name"], current_players["id"]))

                updates = build_damage_updates(edited_ocr_df, name_to_id)
                if not updates:
                    st.warning("No valid player matches selected.")
                    continue

                aggregated_updates = aggregate_damage_updates(updates)
                if not aggregated_updates:
                    st.warning("No valid entries to save after internal deduplication.")
                    continue

                to_upsert, skipped = diff_damage_updates(aggregated_updates, get_existing_damage_map(event_id))
                saved_count = upsert_damage_rows(event_id, to_upsert)

                st.success(f"Saved {saved_count} entries for {trap_type} on {trap_date}. Skipped {skipped} duplicate score(s).")
                del st.session_state.ocr_results_grouped[trap_type]
                st.rerun()
