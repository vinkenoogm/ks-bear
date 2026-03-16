import os
import tempfile
from datetime import date

import pandas as pd
import streamlit as st
from PIL import Image

from auth import require_admin
from src.data.db import init_db
from src.data.event_image_repository import save_event_images
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
from src.services.image_storage import strato_storage_enabled, upload_event_images
from src.services.matching import (
    aggregate_damage_updates,
    build_damage_updates,
    collect_entered_names,
    diff_damage_updates,
    group_ocr_results,
    validate_review_rows,
)

st.set_page_config(page_title="Bear Data Entry", layout="wide")
st.markdown(
    """
    <style>
    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button {
        font-size: 2rem;
        min-height: 22rem;
        padding: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def extract_uploaded_results(uploaded_files) -> tuple[dict, list[str]]:
    all_ocr_results = {}
    excluded_files = []
    with st.spinner("Processing images..."):
        for uploaded_file in uploaded_files:
            suffix = os.path.splitext(uploaded_file.name)[1] or ".png"
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(uploaded_file.getbuffer())
                    temp_path = temp_file.name

                result = extract_bear_scores(temp_path)
                if result.get("excluded"):
                    excluded_files.append(uploaded_file.name)
                    continue

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
    return all_ocr_results, excluded_files


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
    img_name = unique_images[st.session_state[index_key]]
    orig_file = next((item for item in uploaded_files if item.name == img_name), None)
    if orig_file:
        orig_file.seek(0)
        preview_image = Image.open(orig_file)
        preview_image = preview_image.copy()
        preview_image.thumbnail((900, 700))
        st.caption(f"Image {current_idx + 1} of {num_images}")
        btn_col1, image_col, btn_col2 = st.columns([1, 5, 1], vertical_alignment="center")
        with btn_col1:
            if st.button("◀", key=f"prev_{trap_type}", width='stretch'):
                st.session_state[index_key] = (current_idx - 1) % num_images
                st.rerun()
        with image_col:
            _, centered_image_col, _ = st.columns([1, 6, 1])
            with centered_image_col:
                st.image(preview_image, caption=img_name)
        with btn_col2:
            if st.button("▶", key=f"next_{trap_type}", width='stretch'):
                st.session_state[index_key] = (current_idx + 1) % num_images
                st.rerun()


def rerank_review_rows(df: pd.DataFrame) -> pd.DataFrame:
    reranked_df = df.copy()
    reranked_df = reranked_df.reset_index(drop=True)
    reranked_df["rank"] = range(1, len(reranked_df) + 1)
    reranked_df["rank"] = reranked_df["rank"].astype(int)
    return reranked_df


init_db()

st.title("Data Entry")

require_admin()

st.success("You are in admin mode.")

st.subheader("Upload Screenshots")
uploaded_files = st.file_uploader("Upload screenshots of the bear trap damage rewards leaderboards. You can upload"
                                  " trap 1 and 2 at the same time, but not multiple of the same trap at once.", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files and st.button("Extract from images"):
    raw_results, excluded_files = extract_uploaded_results(uploaded_files)
    st.session_state.excluded_bear_files = excluded_files
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

if st.session_state.get("excluded_bear_files"):
    st.warning("Excluded non-bear screenshots: " + ", ".join(st.session_state["excluded_bear_files"]))

if "ocr_results_grouped" in st.session_state and st.session_state.ocr_results_grouped:
    st.write("### Extracted Scores - Verification")
    primary_names = get_primary_player_names()
    member_options = ["SELECT_MANUALLY"] + primary_names
    trap_map = {"Trap 1": "Bear Trap 1", "Trap 2": "Bear Trap 2"}

    for trap_type, df in st.session_state.ocr_results_grouped.items():
        with st.expander(f"Results for {trap_type}", expanded=True):
            st.info(
                "Review every row before saving. Choose an existing username from the dropdown, or leave "
                "`SELECT_MANUALLY` and type a new member name if the member does not exist yet. You can delete rows"
                "by clicking to the left of the row to select, then press the trash icon on the top-right of the table."
                " Every row must resolve to exactly one username, and the same player cannot appear twice in one trap."
            )

            col1, col2 = st.columns([3, 2])

            with col1:
                with st.form(f"ocr_editor_form_{trap_type}"):
                    input_col1, input_col2 = st.columns(2)
                    with input_col1:
                        trap_date = st.date_input(
                            f"Event date for {trap_type}",
                            value=date.today(),
                            key=f"date_{trap_type}",
                        )
                    with input_col2:
                        selected_trap_label = st.selectbox(
                            "Bear Trap",
                            options=["Bear Trap 1", "Bear Trap 2"],
                            index=0 if trap_map.get(trap_type, "Bear Trap 1") == "Bear Trap 1" else 1,
                            key=f"trap_label_{trap_type}",
                            help="Override the detected trap type if OCR recognized the screenshot title incorrectly.",
                        )

                    display_df = df.copy()
                    if "rank" in display_df.columns:
                        display_df["rank"] = pd.to_numeric(display_df["rank"], errors="coerce").fillna(0).astype(int)
                        display_df = display_df.sort_values(by="rank", kind="stable").reset_index(drop=True)
                    display_df["damage_display"] = display_df["damage"].fillna(0).map(lambda value: f"{int(value):,}")
                    if "new_member_name" not in display_df.columns:
                        display_df["new_member_name"] = ""

                    edited_ocr_df = st.data_editor(
                        display_df,
                        column_config={
                            "rank": st.column_config.NumberColumn("Rank", width="small", format="%d"),
                            "ocr_name": st.column_config.TextColumn("OCR Name (from image)", width="medium"),
                            "matched_name": st.column_config.SelectboxColumn(
                                "Matched Player",
                                width="medium",
                                help="Choose an existing alliance member, or leave SELECT_MANUALLY and type a new member name in the next column.",
                                options=member_options,
                            ),
                            "new_member_name": st.column_config.TextColumn(
                                "New Member Name",
                                width="medium",
                                help="Only used when Existing Player is SELECT_MANUALLY.",
                            ),
                            "damage_display": st.column_config.TextColumn("Damage Points", width=150),
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
                    save_trap_clicked = st.form_submit_button(f"Save {trap_type} Data", width='stretch', type="primary")

            with col2:
                render_image_preview(df, trap_type, uploaded_files)

            if save_trap_clicked:
                edited_ocr_df = rerank_review_rows(edited_ocr_df)
                validation_errors = validate_review_rows(edited_ocr_df)
                if validation_errors:
                    for error in validation_errors:
                        st.error(error)
                    continue

                event_id = get_or_create_event_id(event_date=trap_date, bear_label=selected_trap_label)

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

                image_names = set(df.get("source_image", pd.Series(dtype=str)).dropna().tolist())
                trap_files = [file for file in uploaded_files if file.name in image_names]
                if trap_files and strato_storage_enabled():
                    try:
                        with st.spinner("Uploading source screenshots to STRATO..."):
                            uploaded_images = upload_event_images(trap_date, selected_trap_label, trap_files)
                        save_event_images(
                            [
                                {
                                    "event_id": event_id,
                                    "bear_label": selected_trap_label,
                                    **row,
                                }
                                for row in uploaded_images
                            ]
                        )
                    except Exception as exc:
                        st.warning(f"Scores were saved, but uploading screenshots failed: {exc}")

                st.success(
                    f"Saved {saved_count} entries for {selected_trap_label} on {trap_date}. "
                    f"Skipped {skipped} duplicate score(s)."
                )
                del st.session_state.ocr_results_grouped[trap_type]
                st.rerun()
