import pandas as pd
import streamlit as st

from auth import require_admin
from src.data.db import engine, init_db
from src.data.player_repository import (
    add_alias,
    add_player,
    delete_alias,
    delete_player,
    get_aliases_df,
    get_players_df,
    update_player,
)


st.title("Admin")

require_admin()
init_db()

st.write("### Alliance Member Management")
st.caption(f"Connected database: `{engine.dialect.name}`")
st.info(
    "Use Member List to maintain alliance member details. Alliance Rank, player profile fields, and IDs are editable there. "
    "The table starts sorted by Alliance Rank descending, and you can sort it further by clicking column headers. "
    "Internal IDs are hidden from the table but are still used when saving changes."
)

tab1, tab2, tab3 = st.tabs(["Member List", "Add Member", "Manage Aliases"])

with tab3:
    st.write("### Manage Player Aliases")
    st.info("Aliases help the OCR recognize players even if they change their in-game name.")

    players_df = get_players_df()
    if players_df.empty:
        st.warning("Add players first to manage aliases.")
    else:
        with st.form("add_alias_form"):
            col_p, col_a = st.columns(2)
            with col_p:
                target_player = st.selectbox("Select Player", options=players_df["name"].tolist())
            with col_a:
                new_alias = st.text_input("New Alias (e.g. old username)")

            if st.form_submit_button("Add Alias"):
                if new_alias:
                    player_id = players_df[players_df["name"] == target_player]["id"].iloc[0]
                    add_alias(int(player_id), new_alias)
                    st.success(f"Added alias '{new_alias}' for {target_player}")
                    st.rerun()
                else:
                    st.error("Alias name is required")

        st.divider()
        st.write("#### Existing Aliases")
        aliases_df = get_aliases_df()
        if aliases_df.empty:
            st.info("No aliases defined yet.")
        else:
            with st.form("alias_editor_form"):
                edited_aliases = st.data_editor(
                    aliases_df,
                    column_config={
                        "id": None,
                        "player_id": None,
                        "player_name": st.column_config.TextColumn("Player", disabled=True),
                        "alias": st.column_config.TextColumn("Alias", disabled=True),
                    },
                    num_rows="dynamic",
                    use_container_width=True,
                    key="alias_editor",
                )
                delete_aliases_clicked = st.form_submit_button("Delete Selected Aliases", type="primary")

            if delete_aliases_clicked:
                current_ids = set(aliases_df["id"])
                edited_ids = set(edited_aliases["id"].dropna())
                deleted_ids = current_ids - edited_ids

                if deleted_ids:
                    for alias_id in deleted_ids:
                        delete_alias(alias_id)
                    st.success(f"Deleted {len(deleted_ids)} aliases")
                    st.rerun()
                else:
                    st.info("No aliases were deleted. Use the trash icon in the table to remove rows before clicking delete.")

with tab2:
    st.write("### Add Single Member")
    st.caption("Only `Name` is required. All other fields are optional and can be filled in later.")
    if "new_name" not in st.session_state:
        st.session_state.new_name = ""
    if "new_game_id" not in st.session_state:
        st.session_state.new_game_id = ""
    if "new_alliance_rank" not in st.session_state:
        st.session_state.new_alliance_rank = ""
    if "new_highest_own_rally_damage" not in st.session_state:
        st.session_state.new_highest_own_rally_damage = 0
    if "new_highest_total_damage" not in st.session_state:
        st.session_state.new_highest_total_damage = 0
    if "new_town_center_level" not in st.session_state:
        st.session_state.new_town_center_level = ""
    if "new_preferred_trap" not in st.session_state:
        st.session_state.new_preferred_trap = ""
    if "new_row" not in st.session_state:
        st.session_state.new_row = ""

    with st.form("add_player_form", clear_on_submit=True):
        top_col1, top_col2, top_col3 = st.columns(3)
        with top_col1:
            new_name = st.text_input("Name *", value=st.session_state.new_name)
        with top_col2:
            new_game_id = st.text_input("Game ID (UID)", value=st.session_state.new_game_id)
        with top_col3:
            new_alliance_rank = st.text_input("Alliance Rank", value=st.session_state.new_alliance_rank)

        mid_col1, mid_col2, mid_col3 = st.columns(3)
        with mid_col1:
            new_town_center_level = st.text_input("Town Center Level", value=st.session_state.new_town_center_level)
        with mid_col2:
            new_preferred_trap = st.text_input("Preferred Trap", value=st.session_state.new_preferred_trap)
        with mid_col3:
            new_row = st.text_input("Row", value=st.session_state.new_row)

        bottom_col1, bottom_col2 = st.columns(2)
        with bottom_col1:
            new_highest_own_rally_damage = st.number_input(
                "Highest Damage In Own Rally",
                min_value=0,
                step=1,
                value=int(st.session_state.new_highest_own_rally_damage),
            )
        with bottom_col2:
            new_highest_total_damage = st.number_input(
                "Highest Total Damage",
                min_value=0,
                step=1,
                value=int(st.session_state.new_highest_total_damage),
            )
        submitted = st.form_submit_button("Add/Update Player", type="primary")
        if submitted:
            if new_name:
                add_player(
                    new_name,
                    new_game_id,
                    alliance_rank=new_alliance_rank,
                    highest_own_rally_damage=new_highest_own_rally_damage,
                    highest_total_damage=new_highest_total_damage,
                    town_center_level=new_town_center_level,
                    preferred_trap=new_preferred_trap,
                    row=new_row,
                )
                st.success(f"Successfully added/updated {new_name}")
                st.session_state.new_name = ""
                st.session_state.new_game_id = ""
                st.session_state.new_alliance_rank = ""
                st.session_state.new_highest_own_rally_damage = 0
                st.session_state.new_highest_total_damage = 0
                st.session_state.new_town_center_level = ""
                st.session_state.new_preferred_trap = ""
                st.session_state.new_row = ""
            else:
                st.error("Name is required")

    st.divider()
    st.write("### Import Members from CSV")
    st.info("CSV should have 'name' and 'game_id' columns. Existing Game IDs will be skipped.")
    with st.form("import_members_form"):
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        import_submitted = st.form_submit_button("Import CSV", type="primary")

    if import_submitted:
        if uploaded_file is None:
            st.error("Choose a CSV file first.")
        else:
            try:
                with st.spinner("Importing alliance members from CSV..."):
                    import_df = pd.read_csv(uploaded_file)
                    if "name" in import_df.columns and "game_id" in import_df.columns:
                        existing_players = get_players_df()
                        existing_gids = set(existing_players["game_id"].dropna().astype(str))

                        added_count = 0
                        skipped_count = 0
                        for _, row in import_df.iterrows():
                            name = str(row["name"]).strip()
                            gid = str(row["game_id"]).strip() if pd.notna(row["game_id"]) else ""
                            gid = gid or None
                            if gid and gid in existing_gids:
                                skipped_count += 1
                            else:
                                add_player(name, gid)
                                added_count += 1

                        if added_count > 0:
                            st.success(f"Successfully imported {added_count} members. Skipped {skipped_count} existing Game IDs.")
                            st.rerun()
                        else:
                            st.info(f"No new members added. Skipped {skipped_count} existing Game IDs.")
                    else:
                        st.error("CSV must contain 'name' and 'game_id' columns.")
            except Exception as exc:
                st.error(f"Error processing CSV: {exc}")

with tab1:
    df = get_players_df()
    if df.empty:
        st.info("No members in the alliance yet.")
    else:
        column_order = [
            "alliance_rank",
            "name",
            "game_id",
            "town_center_level",
            "preferred_trap",
            "row",
            "highest_own_rally_damage",
            "highest_total_damage",
            "id",
        ]
        with st.form("member_list_form"):
            edited_df = st.data_editor(
                df,
                column_order=column_order,
                column_config={
                    "id": None,
                    "alliance_rank": st.column_config.TextColumn("Alliance Rank", width="medium"),
                    "name": st.column_config.TextColumn("Player Name", required=True, width="medium"),
                    "game_id": st.column_config.TextColumn("Game ID (UID)", width="medium"),
                    "town_center_level": st.column_config.TextColumn("Town Center Level", width="medium"),
                    "preferred_trap": st.column_config.TextColumn("Preferred Trap", width="medium"),
                    "row": st.column_config.TextColumn("Row", width="small"),
                    "highest_own_rally_damage": st.column_config.NumberColumn("Highest Damage In Own Rally", format="%d", min_value=0, width="large"),
                    "highest_total_damage": st.column_config.NumberColumn("Highest Total Damage", format="%d", min_value=0, width="large"),
                },
                num_rows="dynamic",
                width="content",
                key="member_editor",
            )
            save_clicked = st.form_submit_button("Save Changes", type="primary")

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Export to CSV",
            data=csv_data,
            file_name="alliance_members.csv",
            mime="text/csv",
        )

        if save_clicked:
            current_ids = set(df["id"])
            edited_ids = set(edited_df["id"].dropna())
            deleted_ids = current_ids - edited_ids

            for player_id in deleted_ids:
                delete_player(player_id)

            for _, row in edited_df.iterrows():
                if pd.isna(row["id"]):
                    if row["name"]:
                        add_player(
                            row["name"],
                            row["game_id"],
                            alliance_rank=row.get("alliance_rank", ""),
                            highest_own_rally_damage=row.get("highest_own_rally_damage", 0),
                            highest_total_damage=row.get("highest_total_damage", 0),
                            town_center_level=row.get("town_center_level", ""),
                            preferred_trap=row.get("preferred_trap", ""),
                            row=row.get("row", ""),
                        )
                else:
                    update_player(
                        int(row["id"]),
                        row["name"],
                        row["game_id"],
                        alliance_rank=row.get("alliance_rank", ""),
                        highest_own_rally_damage=row.get("highest_own_rally_damage", 0),
                        highest_total_damage=row.get("highest_total_damage", 0),
                        town_center_level=row.get("town_center_level", ""),
                        preferred_trap=row.get("preferred_trap", ""),
                        row=row.get("row", ""),
                    )

            st.success("Changes saved!")
            st.rerun()

st.divider()
st.write("Admin-only configuration panel.")
