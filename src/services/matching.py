import re

import pandas as pd
from fuzzywuzzy import process


def fuzzy_match_username(ocr_name: str, combined_df: pd.DataFrame):
    if not ocr_name or ocr_name == "CHECK_USERNAME_MANUALLY" or combined_df.empty:
        return None

    best_match = process.extractOne(ocr_name, combined_df["name"].tolist())
    if not best_match:
        return None

    best_match_str, score = best_match
    if score <= 75:
        return None

    matched_row = combined_df[combined_df["name"] == best_match_str].iloc[0]
    return matched_row["id"]


def group_ocr_results(raw_results: dict, id_to_name: dict[int, str], combined_df: pd.DataFrame) -> tuple[dict, int]:
    final_grouped_results = {}
    total_dupes_removed = 0

    for trap, scores in raw_results.items():
        processed_scores = []
        for item in scores:
            matched_id = fuzzy_match_username(item["ocr_name"], combined_df)
            updated_item = item.copy()
            updated_item["matched_name"] = id_to_name[matched_id] if matched_id in id_to_name else "SELECT_MANUALLY"
            processed_scores.append(updated_item)

        if not processed_scores:
            continue

        df_trap = pd.DataFrame(processed_scores)
        df_trap["damage"] = pd.to_numeric(df_trap["damage"], errors="coerce").fillna(0).astype("int64")
        df_trap["_name_key"] = df_trap.apply(_normalized_name_key, axis=1)

        before = len(df_trap)
        df_trap = df_trap.drop_duplicates(subset=["_name_key", "damage"], keep="first").reset_index(drop=True)
        total_dupes_removed += max(0, before - len(df_trap))
        df_trap.insert(0, "rank", list(range(1, len(df_trap) + 1)))
        final_grouped_results[trap] = df_trap

    return final_grouped_results, total_dupes_removed


def collect_entered_names(edited_df: pd.DataFrame) -> set[str]:
    names = {
        str(name).strip()
        for name in edited_df.get("matched_name", pd.Series(dtype=str)).tolist()
        if pd.notna(name) and str(name).strip() and str(name).strip() != "SELECT_MANUALLY"
    }
    names.update(
        str(name).strip()
        for name in edited_df.get("new_member_name", pd.Series(dtype=str)).tolist()
        if pd.notna(name) and str(name).strip()
    )
    return names


def build_damage_updates(edited_df: pd.DataFrame, name_to_id: dict[str, int]) -> list[dict[str, int]]:
    updates = []
    for row in edited_df.itertuples():
        matched_name = getattr(row, "matched_name", None)
        new_member_name = getattr(row, "new_member_name", None)
        if matched_name == "SELECT_MANUALLY" and new_member_name and str(new_member_name).strip():
            matched_name = str(new_member_name).strip()

        if not matched_name or not str(matched_name).strip() or matched_name == "SELECT_MANUALLY":
            continue

        player_id = name_to_id.get(str(matched_name).strip())
        if not player_id:
            continue

        final_damage = getattr(row, "damage", 0)
        if (pd.isna(final_damage) or final_damage == 0) and hasattr(row, "damage_display"):
            try:
                final_damage = int(re.sub(r"[^0-9]", "", str(row.damage_display)))
            except Exception:
                final_damage = 0

        if final_damage > 0:
            updates.append({"player_id": int(player_id), "damage": int(final_damage)})
    return updates


def aggregate_damage_updates(updates: list[dict[str, int]]) -> dict[int, int]:
    aggregated = {}
    for update in updates:
        player_id = int(update["player_id"])
        damage = int(update["damage"])
        current = aggregated.get(player_id)
        if current is None or damage > current:
            aggregated[player_id] = damage
    return aggregated


def diff_damage_updates(aggregated_updates: dict[int, int], existing_map: dict[int, int]) -> tuple[list[dict[str, int]], int]:
    to_upsert = []
    skipped = 0
    for player_id, damage in aggregated_updates.items():
        if existing_map.get(player_id) == damage:
            skipped += 1
        else:
            to_upsert.append({"player_id": player_id, "dmg": damage})
    return to_upsert, skipped


def _normalized_name_key(row) -> str:
    name = row.get("matched_name")
    if name and str(name).strip() and str(name) != "SELECT_MANUALLY":
        base = str(name)
    else:
        base = str(row.get("ocr_name", ""))
    return re.sub(r"\s+", " ", base.strip()).lower()
