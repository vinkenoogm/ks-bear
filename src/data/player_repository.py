import pandas as pd
from sqlalchemy import text

from src.data.db import engine


PLAYER_TEXT_COLUMNS = [
    "name",
    "game_id",
    "alliance_rank",
    "town_center_level",
    "preferred_trap",
    "row",
]

PLAYER_INT_COLUMNS = [
    "id",
    "highest_own_rally_damage",
    "highest_total_damage",
]


def get_players_df() -> pd.DataFrame:
    df = pd.read_sql(text("SELECT * FROM players;"), engine)
    df = _normalize_player_df(df)
    return df.sort_values(
        by=["alliance_rank", "name"],
        ascending=[False, True],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)


def get_player_count() -> int:
    return int(pd.read_sql(text("SELECT COUNT(*) AS count FROM players;"), engine).iloc[0]["count"])


def get_aliases_df() -> pd.DataFrame:
    return pd.read_sql(
        text(
            """
            SELECT a.id, a.player_id, p.name AS player_name, a.alias
            FROM player_aliases a
            JOIN players p ON a.player_id = p.id
            ORDER BY p.name, a.alias;
            """
        ),
        engine,
    )


def get_player_id_to_name() -> dict[int, str]:
    players = pd.read_sql(text("SELECT id, name FROM players;"), engine)
    return players.set_index("id")["name"].to_dict()


def get_primary_player_names() -> list[str]:
    return pd.read_sql(text("SELECT name FROM players ORDER BY name;"), engine)["name"].tolist()


def get_all_players_and_aliases() -> pd.DataFrame:
    players = pd.read_sql(text("SELECT id, name FROM players;"), engine)
    aliases = pd.read_sql(text("SELECT player_id AS id, alias AS name FROM player_aliases;"), engine)
    return pd.concat([players, aliases], ignore_index=True)


def add_alias(player_id: int, alias: str):
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO player_aliases (player_id, alias)
                VALUES (:player_id, :alias)
                ON CONFLICT(alias) DO NOTHING;
                """
            ),
            {"player_id": player_id, "alias": alias},
        )


def delete_alias(alias_id: int):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM player_aliases WHERE id = :id;"), {"id": alias_id})


def add_player(
    name: str,
    game_id: str,
    alliance_rank: str = "",
    highest_own_rally_damage: int = 0,
    highest_total_damage: int = 0,
    town_center_level: str = "",
    preferred_trap: str = "",
    row: str = "",
):
    normalized_game_id = _nullable_text(game_id)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO players (
                    name,
                    game_id,
                    alliance_rank,
                    highest_own_rally_damage,
                    highest_total_damage,
                    town_center_level,
                    preferred_trap,
                    row
                )
                VALUES (
                    :name,
                    :game_id,
                    :alliance_rank,
                    :highest_own_rally_damage,
                    :highest_total_damage,
                    :town_center_level,
                    :preferred_trap,
                    :row
                )
                ON CONFLICT(name) DO UPDATE SET
                    game_id = EXCLUDED.game_id,
                    alliance_rank = EXCLUDED.alliance_rank,
                    highest_own_rally_damage = EXCLUDED.highest_own_rally_damage,
                    highest_total_damage = EXCLUDED.highest_total_damage,
                    town_center_level = EXCLUDED.town_center_level,
                    preferred_trap = EXCLUDED.preferred_trap,
                    row = EXCLUDED.row;
                """
            ),
            {
                "name": name,
                "game_id": normalized_game_id,
                "alliance_rank": alliance_rank or "",
                "highest_own_rally_damage": int(highest_own_rally_damage or 0),
                "highest_total_damage": int(highest_total_damage or 0),
                "town_center_level": town_center_level or "",
                "preferred_trap": preferred_trap or "",
                "row": row or "",
            },
        )


def update_player(
    player_id: int,
    name: str,
    game_id: str,
    alliance_rank: str = "",
    highest_own_rally_damage: int = 0,
    highest_total_damage: int = 0,
    town_center_level: str = "",
    preferred_trap: str = "",
    row: str = "",
):
    normalized_game_id = _nullable_text(game_id)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE players
                SET name = :name,
                    game_id = :game_id,
                    alliance_rank = :alliance_rank,
                    highest_own_rally_damage = :highest_own_rally_damage,
                    highest_total_damage = :highest_total_damage,
                    town_center_level = :town_center_level,
                    preferred_trap = :preferred_trap,
                    row = :row
                WHERE id = :id;
                """
            ),
            {
                "name": name,
                "game_id": normalized_game_id,
                "alliance_rank": alliance_rank or "",
                "highest_own_rally_damage": int(highest_own_rally_damage or 0),
                "highest_total_damage": int(highest_total_damage or 0),
                "town_center_level": town_center_level or "",
                "preferred_trap": preferred_trap or "",
                "row": row or "",
                "id": player_id,
            },
        )


def delete_player(player_id: int):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM damage WHERE player_id = :id;"), {"id": player_id})
        conn.execute(text("DELETE FROM players WHERE id = :id;"), {"id": player_id})


def add_missing_players(names: list[str]):
    if not names:
        return

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO players (name)
                VALUES (:name)
                ON CONFLICT(name) DO NOTHING;
                """
            ),
            [{"name": name} for name in names],
        )


def _normalize_player_df(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()

    for column in PLAYER_TEXT_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = ""
        else:
            normalized[column] = normalized[column].fillna("").astype(str)

    for column in PLAYER_INT_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = 0
        else:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce").fillna(0).astype(int)

    return normalized


def _nullable_text(value):
    if value is None:
        return None
    text_value = str(value).strip()
    return text_value or None
