from pathlib import Path
import sys

from sqlalchemy import text


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.db import engine


def main():
    with engine.begin() as conn:
        columns = _player_columns(conn)
        if "alliance_rank" not in columns:
            conn.execute(text("ALTER TABLE players ADD COLUMN alliance_rank TEXT NOT NULL DEFAULT ''"))
        if "highest_own_rally_damage" not in columns:
            conn.execute(text("ALTER TABLE players ADD COLUMN highest_own_rally_damage INTEGER NOT NULL DEFAULT 0"))
        if "highest_total_damage" not in columns:
            conn.execute(text("ALTER TABLE players ADD COLUMN highest_total_damage INTEGER NOT NULL DEFAULT 0"))
        if "town_center_level" not in columns:
            conn.execute(text("ALTER TABLE players ADD COLUMN town_center_level TEXT NOT NULL DEFAULT ''"))
        if "preferred_trap" not in columns:
            conn.execute(text("ALTER TABLE players ADD COLUMN preferred_trap TEXT NOT NULL DEFAULT ''"))
        if "row" not in columns:
            conn.execute(text("ALTER TABLE players ADD COLUMN row TEXT NOT NULL DEFAULT ''"))

        conn.execute(
            text(
                """
                UPDATE players
                SET alliance_rank = COALESCE(alliance_rank, ''),
                    highest_own_rally_damage = COALESCE(highest_own_rally_damage, 0),
                    highest_total_damage = COALESCE(highest_total_damage, 0),
                    town_center_level = COALESCE(town_center_level, ''),
                    preferred_trap = COALESCE(preferred_trap, ''),
                    row = COALESCE(row, '')
                """
            )
        )

    print("Player profile fields migrated successfully.")


def _player_columns(conn) -> set[str]:
    if engine.dialect.name == "sqlite":
        rows = conn.execute(text("PRAGMA table_info(players)")).fetchall()
        return {row[1] for row in rows}

    rows = conn.execute(
        text(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'players'
            """
        )
    ).fetchall()
    return {row[0] for row in rows}


if __name__ == "__main__":
    main()
