from pathlib import Path
import sys

from sqlalchemy import text


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.db import engine


def main():
    with engine.begin() as conn:
        columns = _event_columns(conn)
        if "total_damage" not in columns:
            conn.execute(text("ALTER TABLE events ADD COLUMN total_damage BIGINT NOT NULL DEFAULT 0"))
        if "participant_count" not in columns:
            conn.execute(text("ALTER TABLE events ADD COLUMN participant_count INTEGER NOT NULL DEFAULT 0"))
        if "rallies" not in columns:
            conn.execute(text("ALTER TABLE events ADD COLUMN rallies INTEGER NOT NULL DEFAULT 0"))

        conn.execute(
            text(
                """
                UPDATE events
                SET total_damage = COALESCE(
                        (SELECT SUM(dmg) FROM damage WHERE damage.event_id = events.id),
                        0
                    ),
                    participant_count = COALESCE(
                        (SELECT COUNT(*) FROM damage WHERE damage.event_id = events.id AND damage.dmg > 0),
                        0
                    ),
                    rallies = COALESCE(rallies, 0)
                """
            )
        )

    print("Event summary fields migrated successfully.")


def _event_columns(conn) -> set[str]:
    if engine.dialect.name == "sqlite":
        rows = conn.execute(text("PRAGMA table_info(events)")).fetchall()
        return {row[1] for row in rows}

    rows = conn.execute(
        text(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'events'
            """
        )
    ).fetchall()
    return {row[0] for row in rows}


if __name__ == "__main__":
    main()
