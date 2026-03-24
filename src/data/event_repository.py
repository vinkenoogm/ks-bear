from datetime import date, datetime

import pandas as pd
from sqlalchemy import text

from src.data.db import engine


def get_or_create_event_id(event_date: date, bear_label: str) -> int:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO events (event_date, bear_label, total_damage, participant_count, rallies)
                VALUES (:event_date, :bear_label, 0, 0, 0)
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


def events_count_for_trap(bear_label: str, start_date):
    with engine.begin() as conn:
        return int(
            conn.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM events
                    WHERE bear_label = :bear_label
                      AND (:start_date IS NULL OR event_date >= :start_date);
                    """
                ),
                {"bear_label": bear_label, "start_date": start_date},
            ).scalar_one()
        )


def leaderboard_for_trap(bear_label: str, start_date) -> pd.DataFrame:
    return pd.read_sql(
        text(
            """
            SELECT
                p.name,
                COALESCE(SUM(CASE WHEN e.id IS NOT NULL THEN d.dmg ELSE 0 END), 0) AS total_damage,
                COALESCE(SUM(CASE WHEN e.id IS NOT NULL AND d.dmg > 0 THEN 1 ELSE 0 END), 0) AS events_attended,
                COALESCE(AVG(CASE WHEN e.id IS NOT NULL AND d.dmg > 0 THEN d.dmg END), 0) AS avg_damage_when_present
            FROM players p
            LEFT JOIN damage d
              ON d.player_id = p.id
            LEFT JOIN events e
              ON e.id = d.event_id
             AND e.bear_label = :bear_label
             AND (:start_date IS NULL OR e.event_date >= :start_date)
            GROUP BY p.id, p.name
            HAVING COALESCE(SUM(CASE WHEN e.id IS NOT NULL THEN d.dmg ELSE 0 END), 0) > 0
            ORDER BY total_damage DESC, p.name ASC;
            """
        ),
        engine,
        params={"bear_label": bear_label, "start_date": start_date},
    )


def get_event_filters():
    with engine.connect() as conn:
        dates = [row[0] for row in conn.execute(text("SELECT DISTINCT event_date FROM events ORDER BY event_date DESC")).fetchall()]
    return dates, ["Bear Trap 1", "Bear Trap 2"]


def get_event_id(event_date: str, bear_label: str):
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT id FROM events WHERE event_date = :d AND bear_label = :b"),
            {"d": event_date, "b": bear_label},
        ).fetchone()
    return row[0] if row else None


def load_scores(event_id: int) -> pd.DataFrame:
    if not event_id:
        return pd.DataFrame(columns=["rank", "player_id", "player", "damage"])

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT p.id AS player_id,
                       p.name AS player,
                       d.dmg AS damage
                FROM damage d
                JOIN players p ON p.id = d.player_id
                WHERE d.event_id = :eid
                ORDER BY d.dmg DESC, p.name ASC
                """
            ),
            {"eid": event_id},
        ).fetchall()

    df = pd.DataFrame(rows, columns=["player_id", "player", "damage"]) if rows else pd.DataFrame(columns=["player_id", "player", "damage"])
    if not df.empty:
        df.insert(0, "rank", range(1, len(df) + 1))
        df["rank"] = df["rank"].astype(int)
        df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").fillna(0).astype(int)
        df["damage"] = pd.to_numeric(df["damage"], errors="coerce").fillna(0).astype(int)
    return df


def load_all_events() -> pd.DataFrame:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT id, event_date, bear_label, total_damage, participant_count, rallies
                FROM events
                ORDER BY event_date DESC, bear_label ASC
                """
            )
        ).fetchall()

    df = pd.DataFrame(
        rows,
        columns=["id", "event_date", "bear_label", "total_damage", "participant_count", "rallies"],
    ) if rows else pd.DataFrame(columns=["id", "event_date", "bear_label", "total_damage", "participant_count", "rallies"])
    if not df.empty:
        df["event_date"] = df["event_date"].map(_to_date)
        df["total_damage"] = pd.to_numeric(df["total_damage"], errors="coerce").fillna(0).astype("int64")
        df["participant_count"] = pd.to_numeric(df["participant_count"], errors="coerce").fillna(0).astype("int64")
        df["rallies"] = pd.to_numeric(df["rallies"], errors="coerce").fillna(0).astype("int64")
    return df


def normalize_event_date(value) -> str:
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, datetime):
        return value.date().strftime("%Y-%m-%d")
    return str(value) if value is not None else None


def save_event_changes(orig_events_df: pd.DataFrame, edited_events_df: pd.DataFrame) -> tuple[int, int, int]:
    updated, deleted, skipped_dupes = 0, 0, 0
    edited_map = {int(row["id"]): row for _, row in edited_events_df.iterrows()}
    orig_map = {int(row["id"]): row for _, row in orig_events_df.iterrows()}

    with engine.begin() as conn:
        for event_id, row in edited_map.items():
            if row.get("delete"):
                conn.execute(text("DELETE FROM event_images WHERE event_id = :eid"), {"eid": event_id})
                conn.execute(text("DELETE FROM damage WHERE event_id = :eid"), {"eid": event_id})
                conn.execute(text("DELETE FROM events WHERE id = :eid"), {"eid": event_id})
                deleted += 1

        for event_id, row in edited_map.items():
            if row.get("delete") or event_id not in orig_map:
                continue

            old = orig_map[event_id]
            new_date = normalize_event_date(row.get("event_date"))
            old_date = normalize_event_date(old.get("event_date"))
            new_label = row.get("bear_label")
            old_label = old.get("bear_label")
            new_rallies = int(row.get("rallies", 0) or 0)
            old_rallies = int(old.get("rallies", 0) or 0)
            if new_date == old_date and new_label == old_label and new_rallies == old_rallies:
                continue

            dup = conn.execute(
                text("SELECT id FROM events WHERE event_date = :d AND bear_label = :b AND id <> :id"),
                {"d": new_date, "b": new_label, "id": event_id},
            ).fetchone()
            if dup:
                skipped_dupes += 1
                continue

            conn.execute(
                text("UPDATE events SET event_date = :d, bear_label = :b, rallies = :r WHERE id = :id"),
                {"d": new_date, "b": new_label, "r": new_rallies, "id": event_id},
            )
            updated += 1

    return updated, deleted, skipped_dupes


def get_existing_damage_map(event_id: int) -> dict[int, int]:
    with engine.connect() as conn:
        existing_rows = pd.read_sql(
            text("SELECT player_id, dmg FROM damage WHERE event_id = :eid"),
            conn,
            params={"eid": event_id},
        )
    if existing_rows.empty:
        return {}
    return dict(zip(existing_rows["player_id"], existing_rows["dmg"]))


def upsert_damage_rows(event_id: int, rows: list[dict[str, int]]) -> int:
    if not rows:
        return 0

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
            [{"event_id": event_id, **row} for row in rows],
        )
        _refresh_event_stats(conn, event_id)
    return len(rows)


def refresh_event_stats(event_id: int):
    with engine.begin() as conn:
        _refresh_event_stats(conn, event_id)


def update_event_scores(event_id: int, edited_scores_df: pd.DataFrame):
    rows = []
    for row in edited_scores_df.itertuples(index=False):
        player_id = int(getattr(row, "player_id"))
        damage = getattr(row, "damage", 0)
        if pd.isna(damage):
            damage = 0
        rows.append({"event_id": event_id, "player_id": player_id, "dmg": max(0, int(damage))})

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
        _refresh_event_stats(conn, event_id)


def _refresh_event_stats(conn, event_id: int):
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
                )
            WHERE id = :event_id
            """
        ),
        {"event_id": event_id},
    )


def _to_date(value):
    if isinstance(value, (date, datetime)):
        return value.date() if isinstance(value, datetime) else value
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except Exception:
        return None
