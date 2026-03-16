from sqlalchemy import text

from src.data.db import engine


def save_event_images(rows: list[dict]):
    if not rows:
        return

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO event_images (
                    event_id,
                    bear_label,
                    original_filename,
                    storage_path,
                    public_url
                )
                VALUES (
                    :event_id,
                    :bear_label,
                    :original_filename,
                    :storage_path,
                    :public_url
                );
                """
            ),
            rows,
        )
