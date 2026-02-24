from sqlalchemy import create_engine, text
from config.settings import DATABASE_URL

engine = create_engine(DATABASE_URL, future=True)

def init_db():
    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS players (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            );
            """))

            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_date DATE NOT NULL,
                bear_label TEXT NOT NULL CHECK (bear_label IN ('Trap 1', 'Trap 2')),
                UNIQUE (event_date, bear_label)
            );
            """))

            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS damage (
                event_id INTEGER REFERENCES events(id),
                player_id INTEGER REFERENCES players(id),
                dmg BIGINT NOT NULL CHECK (dmg >= 0),
                PRIMARY KEY (event_id, player_id)
            );
            """))
        else:
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS players (
                id SERIAL PRIMARY KEY,
                name TEXT UNIQUE NOT NULL
            );
            """))

            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS events (
                id SERIAL PRIMARY KEY,
                event_date DATE NOT NULL,
                bear_label TEXT NOT NULL CHECK (bear_label IN ('Trap 1', 'Trap 2')),
                UNIQUE (event_date, bear_label)
            );
            """))

            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS damage (
                event_id INTEGER REFERENCES events(id),
                player_id INTEGER REFERENCES players(id),
                dmg BIGINT NOT NULL CHECK (dmg >= 0),
                PRIMARY KEY (event_id, player_id)
            );
            """))

        conn.execute(text("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_events_unique_date_trap
        ON events (event_date, bear_label);
        """))

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_events_date_trap
        ON events (event_date, bear_label);
        """))
