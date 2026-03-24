-- SQLite/PostgreSQL schema definitions
CREATE TABLE IF NOT EXISTS players (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    game_id TEXT UNIQUE,
    alliance_rank TEXT NOT NULL DEFAULT '',
    highest_own_rally_damage INTEGER NOT NULL DEFAULT 0,
    highest_total_damage INTEGER NOT NULL DEFAULT 0,
    town_center_level TEXT NOT NULL DEFAULT '',
    preferred_trap TEXT NOT NULL DEFAULT '',
    row TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS player_aliases (
    id SERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES players(id) ON DELETE CASCADE,
    alias TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id SERIAL PRIMARY KEY,
    event_date DATE NOT NULL,
    bear_label TEXT NOT NULL CHECK (bear_label IN ('Bear Trap 1', 'Bear Trap 2')),
    total_damage BIGINT NOT NULL DEFAULT 0,
    participant_count INTEGER NOT NULL DEFAULT 0,
    rallies INTEGER NOT NULL DEFAULT 0,
    UNIQUE (event_date, bear_label)
);

CREATE TABLE IF NOT EXISTS damage (
    event_id INTEGER REFERENCES events(id) ON DELETE CASCADE,
    player_id INTEGER REFERENCES players(id),
    dmg BIGINT NOT NULL CHECK (dmg >= 0),
    PRIMARY KEY (event_id, player_id)
);

CREATE TABLE IF NOT EXISTS event_images (
    id SERIAL PRIMARY KEY,
    event_id INTEGER REFERENCES events(id) ON DELETE CASCADE,
    bear_label TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    public_url TEXT,
    uploaded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
