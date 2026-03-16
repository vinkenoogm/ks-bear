from pathlib import Path

from sqlalchemy import create_engine, text

from config.settings import DATABASE_URL


engine = create_engine(DATABASE_URL, future=True)


def init_db():
    schema_path = Path(__file__).resolve().parents[2] / "config" / "schema.sql"
    if not schema_path.exists():
        return

    schema_sql = schema_path.read_text(encoding="utf-8")

    if engine.dialect.name == "sqlite":
        schema_sql = schema_sql.replace("SERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
        schema_sql = schema_sql.replace("SERIAL", "INTEGER")

    statements = []
    for line in schema_sql.splitlines():
        line = line.strip()
        if line and not line.startswith("--"):
            statements.append(line)

    with engine.begin() as conn:
        for statement in " ".join(statements).split(";"):
            statement = statement.strip()
            if not statement:
                continue
            try:
                conn.execute(text(statement))
            except Exception as exc:
                if "already exists" not in str(exc).lower():
                    print(f"Error executing statement: {statement}\n{exc}")
