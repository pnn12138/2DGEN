from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple


def list_tables(con: sqlite3.Connection) -> Dict[str, List[Tuple]]:
    """Return table names and their column info."""
    tables = {}
    cur = con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    for (table_name,) in cur.fetchall():
        cols = con.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        tables[table_name] = cols
    return tables


def distinct_keys(con: sqlite3.Connection) -> List[str]:
    """Keys stored in the ASE key_value_pairs field."""
    cur = con.execute("SELECT DISTINCT key FROM keys ORDER BY key;")
    return [row[0] for row in cur.fetchall()]


def sample_metadata(con: sqlite3.Connection) -> Dict:
    """Parse key_value_pairs for a single entry."""
    row_id = con.execute("SELECT id FROM systems LIMIT 1").fetchone()[0]
    raw = con.execute("SELECT key_value_pairs FROM systems WHERE id=?", (row_id,)).fetchone()[0]
    return json.loads(raw)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    db_path = base_dir / "c2db.db"
    con = sqlite3.connect(db_path)

    # Basic stats
    total_rows = con.execute("SELECT COUNT(*) FROM systems;").fetchone()[0]
    print(f"DB path: {db_path}")
    print(f"Total entries: {total_rows}")
    print("\nTables and columns:")
    for table_name, cols in list_tables(con).items():
        col_desc = ", ".join(f"{c[1]} ({c[2]})" for c in cols)
        print(f"- {table_name}: {col_desc}")

    # Key list stored in ASE metadata
    keys = distinct_keys(con)
    print(f"\nDistinct metadata keys ({len(keys)}):")
    print(", ".join(keys))

    # One example entry
    sample = sample_metadata(con)
    print("\nSample key_value_pairs (one entry):")
    for k in sorted(sample):
        v = sample[k]
        if isinstance(v, list):
            v_repr = f"list(len={len(v)})"
        else:
            v_repr = v
        print(f"{k}: {v_repr}")


if __name__ == "__main__":
    main()
