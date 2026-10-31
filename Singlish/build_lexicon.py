"""Build the Singlish lexicon vector DB (LanceDB) from a JSON file.

Reads lexicon_singlish.json (or a path given on the command line), computes a
semantic embedding and a phonetic key for each term, and writes a LanceDB table
that service.py / lexicon_lookup.py can query.

Expected JSON schema (either a bare list of entries, or an object with an
"entries" list):

    {
      "schema_version": 1,
      "entries": [
        {
          "term": "shiok",            # required: canonical spelling
          "type": "term",             # required: "particle" | "term" | "acronym"
          "meaning": "great ...",     # required: short gloss (used for embedding)
          "expansion": "",            # optional: full form for acronyms
          "variants": ["shook"]       # optional: common mis-hearings / spellings
        }
      ]
    }

Each entry becomes one canonical row. Every variant becomes an extra row that
shares the canonical term/type/meaning/expansion but carries its own phonetic
key, so a mis-heard token still resolves to the canonical term.

Usage:
    python build_lexicon.py                              # overwrite (default)
    python build_lexicon.py singlish_lexicon.json        # custom input
    python build_lexicon.py singlish_lexicon.json --append   # add to existing table
    LEXICON_JSON=... LEXICON_DB=... python build_lexicon.py

Write modes:
    overwrite (default) - (re)create the table from scratch.
    append              - add rows to the existing table, skipping any row whose
                          id already exists (safe to re-run; no duplicates). If
                          the table does not exist yet, it is created.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import lancedb
from metaphone import doublemetaphone
from sentence_transformers import SentenceTransformer

DEFAULT_JSON = os.environ.get("LEXICON_JSON", "lexicon_singlish.json")
DEFAULT_DB = os.environ.get("LEXICON_DB", "/home/cdsw/lexicon_db")
DEFAULT_TABLE = os.environ.get("LEXICON_TABLE", "singlish")
DEFAULT_EMBED_MODEL = os.environ.get("LEXICON_EMBED_MODEL", "all-MiniLM-L6-v2")

REQUIRED_FIELDS = ("term", "type", "meaning")
VALID_TYPES = {"particle", "term", "acronym"}


def load_entries(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Lexicon file not found: {path}\n"
            "Create it (see the schema in this file's docstring) or pass a path."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data["entries"] if isinstance(data, dict) else data
    if not isinstance(entries, list) or not entries:
        raise ValueError("Lexicon must contain a non-empty list of entries.")
    return entries


def validate(entry: dict, idx: int) -> None:
    missing = [f for f in REQUIRED_FIELDS if not entry.get(f)]
    if missing:
        raise ValueError(f"Entry #{idx} ({entry.get('term', '?')}) missing: {missing}")
    if entry["type"] not in VALID_TYPES:
        raise ValueError(
            f"Entry #{idx} ({entry['term']}) has invalid type "
            f"'{entry['type']}'; expected one of {sorted(VALID_TYPES)}."
        )


def embedding_text(entry: dict) -> str:
    """Rich text to embed: term (+ expansion) + meaning gives semantic signal."""
    parts = [entry["term"]]
    if entry.get("expansion"):
        parts.append(entry["expansion"])
    parts.append(entry["meaning"])
    return " - ".join(parts)


def build_rows(entries: list[dict], embedder: SentenceTransformer) -> list[dict]:
    rows: list[dict] = []
    seen_ids: set[str] = set()

    for idx, entry in enumerate(entries):
        validate(entry, idx)
        vector = embedder.encode(embedding_text(entry)).tolist()
        term = entry["term"]

        # canonical surface form + every variant share the same canonical fields,
        # but each gets its own phonetic key and row id.
        surfaces = [term, *entry.get("variants", [])]
        for surface in surfaces:
            dmeta = doublemetaphone(surface)[0]
            row_id = f"{term.lower()}::{surface.lower()}"
            if row_id in seen_ids:
                continue
            seen_ids.add(row_id)
            rows.append({
                "id": row_id,
                "term": term,                       # canonical answer returned to caller
                "surface": surface,                 # what was matched (canonical or variant)
                "type": entry["type"],
                "meaning": entry["meaning"],
                "expansion": entry.get("expansion", ""),
                "dmeta": dmeta,                     # phonetic key for repair()
                "vector": vector,
            })
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build/append the Singlish lexicon LanceDB table.")
    p.add_argument("json_path", nargs="?", default=DEFAULT_JSON,
                   help=f"Lexicon JSON file (default: {DEFAULT_JSON})")
    p.add_argument("--append", action="store_true",
                   help="Add to the existing table instead of overwriting it.")
    p.add_argument("--db", default=DEFAULT_DB, help=f"LanceDB path (default: {DEFAULT_DB})")
    p.add_argument("--table", default=DEFAULT_TABLE, help=f"Table name (default: {DEFAULT_TABLE})")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL,
                   help=f"SentenceTransformer model (default: {DEFAULT_EMBED_MODEL})")
    return p.parse_args()


def write_overwrite(db, table: str, rows: list[dict]) -> None:
    db.create_table(table, data=rows, mode="overwrite")


def _list_tables(db) -> list[str]:
    """Use list_tables() on newer LanceDB, fall back to table_names()."""
    lister = getattr(db, "list_tables", None) or db.table_names
    return list(lister())


def write_append(db, table: str, rows: list[dict]) -> int:
    """Append rows, skipping ids already present. Returns number actually added."""
    if table not in _list_tables(db):
        db.create_table(table, data=rows)
        return len(rows)
    tbl = db.open_table(table)
    if "id" not in tbl.schema.names:
        raise ValueError(
            f"Existing table '{table}' has no 'id' column (schema: {tbl.schema.names}). "
            "It was built by an older/different script. Rebuild it cleanly with overwrite:\n"
            "  python build_lexicon.py singlish_lexicon.json   (without --append)"
        )
    existing = set(tbl.to_arrow().column("id").to_pylist())
    new_rows = [r for r in rows if r["id"] not in existing]
    skipped = len(rows) - len(new_rows)
    if skipped:
        print(f"Skipping {skipped} row(s) already present in '{table}'.")
    if new_rows:
        tbl.add(new_rows)
    return len(new_rows)


def main() -> None:
    args = parse_args()
    json_path = Path(args.json_path)

    print(f"Reading lexicon:   {json_path}")
    entries = load_entries(json_path)
    print(f"Entries:           {len(entries)}")

    print(f"Loading embedder:  {args.embed_model}")
    embedder = SentenceTransformer(args.embed_model)

    rows = build_rows(entries, embedder)
    print(f"Rows (incl. variants): {len(rows)}")

    mode = "append" if args.append else "overwrite"
    print(f"Writing LanceDB:   {args.db} (table '{args.table}', mode '{mode}')")
    db = lancedb.connect(args.db)
    if args.append:
        added = write_append(db, args.table, rows)
        print(f"Appended {added} new row(s).")
    else:
        write_overwrite(db, args.table, rows)

    count = db.open_table(args.table).count_rows()
    print(f"Done. Table '{args.table}' now has {count} rows.")


if __name__ == "__main__":
    main()
