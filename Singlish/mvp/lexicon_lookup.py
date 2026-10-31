"""Query the Singlish lexicon (LanceDB) and correct transcript words.

Pairs with build_lexicon.py. Strategy:
  1. Phonetic match first  - catches ASR mis-hearings ("shook" -> "shiok")
     via the precomputed double-metaphone key.
  2. Semantic match second - vector search over the term embeddings, gated by a
     cosine-distance threshold.
A PROTECTED set and a minimum token length guard against "correcting" ordinary
English words that collide with short Singlish forms.

Example:
    from lexicon_lookup import Lexicon
    lex = Lexicon()
    lex.repair("shook")          # -> canonical 'shiok' row
    lex.correct([{ "text": "shook", "timestamp": (1.2, 1.5) }])
"""
from __future__ import annotations

import os
import re

import lancedb
from metaphone import doublemetaphone
from sentence_transformers import SentenceTransformer

DB_PATH = os.environ.get("LEXICON_DB", "/home/cdsw/lexicon_db")
TABLE_NAME = os.environ.get("LEXICON_TABLE", "singlish")
EMBED_MODEL = os.environ.get("LEXICON_EMBED_MODEL", "all-MiniLM-L6-v2")

# Max cosine distance (LanceDB cosine = 1 - similarity) for a semantic match.
SEMANTIC_MAX_DISTANCE = float(os.environ.get("LEXICON_SEMANTIC_MAX_DISTANCE", "0.45"))
# Tokens shorter than this are never auto-corrected.
MIN_TOKEN_LEN = int(os.environ.get("LEXICON_MIN_TOKEN_LEN", "3"))

# Ordinary English words (or ambiguous short forms) we must NOT auto-replace.
# Tune this against real transcripts - it is your main false-positive control.
PROTECTED = {
    "la", "lay", "law", "or", "the", "a", "an", "and", "ah", "ma", "ha",
    "shock", "look", "loud", "blur", "steady", "can", "chop", "wa", "war",
    "makan",  # keep only if it causes issues; remove to allow correction
}


def normalize(token: str) -> str:
    return re.sub(r"[^\w]", "", token.lower()).strip()


# Matches word-like tokens (letters/digits/apostrophes), leaving punctuation in place.
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


class Lexicon:
    def __init__(self, db_path: str = DB_PATH, table: str = TABLE_NAME,
                 embed_model: str = EMBED_MODEL,
                 semantic_max_distance: float = SEMANTIC_MAX_DISTANCE,
                 min_token_len: int = MIN_TOKEN_LEN,
                 protected: set[str] | None = None):
        self.tbl = lancedb.connect(db_path).open_table(table)
        self.embedder = SentenceTransformer(embed_model)
        self.semantic_max_distance = semantic_max_distance
        self.min_token_len = min_token_len
        self.protected = PROTECTED if protected is None else protected

    def phonetic(self, token: str) -> list[dict]:
        key = doublemetaphone(token)[0]
        if not key:
            return []
        return self.tbl.search().where(f"dmeta = '{key}'").limit(1).to_list()

    def semantic(self, token: str) -> list[dict]:
        query = self.embedder.encode(token)
        hits = self.tbl.search(query).metric("cosine").limit(1).to_list()
        return [h for h in hits if h.get("_distance", 1.0) <= self.semantic_max_distance]

    def repair(self, token: str) -> dict | None:
        """Return the best lexicon row for a token, or None if no safe match."""
        key = normalize(token)
        if len(key) < self.min_token_len or key in self.protected:
            return None
        hit = self.phonetic(key) or self.semantic(key)
        return hit[0] if hit else None

    @staticmethod
    def _apply_case(original: str, replacement: str) -> str:
        if original.isupper():
            return replacement.upper()
        if original.istitle():
            return replacement.capitalize()
        return replacement

    def render(self, hit: dict, original: str) -> str:
        """Turn a matched row into output text (expand acronyms, else normalize)."""
        if hit["type"] == "acronym" and hit.get("expansion"):
            return f'{hit["term"]} ({hit["expansion"]})'
        return self._apply_case(original, hit["term"])

    def correct_text(self, text: str) -> tuple[str, list[dict]]:
        """Correct a plain transcript string (no word timestamps needed).

        Returns (corrected_text, changes) where changes lists every replacement:
        [{'original', 'replacement', 'term'}]. Punctuation and spacing are preserved.
        """
        changes: list[dict] = []

        def _repl(match: re.Match) -> str:
            token = match.group(0)
            hit = self.repair(token)
            if not hit:
                return token
            new = self.render(hit, token)
            if new != token:
                changes.append({"original": token, "replacement": new, "term": hit["term"]})
            return new

        return _WORD_RE.sub(_repl, text), changes

    def correct(self, words: list[dict]) -> list[dict]:
        """words: [{'text', 'timestamp'}]  ->  same list with corrections applied.

        Each output row carries 'original' and 'changed' so a UI can highlight edits.
        """
        out = []
        for w in words:
            original = w.get("text", "")
            hit = self.repair(original)
            text = self.render(hit, original) if hit else original
            out.append({
                "text": text,
                "original": original,
                "changed": text != original,
                "matched_term": hit["term"] if hit else None,
                "timestamp": w.get("timestamp"),
            })
        return out


if __name__ == "__main__":
    # quick smoke test
    lex = Lexicon()
    for t in ["shook", "PSLE", "hdb", "kiasu", "the", "shock"]:
        print(f"{t!r:10} -> {lex.repair(t)}")
