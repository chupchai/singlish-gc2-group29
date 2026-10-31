"""
analyze_zeroshot_formatting.py
Read-only diagnostic: how much of a zero-shot (un-fine-tuned) Whisper WER is written-style
formatting rather than mishearing. Compares a base-model per-row eval CSV with a fine-tuned
one on identical rows (evaluate_whisper_speaker_split.py output) and counts:
  - digits:       rows where the prediction has a digit and the reference none (and reverse);
                  rows where the reference has a spelled number word and the prediction a digit
  - fillers:      uh/um/uhm/er/erm/hmm/mm/mhm in references vs predictions, and how many
                  reference fillers are missing from the prediction ('ah'/'eh' are left out:
                  they are Singlish particles in the MNSC references)
  - contractions: tokens with an apostrophe
  - spelling:     rows where the reference uses the British form of a short fixed word list
                  and the prediction the US form (and reverse)
plus a DIAGNOSTIC corpus WER with Whisper's EnglishTextNormalizer applied to the reference AND
prediction of BOTH models (sensitivity check only, not an official number).
The CSV text is already normalize_for_evaluation() output (lowercase, punctuation except
apostrophes removed), so e.g. a predicted '5%' is already '5' in the CSV, and Whisper's
normalizer may turn a reference 'five percent' into '5%' while the prediction stays '5'.
Never writes to results/; the official normalizer and numbers are unchanged.
Usage (from the project root):
    python scripts/analyze_zeroshot_formatting.py \
        --pair "Held-out rows, fallback" results/eval_whisper-small-zeroshot_ho-part3_15k_heldout_fb.csv \
               results/eval_whisper-small-mnsc-sg-15k-heldout_ho-part3_15k_heldout_fb.csv \
        --out_md formatting_analysis.md
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import jiwer
import pandas as pd

from evaluate_whisper_speaker_split import RESULTS_DIR, corpus_metric

KEY = ["config", "row_index"]
FILLERS = ("uh", "um", "uhm", "er", "erm", "hmm", "mm", "mhm")
# 'one' is left out: mostly a pronoun ('one of the best') in the references
NUMBER_WORDS = ("zero", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
                "eighty", "ninety", "hundred", "thousand", "million", "percent")
SPELLING_PAIRS = (("colour", "color"), ("favourite", "favorite"), ("realise", "realize"),
                  ("organise", "organize"), ("centre", "center"), ("theatre", "theater"),
                  ("neighbour", "neighbor"), ("behaviour", "behavior"), ("travelled", "traveled"),
                  ("programme", "program"), ("metres", "meters"), ("metre", "meter"),
                  ("licence", "license"))
DEFAULT_NORMALIZER_MODEL = "openai/whisper-small"
DEFAULT_EXAMPLES = 8


# ============================================================
# INPUT
# ============================================================

def read_eval_csv(path) -> pd.DataFrame:
    """Per-row eval CSV with empty strings kept as '' (as scored by the eval script)."""
    df = pd.read_csv(path, keep_default_na=False, dtype={"reference": str, "prediction": str})
    missing = [c for c in KEY + ["reference", "prediction"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df


def align(base: pd.DataFrame, ft: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Both frames sorted by (config, row_index); raises unless they hold identical rows and
    identical references."""
    a = base.sort_values(KEY).reset_index(drop=True)
    b = ft.sort_values(KEY).reset_index(drop=True)
    if len(a) != len(b) or not (a[KEY].values == b[KEY].values).all():
        raise ValueError("the two CSVs do not hold the same (config, row_index) rows")
    if not (a["reference"].values == b["reference"].values).all():
        raise ValueError("the two CSVs have different references for the same rows")
    return a, b


# ============================================================
# COUNTS
# ============================================================

def _words(text: str) -> List[str]:
    return text.split()


def _count(words: List[str], vocab) -> int:
    return sum(w in vocab for w in words)


def has_digit(text: str) -> bool:
    return bool(re.search(r"\d", text))


def format_counts(df: pd.DataFrame) -> Dict[str, int]:
    """Formatting counts of one model's rows (reference vs prediction)."""
    fill, nums = set(FILLERS), set(NUMBER_WORDS)
    c = dict.fromkeys(["rows", "pred_digit_ref_none", "ref_digit_pred_none", "ref_numword_pred_digit",
                       "ref_digit_tokens", "pred_digit_tokens", "ref_fillers", "pred_fillers",
                       "fillers_dropped", "fillers_added", "ref_apostrophe_tokens",
                       "pred_apostrophe_tokens", "ref_uk_pred_us", "ref_us_pred_uk"], 0)
    for ref, pred in zip(df["reference"], df["prediction"]):
        rw, pw = _words(ref), _words(pred)
        c["rows"] += 1
        rd, pd_ = has_digit(ref), has_digit(pred)
        c["pred_digit_ref_none"] += int(pd_ and not rd)
        c["ref_digit_pred_none"] += int(rd and not pd_)
        c["ref_numword_pred_digit"] += int(pd_ and not rd and _count(rw, nums) > 0)
        c["ref_digit_tokens"] += sum(has_digit(w) for w in rw)
        c["pred_digit_tokens"] += sum(has_digit(w) for w in pw)
        rf, pf = _count(rw, fill), _count(pw, fill)
        c["ref_fillers"] += rf
        c["pred_fillers"] += pf
        c["fillers_dropped"] += max(0, rf - pf)
        c["fillers_added"] += max(0, pf - rf)
        c["ref_apostrophe_tokens"] += sum("'" in w for w in rw)
        c["pred_apostrophe_tokens"] += sum("'" in w for w in pw)
        rs, ps = set(rw), set(pw)
        c["ref_uk_pred_us"] += int(any(uk in rs and us in ps for uk, us in SPELLING_PAIRS))
        c["ref_us_pred_uk"] += int(any(us in rs and uk in ps for uk, us in SPELLING_PAIRS))
    return c


# ============================================================
# DIAGNOSTIC WER
# ============================================================

def row_errors(refs: List[str], preds: List[str]) -> Tuple[List[int], List[int], int]:
    """(word errors per row, reference words per row, rows with an empty reference). An
    empty reference counts every predicted word as an insertion."""
    errs, nref, empty = [], [], 0
    for r, p in zip(refs, preds):
        if not r.split():
            empty += 1
            errs.append(len(p.split()))
            nref.append(0)
            continue
        out = jiwer.process_words(r, p)
        errs.append(out.substitutions + out.deletions + out.insertions)
        nref.append(len(r.split()))
    return errs, nref, empty


def diagnostic_wer(df: pd.DataFrame, normalize: Callable[[str], str]) -> Tuple[float, int, List[int]]:
    """(corpus WER after normalize() on both columns, empty-reference rows, per-row errors)."""
    refs = [normalize(t) for t in df["reference"]]
    preds = [normalize(t) for t in df["prediction"]]
    errs, nref, empty = row_errors(refs, preds)
    return sum(errs) / max(1, sum(nref)), empty, errs


def load_whisper_normalizer(model_id: str, revision: str) -> Callable[[str], str]:
    """Whisper's EnglishTextNormalizer with its spelling map (normalizer.json from the HF cache)."""
    from huggingface_hub import hf_hub_download
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    path = hf_hub_download(repo_id=model_id, filename="normalizer.json", revision=revision)
    with open(path, "r", encoding="utf-8") as fh:
        return EnglishTextNormalizer(json.load(fh))


# ============================================================
# REPORT
# ============================================================

def groups_of(df: pd.DataFrame) -> List[Tuple[str, pd.Series]]:
    out = [("Overall", pd.Series(True, index=df.index))]
    out += [(g, df["config"] == g) for g in df["config"].drop_duplicates()]
    return out


def pair_report(label: str, base_path, ft_path, normalize: Callable[[str], str],
                n_examples: int = DEFAULT_EXAMPLES) -> str:
    base, ft = align(read_eval_csv(base_path), read_eval_csv(ft_path))
    lines = [f"## {label}", "", f"- base: `{Path(base_path).as_posix()}`",
             f"- fine-tuned: `{Path(ft_path).as_posix()}`", ""]

    lines += ["### WER: official vs diagnostic (Whisper EnglishTextNormalizer on both sides)", "",
              "| group | rows | base official | base diagnostic | fine-tuned official | "
              "fine-tuned diagnostic | empty refs after normalizer |",
              "|---|---|---|---|---|---|---|"]
    diag_errs = {}
    for g, m in groups_of(base):
        vals = []
        for name, df in (("base", base), ("ft", ft)):
            sub = df[m]
            off = corpus_metric(sub, "WER")
            dg, empty, errs = diagnostic_wer(sub, normalize)
            vals.append((off, dg, empty))
            if g == "Overall":
                diag_errs[name] = errs
        lines.append(f"| {g} | {int(m.sum())} | {vals[0][0]:.4f} | {vals[0][1]:.4f} | {vals[1][0]:.4f} | "
                     f"{vals[1][1]:.4f} | {vals[0][2]} |")
    lines.append("")

    keys = list(format_counts(base.head(0)).keys())
    lines += ["### Formatting counts (on the CSV text)", "",
              "| group | model | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
    for g, m in groups_of(base):
        for name, df in (("base", base), ("fine-tuned", ft)):
            c = format_counts(df[m])
            lines.append(f"| {g} | {name} | " + " | ".join(str(c[k]) for k in keys) + " |")
    lines.append("")

    off_errs, _, _ = row_errors(list(base["reference"]), list(base["prediction"]))
    gain = pd.Series([o - d for o, d in zip(off_errs, diag_errs["base"])], index=base.index)
    top = gain.sort_values(ascending=False, kind="stable").head(n_examples)
    lines += [f"### Base rows whose word errors drop most under the diagnostic normalizer "
              f"(top {len(top)})", "",
              "| config | row_index | errors official -> diagnostic | reference | base prediction | "
              "fine-tuned prediction |", "|---|---|---|---|---|---|"]
    for i in top.index:
        r = base.loc[i]
        cut = (lambda s: s if len(s) <= 160 else s[:157] + "...")
        lines.append(f"| {r['config']} | {r['row_index']} | {off_errs[i]} -> {diag_errs['base'][i]} | "
                     f"{cut(r['reference'])} | {cut(r['prediction'])} | {cut(ft.loc[i, 'prediction'])} |")
    lines.append("")
    return "\n".join(lines)


HEADER = """# Zero-shot formatting-mismatch diagnostic

Not official numbers. "official" = corpus WER recomputed from the per-row CSV, which equals the
eval summary. "diagnostic" = corpus WER after Whisper's EnglishTextNormalizer is applied to the
reference and the prediction of BOTH models (sensitivity check only). The CSV text is already
normalize_for_evaluation() output, so punctuation and symbols such as '%' were removed before
this script runs. Fillers counted: {fillers}. Spelling pairs: {pairs}.
"""


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Read-only formatting-mismatch diagnostic for a zero-shot "
                                            "vs fine-tuned pair of per-row eval CSVs.")
    p.add_argument("--pair", nargs=3, action="append", required=True, metavar=("LABEL", "BASE_CSV", "FT_CSV"),
                   help="Label, base-model eval CSV, fine-tuned eval CSV on the same rows (repeatable).")
    p.add_argument("--normalizer_model", default=DEFAULT_NORMALIZER_MODEL,
                   help=f"Hub repo whose normalizer.json is used (default {DEFAULT_NORMALIZER_MODEL}).")
    p.add_argument("--normalizer_revision", default="main", help="Revision of --normalizer_model.")
    p.add_argument("--examples", type=int, default=DEFAULT_EXAMPLES, help="Example rows per pair.")
    p.add_argument("--out_md", default=None,
                   help="Also write the markdown report here (not inside results/).")
    p.add_argument("--overwrite", action="store_true", help="Replace an existing --out_md file.")
    args = p.parse_args(argv)
    if args.out_md:
        out = Path(args.out_md).resolve()
        if out.is_relative_to(RESULTS_DIR.resolve()):
            p.error("--out_md must not be inside results/ (this diagnostic is not an official result)")
        if out.exists() and not args.overwrite:
            p.error(f"{out} exists; pass --overwrite to replace it")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    normalize = load_whisper_normalizer(args.normalizer_model, args.normalizer_revision)
    parts = [HEADER.format(fillers=", ".join(FILLERS),
                           pairs=", ".join(f"{a}/{b}" for a, b in SPELLING_PAIRS))]
    parts += [pair_report(label, b, f, normalize, args.examples) for label, b, f in args.pair]
    text = "\n".join(parts)
    sys.stdout.buffer.write(text.encode("utf-8") + b"\n")
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
