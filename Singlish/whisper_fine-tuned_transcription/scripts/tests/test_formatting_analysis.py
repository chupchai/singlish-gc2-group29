"""
Offline tests for analyze_zeroshot_formatting.py (no network, no model loading).

Run from the project root:
    python -m unittest discover -s scripts/tests -t . -v
"""

import io
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import analyze_zeroshot_formatting as af        # noqa: E402
import evaluate_whisper_speaker_split as ev     # noqa: E402


def frame(refs, preds, config="ASR-PART3-Test"):
    return pd.DataFrame({"config": config, "row_index": list(range(len(refs))),
                         "reference": refs, "prediction": preds})


class FormattingAnalysisTests(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_counts(self):
        df = frame(["about three metres away", "uh i um think so", "it's my favourite colour", "5 cats"],
                   ["about 3 meters away", "i think so uh", "it is my favorite color", "five cats"])
        c = af.format_counts(df)
        self.assertEqual(c["rows"], 4)
        self.assertEqual((c["pred_digit_ref_none"], c["ref_digit_pred_none"]), (1, 1))
        self.assertEqual(c["ref_numword_pred_digit"], 1)
        self.assertEqual((c["ref_digit_tokens"], c["pred_digit_tokens"]), (1, 1))
        self.assertEqual((c["ref_fillers"], c["pred_fillers"]), (2, 1))
        self.assertEqual((c["fillers_dropped"], c["fillers_added"]), (1, 0))
        self.assertEqual((c["ref_apostrophe_tokens"], c["pred_apostrophe_tokens"]), (1, 0))
        self.assertEqual((c["ref_uk_pred_us"], c["ref_us_pred_uk"]), (2, 0))

    def test_align_raises_on_mismatch(self):
        a = frame(["a b", "c d"], ["a b", "c"])
        b = frame(["a b", "c d"], ["a", "c d"])
        x, y = af.align(a, b.iloc[::-1])
        self.assertEqual(list(x["row_index"]), list(y["row_index"]))
        with self.assertRaises(ValueError):
            af.align(a, b.head(1))
        c = b.copy()
        c.loc[1, "row_index"] = 7
        with self.assertRaises(ValueError):
            af.align(a, c)
        d = b.copy()
        d.loc[0, "reference"] = "other"
        with self.assertRaises(ValueError):
            af.align(a, d)

    def test_diagnostic_wer_identity_equals_official(self):
        df = frame(["the cat sat", "hello there world", "uh yes"], ["the cat sit", "hello world", "yes yes"])
        wer, empty, errs = af.diagnostic_wer(df, lambda s: s)
        self.assertAlmostEqual(wer, ev.corpus_metric(df, "WER"), places=12)
        self.assertEqual((empty, errs), (0, [1, 1, 1]))
        # a reference emptied by the normalizer counts its predicted words as errors
        wer, empty, errs = af.diagnostic_wer(frame(["uh", "a b"], ["oh", "a b"]),
                                             lambda s: s.replace("uh", "").strip())
        self.assertEqual((empty, errs, wer), (1, [1, 0], 0.5))

    def test_out_md_guard(self):
        pair = ["--pair", "x", "a.csv", "b.csv"]
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            af.parse_args(pair + ["--out_md", str(ev.RESULTS_DIR / "x.md")])
        existing = self.tmp / "r.md"
        existing.write_text("old", encoding="utf-8")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            af.parse_args(pair + ["--out_md", str(existing)])
        self.assertTrue(af.parse_args(pair + ["--out_md", str(existing), "--overwrite"]).overwrite)
        self.assertEqual(existing.read_text(encoding="utf-8"), "old")

    def test_main_writes_only_out_md(self):
        base = self.tmp / "eval_base.csv"
        ft = self.tmp / "eval_ft.csv"
        frame(["three cats", "uh ok"], ["3 cats", "ok"]).to_csv(base, index=False)
        frame(["three cats", "uh ok"], ["three cats", "uh ok"]).to_csv(ft, index=False)
        out = self.tmp / "report.md"
        with mock.patch.object(af, "load_whisper_normalizer", return_value=lambda s: s), \
                mock.patch.object(sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding="utf-8")):
            self.assertEqual(af.main(["--pair", "t", str(base), str(ft), "--out_md", str(out)]), 0)
        text = out.read_text(encoding="utf-8")
        self.assertIn("Not official numbers", text)
        self.assertIn("| Overall | 2 | 0.5000 | 0.5000 | 0.0000 | 0.0000 | 0 |", text)
        self.assertEqual(sorted(p.name for p in self.tmp.iterdir()),
                         ["eval_base.csv", "eval_ft.csv", "report.md"])


if __name__ == "__main__":
    unittest.main()
