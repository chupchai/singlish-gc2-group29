"""
Offline tests for the shared decoding module, the eval script's decoding defaults /
overwrite guard and transcribe.py (no model loading, no GPU, no downloads).

Run from the project root:
    python -m unittest discover -s scripts/tests -t . -v
"""

import csv
import io
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf
import torch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import evaluate_whisper_speaker_split as ev  # noqa: E402
import transcribe as tx                      # noqa: E402
import whisper_decoding as wd                # noqa: E402

FALLBACK_KW = {"temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
               "compression_ratio_threshold": 1.35, "logprob_threshold": -1.0}


class _Features:
    def __init__(self, n):
        self.input_features = torch.zeros(n, 80, 3000)


class FakeProcessor:
    """Stands in for WhisperProcessor: records calls, decodes to 'chunk<i>'."""

    def __init__(self):
        self.sampling_rates = []
        self.lengths = []
        self.decoded = 0
        self.tokenizer = mock.Mock(eos_token_id=50257)

    def __call__(self, audios, sampling_rate, return_tensors):
        self.sampling_rates.append(sampling_rate)
        self.lengths.extend(len(a) for a in audios)
        return _Features(len(audios))

    def batch_decode(self, ids, skip_special_tokens=True):
        out = [f"chunk{self.decoded + i}" for i in range(len(ids))]
        self.decoded += len(ids)
        return out


class FakeModel:
    """Stands in for WhisperForConditionalGeneration.generate."""

    def __init__(self):
        self.calls = []
        self.config = mock.Mock(vocab_size=51865)

    def generate(self, input_features, **kw):
        self.calls.append((input_features.dtype, kw))
        return torch.zeros(input_features.shape[0], 3, dtype=torch.long)


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


# ============================================================
# SHARED DECODING MODULE
# ============================================================

class WhisperDecodingTests(unittest.TestCase):

    def test_constants_and_default_kwargs(self):
        self.assertEqual(wd.FALLBACK_TEMPERATURES, (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
        self.assertEqual((wd.DEFAULT_COMPRESSION_RATIO_THRESHOLD, wd.DEFAULT_LOGPROB_THRESHOLD,
                          wd.DEFAULT_SEED), (1.35, -1.0, 42))
        self.assertEqual((wd.LANGUAGE, wd.TASK, wd.SAMPLE_RATE), ("en", "transcribe", 16000))
        kw = wd.generate_kwargs()
        self.assertEqual(kw, FALLBACK_KW)
        self.assertNotIn("do_sample", kw)
        self.assertEqual(wd.generate_kwargs(temperature_fallback=False), {})
        kw = wd.generate_kwargs(5, 225)
        self.assertEqual(list(kw), ["no_repeat_ngram_size", "max_new_tokens", "temperature",
                                    "compression_ratio_threshold", "logprob_threshold"])
        self.assertEqual(wd.generate_kwargs(3, temperature_fallback=False), {"no_repeat_ngram_size": 3})

    def test_eval_and_transcribe_share_implementation(self):
        for name in ("token_compression_ratio", "generated_sequences", "decoding_kwargs",
                     "decode_batch", "load_model", "FALLBACK_TEMPERATURES",
                     "DEFAULT_COMPRESSION_RATIO_THRESHOLD", "DEFAULT_LOGPROB_THRESHOLD", "DEFAULT_SEED"):
            self.assertIs(getattr(ev, name), getattr(wd, name), name)
        for name in ("decode_batch", "load_model", "decoding_kwargs", "add_decoding_args",
                     "resolve_decoding_args"):
            self.assertIs(getattr(tx, name), getattr(wd, name), name)

    def test_decode_batch_calls_generate(self):
        proc, model = FakeProcessor(), FakeModel()
        audios = [np.zeros(16000, np.float32), np.zeros(8000, np.float32)]
        texts, seqs = wd.decode_batch(model, proc, audios, 16000, "cpu", torch.float32,
                                      {"temperature": (0.0, 0.2)})
        self.assertEqual(texts, ["chunk0", "chunk1"])
        self.assertEqual(seqs.shape[0], 2)
        self.assertEqual(proc.sampling_rates, [16000])
        dtype, kw = model.calls[0]
        self.assertEqual(dtype, torch.float32)
        self.assertEqual(kw, {"language": "en", "task": "transcribe", "temperature": (0.0, 0.2)})

    def test_load_model_in_memory_only(self):
        fake = mock.MagicMock()
        with mock.patch.object(wd.WhisperProcessor, "from_pretrained", return_value="proc") as fp, \
                mock.patch.object(wd.WhisperForConditionalGeneration, "from_pretrained",
                                  return_value=fake), \
                mock.patch.object(wd.torch.cuda, "is_available", return_value=False):
            proc, model, device, dtype = wd.load_model(Path("some") / "model")
        fp.assert_called_once_with(str(Path("some") / "model"))
        self.assertEqual((proc, model, device, dtype), ("proc", fake, "cpu", torch.float32))
        fake.to.assert_called_once_with("cpu", dtype=torch.float32)
        fake.eval.assert_called_once()
        self.assertEqual((fake.generation_config.language, fake.generation_config.task),
                         ("en", "transcribe"))
        fake.save_pretrained.assert_not_called()

    def test_mode_label(self):
        self.assertEqual(wd.decoding_mode_label(True), "temperature fallback (default)")
        self.assertEqual(wd.decoding_mode_label(False), "plain greedy (--no_temperature_fallback)")
        self.assertEqual(wd.decoding_mode_label(True, 5),
                         "temperature fallback (default) + no_repeat_ngram_size 5")


# ============================================================
# EVAL SCRIPT: DEFAULT DECODING, FILE NAMES, OVERWRITE GUARD
# ============================================================

class EvalDefaultTests(TempDirCase):

    def _err(self, argv):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            ev.parse_args(argv)

    def test_fallback_default_and_flags(self):
        a = ev.parse_args([])
        self.assertTrue(a.temperature_fallback)
        self.assertEqual(ev.decoding_kwargs(a), FALLBACK_KW)
        self.assertFalse(a.overwrite)
        b = ev.parse_args(["--temperature_fallback"])  # accepted no-op
        self.assertEqual((b.temperature_fallback, b.compression_ratio_threshold, b.logprob_threshold),
                         (a.temperature_fallback, a.compression_ratio_threshold, a.logprob_threshold))
        self.assertEqual(ev.decoding_kwargs(b), ev.decoding_kwargs(a))
        c = ev.parse_args(["--no_temperature_fallback"])
        self.assertFalse(c.temperature_fallback)
        self.assertEqual(ev.decoding_kwargs(c), {})
        self._err(["--temperature_fallback", "--no_temperature_fallback"])
        self._err(["--no_temperature_fallback", "--logprob_threshold", "-999"])
        d = ev.parse_args(["--compression_ratio_threshold", "1.5"])
        self.assertEqual(d.compression_ratio_threshold, 1.5)
        self.assertTrue(ev.parse_args(["--overwrite"]).overwrite)

    def test_run_suffix_names(self):
        def s(*argv):
            return ev.run_suffix(ev.parse_args(list(argv)))
        self.assertEqual(s(), "_fb")
        self.assertEqual(s("--temperature_fallback"), "_fb")
        self.assertEqual(s("--no_temperature_fallback"), "")
        self.assertEqual(s("--heldout_test", "part3_15k_heldout"), "_ho-part3_15k_heldout_fb")
        self.assertEqual(s("--heldout_test", "part3_15k_heldout", "--no_temperature_fallback"),
                         "_ho-part3_15k_heldout")
        self.assertEqual(s("--split", "val", "--no_repeat_ngram_size", "5"), "_val_nrng5_fb")
        self.assertEqual(s("--split", "val", "--no_repeat_ngram_size", "5", "--no_temperature_fallback"),
                         "_val_nrng5")
        self.assertEqual(s("--max_rows", "16"), "_rows16_fb")
        self.assertEqual(s("--max_rows", "16", "--no_temperature_fallback"), "_rows16")
        self.assertEqual(s("--logprob_threshold", "-999", "--split", "val"), "_val_fb_lp-999")
        self.assertEqual(s("--seed", "7"), "_fb_seed7")
        self.assertEqual(s("--seed", "7", "--no_temperature_fallback"), "")
        run = "whisper-small-mnsc-sg-15k"
        self.assertEqual(ev.output_paths(run, s("--split", "val"))[0].name, f"eval_{run}_val_fb.csv")
        self.assertEqual(ev.output_paths(run, s("--split", "val", "--no_temperature_fallback"))[1].name,
                         f"summary_{run}_val.csv")

    def _main(self, argv):
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = ev.main(argv)
        return code, out.getvalue()

    def test_overwrite_guard(self):
        run = "fake-run"
        models, results = self.tmp / "models", self.tmp / "results"
        (models / run).mkdir(parents=True)
        (models / run / "config.json").write_text("{}", encoding="utf-8")
        results.mkdir()

        class Loaded(Exception):
            pass

        loader = mock.Mock(side_effect=Loaded("data loading reached"))
        model_loader = mock.Mock(side_effect=AssertionError("must not load"))
        with mock.patch.object(ev, "MODELS_DIR", models), mock.patch.object(ev, "RESULTS_DIR", results), \
                mock.patch.object(ev, "load_test_manifest", loader), \
                mock.patch.object(ev, "load_model", model_loader):
            for name in (f"eval_{run}_fb.csv", f"summary_{run}_fb.csv"):
                target = results / name
                target.write_bytes(b"old results\n")
                code, out = self._main(["--run_name", run])
                self.assertEqual(code, 2)
                self.assertIn("refusing to overwrite", out)
                self.assertIn(name, out)
                self.assertIn("temperature fallback (default)", out)
                self.assertEqual(target.read_bytes(), b"old results\n")
                loader.assert_not_called()
                model_loader.assert_not_called()
                target.unlink()
            # only _fb files exist -> a plain run targets other names and passes the guard
            (results / f"eval_{run}_fb.csv").write_bytes(b"old results\n")
            out = io.StringIO()
            with redirect_stdout(out), self.assertRaises(Loaded):
                ev.main(["--run_name", run, "--no_temperature_fallback"])
            self.assertIn("plain greedy (--no_temperature_fallback)", out.getvalue())
            self.assertNotIn("refusing", out.getvalue())
            # --overwrite passes the guard too
            with redirect_stdout(io.StringIO()), self.assertRaises(Loaded):
                ev.main(["--run_name", run, "--overwrite"])
            self.assertEqual(loader.call_count, 2)
            model_loader.assert_not_called()
            self.assertEqual((results / f"eval_{run}_fb.csv").read_bytes(), b"old results\n")


# ============================================================
# EVAL SCRIPT: --model_id (ZERO-SHOT HUB BASELINE)
# ============================================================

class EvalHubModelTests(TempDirCase):

    HUB = "openai/whisper-small"
    COMMIT = "a" * 40

    def _err(self, argv):
        err = io.StringIO()
        with redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            ev.parse_args(argv)
        self.assertEqual(cm.exception.code, 2)
        return err.getvalue()

    def _main(self, argv):
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = ev.main(argv)
        return code, out.getvalue()

    def test_defaults_unchanged_without_model_id(self):
        a = ev.parse_args([])
        self.assertEqual((a.run_name, a.model_id, a.model_revision),
                         ("whisper-small-mnsc-sg-15k", None, None))
        self.assertEqual(ev.parse_args(["--run_name", "whisper-small-mnsc-sg-4k"]).run_name,
                         "whisper-small-mnsc-sg-4k")
        v = ev.parse_args(["--run_name", "whisper-small-mnsc-sg-15k-heldout", "--split", "val"])
        self.assertEqual(v.data_name, "part3_15k_heldout")

    def test_model_id_naming_and_revision(self):
        a = ev.parse_args(["--model_id", self.HUB])
        self.assertEqual((a.run_name, a.model_revision), ("whisper-small-zeroshot", "main"))
        self.assertEqual(ev.hub_run_name(self.HUB), "whisper-small-zeroshot")
        self.assertEqual(ev.parse_args(["--model_id", self.HUB, "--model_revision", "abc"]).model_revision,
                         "abc")
        with mock.patch.object(ev, "MODELS_DIR", self.tmp / "models"):
            self.assertEqual(ev.parse_args(["--model_id", self.HUB, "--run_name", "base-x"]).run_name,
                             "base-x")

    def test_model_id_errors(self):
        self.assertIn("requires --model_id", self._err(["--model_revision", "main"]))
        for bad in ("models/x/y", "C:\\x", "..\\x", "whisper-small", "", "a/..", "/x"):
            self.assertIn("--model_id must be", self._err(["--model_id", bad]))
        self.assertIn("requires --split test", self._err(["--model_id", self.HUB, "--split", "val"]))
        models = self.tmp / "models"
        (models / "fine-tuned").mkdir(parents=True)
        with mock.patch.object(ev, "MODELS_DIR", models):
            self.assertIn("model folder under models/",
                          self._err(["--model_id", self.HUB, "--run_name", "fine-tuned"]))

    def test_output_names(self):
        def names(*argv):
            a = ev.parse_args(["--model_id", self.HUB, *argv])
            return [p.name for p in ev.output_paths(a.run_name, ev.run_suffix(a))]
        z = "whisper-small-zeroshot"
        self.assertEqual(names("--no_temperature_fallback"), [f"eval_{z}.csv", f"summary_{z}.csv"])
        self.assertEqual(names()[0], f"eval_{z}_fb.csv")
        self.assertEqual(names("--heldout_test", "part3_15k_heldout")[1],
                         f"summary_{z}_ho-part3_15k_heldout_fb.csv")
        self.assertEqual(names("--heldout_test", "part3_15k_heldout", "--no_temperature_fallback")[0],
                         f"eval_{z}_ho-part3_15k_heldout.csv")
        self.assertEqual(names("--heldout_extra_test", "part3_15k_heldout_extra_test")[0],
                         f"eval_{z}_hx-part3_15k_heldout_extra_test_fb.csv")
        self.assertEqual(names("--max_rows", "16", "--no_temperature_fallback")[0], f"eval_{z}_rows16.csv")

    def test_resolve_hub_model(self):
        snap = self.tmp / "snapshots" / ("0123456789abcdef" * 3)[:40]
        with mock.patch.object(ev, "snapshot_download", return_value=str(snap)) as sd:
            path, commit = ev.resolve_hub_model(self.HUB, "main")
        self.assertEqual((path, commit), (snap, snap.name))
        sd.assert_called_once_with(repo_id=self.HUB, revision="main", allow_patterns=ev.HUB_MODEL_FILES)
        with mock.patch.object(ev, "snapshot_download", return_value=str(snap)):
            self.assertEqual(ev.resolve_hub_model(self.HUB, snap.name)[1], snap.name)
            with self.assertRaises(ValueError):
                ev.resolve_hub_model(self.HUB, "b" * 40)
        with mock.patch.object(ev, "snapshot_download", return_value=str(self.tmp / "not-a-hash")):
            with self.assertRaises(ValueError):
                ev.resolve_hub_model(self.HUB, "main")

    def test_overwrite_guard_before_resolving(self):
        results = self.tmp / "results"
        results.mkdir()
        (results / "eval_whisper-small-zeroshot_fb.csv").write_bytes(b"old\n")
        never = mock.Mock(side_effect=AssertionError("must not be called"))
        with mock.patch.object(ev, "RESULTS_DIR", results), \
                mock.patch.object(ev, "MODELS_DIR", self.tmp / "models"), \
                mock.patch.object(ev, "resolve_hub_model", never), \
                mock.patch.object(ev, "load_test_manifest", never), \
                mock.patch.object(ev, "load_model", never):
            code, out = self._main(["--model_id", self.HUB])
        self.assertEqual(code, 2)
        self.assertIn("refusing to overwrite", out)
        never.assert_not_called()
        self.assertEqual((results / "eval_whisper-small-zeroshot_fb.csv").read_bytes(), b"old\n")

    def test_resolve_error_returns_2(self):
        results = self.tmp / "results"
        never = mock.Mock(side_effect=AssertionError("must not be called"))
        with mock.patch.object(ev, "RESULTS_DIR", results), \
                mock.patch.object(ev, "MODELS_DIR", self.tmp / "models"), \
                mock.patch.object(ev, "resolve_hub_model", side_effect=OSError("offline")), \
                mock.patch.object(ev, "load_test_manifest", never), \
                mock.patch.object(ev, "load_model", never):
            code, out = self._main(["--model_id", self.HUB])
        self.assertEqual(code, 2)
        self.assertIn("ERROR: cannot load openai/whisper-small@main", out)

    def _mocked_run(self, argv, models):
        results = self.tmp / "results"
        manifest = ev.pd.DataFrame({"sample_id": ["p1", "p3"], "dataset": "MNSC-Extended",
                                    "config": ["ASR-PART1-Test", "ASR-PART3-Test"],
                                    "row_index": [0, 0]})
        proc, model = FakeProcessor(), FakeModel()
        loader = mock.Mock(return_value=(proc, model, "cpu", torch.float32))
        with mock.patch.object(ev, "RESULTS_DIR", results), mock.patch.object(ev, "MODELS_DIR", models), \
                mock.patch.object(ev, "resolve_hub_model",
                                  return_value=(self.tmp / "snap", self.COMMIT)) as resolver, \
                mock.patch.object(ev, "load_test_manifest", return_value=(manifest, {})), \
                mock.patch.object(ev, "test_clip",
                                  return_value=(np.zeros(16000, np.float32), 16000, "hello world")), \
                mock.patch.object(ev, "test_cluster_ids", return_value=("", "spk")), \
                mock.patch.object(ev, "load_model", loader):
            code, out = self._main(argv)
        self.assertEqual(code, 0, out)
        return results, model, loader, resolver, out

    def test_mocked_hub_run(self):
        models = self.tmp / "models"
        for extra, kw, suffix in (([], FALLBACK_KW, "_fb"), (["--no_temperature_fallback"], {}, "")):
            results, model, loader, resolver, out = self._mocked_run(["--model_id", self.HUB, *extra],
                                                                     models)
            resolver.assert_called_once_with(self.HUB, "main")
            loader.assert_called_once_with(self.tmp / "snap")
            self.assertTrue(model.calls)
            for _, call_kw in model.calls:
                self.assertEqual(call_kw, {"language": "en", "task": "transcribe", **kw})
            self.assertIn(f"model source         : {self.HUB} @ {self.COMMIT}", out)
            self.assertIn("LOADING BASE MODEL (zero-shot, openai/whisper-small)", out)
            z = "whisper-small-zeroshot"
            self.assertEqual(sorted(p.name for p in results.iterdir()),
                             [f"eval_{z}{suffix}.csv", f"summary_{z}{suffix}.csv"])
            with open(results / f"summary_{z}{suffix}.csv", newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual([r["group"] for r in rows], ["Overall", "ASR-PART1-Test", "ASR-PART3-Test"])
            for r in rows:
                self.assertEqual((r["model_id"], r["model_revision"], r["model_commit"]),
                                 (self.HUB, "main", self.COMMIT))
            self.assertFalse(models.exists())
            shutil.rmtree(results)

    def test_mocked_fine_tuned_run_unchanged(self):
        models, run = self.tmp / "models", "fake-run"
        (models / run).mkdir(parents=True)
        (models / run / "config.json").write_text("{}", encoding="utf-8")
        results, model, loader, resolver, out = self._mocked_run(["--run_name", run], models)
        resolver.assert_not_called()
        loader.assert_called_once_with(models / run)
        self.assertIn("LOADING FINE-TUNED MODEL", out)
        self.assertNotIn("model source", out)
        with open(results / f"summary_{run}_fb.csv", newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh))
        self.assertEqual(header, ["group", "samples", "mean_WER", "mean_CER", "corpus_WER", "corpus_CER",
                                  "n_pred_gt3x_ref", "n_cr_gt_threshold"])


# ============================================================
# TRANSCRIBE.PY
# ============================================================

class TranscribeTests(TempDirCase):

    def test_parse_args(self):
        a = tx.parse_args(["a.wav"])
        self.assertEqual(a.model, str(Path("models") / "whisper-small-mnsc-sg-15k-heldout"))
        self.assertTrue(a.temperature_fallback)
        self.assertEqual((a.seed, a.compression_ratio_threshold, a.logprob_threshold), (42, 1.35, -1.0))
        self.assertEqual((a.no_repeat_ngram_size, a.max_new_tokens, a.batch_size), (0, 0, 8))
        self.assertEqual((a.out, a.overwrite), (None, False))
        self.assertEqual(tx.decoding_kwargs(a), FALLBACK_KW)
        self.assertEqual(tx.decoding_kwargs(tx.parse_args(["a.wav", "--no_temperature_fallback"])), {})
        self.assertEqual(tx.parse_args(["b.wav", "a.mp3", "c.flac"]).audio, ["b.wav", "a.mp3", "c.flac"])
        for argv in ([], ["a.wav", "--temperature_fallback", "--no_temperature_fallback"],
                     ["a.wav", "--no_temperature_fallback", "--compression_ratio_threshold", "2"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                tx.parse_args(argv)

    def test_resolve_model_dir(self):
        self.assertEqual(tx.resolve_model_dir("models/x"), tx.PROJECT_ROOT / "models" / "x")
        self.assertEqual(tx.resolve_model_dir(str(self.tmp)), self.tmp)
        self.assertEqual(tx.resolve_model_dir(tx.parse_args(["a.wav"]).model),
                         tx.PROJECT_ROOT / "models" / "whisper-small-mnsc-sg-15k-heldout")
        self.assertEqual(tx.PROJECT_ROOT, SCRIPTS.parent)
        source = (SCRIPTS / "transcribe.py").read_text(encoding="utf-8")
        self.assertNotRegex(source, r"[A-Za-z]:\\\\|C:/|/home/|/Users/")

    def test_chunk_audio(self):
        sr = 16000
        self.assertEqual(tx.chunk_audio(np.zeros(0, np.float32), sr), [])
        self.assertEqual(len(tx.chunk_audio(np.zeros(10 * sr), sr)), 1)
        self.assertEqual(len(tx.chunk_audio(np.zeros(30 * sr), sr)), 1)
        audio = np.arange(65 * sr, dtype=np.float32)
        chunks = tx.chunk_audio(audio, sr)
        self.assertEqual([len(c) for c in chunks], [30 * sr, 30 * sr, 5 * sr])
        np.testing.assert_array_equal(np.concatenate(chunks), audio)
        tail = tx.chunk_audio(np.zeros(30 * sr + 800), sr)  # 30 s + 0.05 s
        self.assertEqual([len(c) for c in tail], [30 * sr])
        self.assertEqual([len(c) for c in tx.chunk_audio(np.zeros(800), sr)], [800])
        sr2 = 44100
        audio2 = np.arange(int(61.5 * sr2), dtype=np.float32)
        chunks2 = tx.chunk_audio(audio2, sr2)
        self.assertEqual(len(chunks2), 3)
        self.assertTrue(all(len(c) <= 30 * sr2 for c in chunks2))
        np.testing.assert_array_equal(np.concatenate(chunks2), audio2)

    def test_resample_audio(self):
        x = np.random.default_rng(0).standard_normal(1000).astype(np.float32)
        same = tx.resample_audio(x, 16000)
        np.testing.assert_array_equal(same, x)
        self.assertEqual(same.dtype, np.float32)
        self.assertEqual(len(tx.resample_audio(np.zeros(48000 * 2), 48000)), 32000)
        self.assertEqual(len(tx.resample_audio(np.zeros(8001), 48000)), round(8001 / 3))
        self.assertEqual(len(tx.resample_audio(np.zeros(8000), 8000)), 16000)
        self.assertEqual(len(tx.resample_audio(np.zeros(0, np.float32), 44100)), 0)
        t = np.arange(44100 * 2) / 44100
        y = tx.resample_audio(np.sin(2 * np.pi * 440 * t).astype(np.float32), 44100)
        self.assertEqual(y.dtype, np.float32)
        freqs = np.fft.rfftfreq(len(y), 1 / 16000)
        self.assertAlmostEqual(freqs[np.argmax(np.abs(np.fft.rfft(y)))], 440, delta=2)
        self.assertAlmostEqual(float(np.abs(y).max()), 1.0, delta=0.05)

    def test_load_audio(self):
        stereo = self.tmp / "stereo.wav"
        sf.write(str(stereo), np.stack([np.full(44100, 0.5), np.full(44100, -0.1)], 1)
                 .astype(np.float32), 44100)
        audio, sr = tx.load_audio(stereo)
        self.assertEqual((sr, audio.shape, audio.dtype), (44100, (44100,), np.float32))
        self.assertAlmostEqual(float(audio[100]), 0.2, places=4)
        mono = self.tmp / "mono.flac"
        sf.write(str(mono), np.zeros(16000 * 3, np.float32), 16000)
        audio, sr = tx.load_audio(mono)
        self.assertEqual((sr, len(audio)), (16000, 48000))
        bad = self.tmp / "not_audio.wav"
        bad.write_text("hello", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "ffmpeg"):
            tx.load_audio(bad)

    def _main(self, argv, fake_load):
        out = io.StringIO()
        with mock.patch.object(tx, "load_model", fake_load), redirect_stdout(out), \
                redirect_stderr(io.StringIO()):
            code = tx.main(argv)
        return code, out.getvalue()

    def test_main_end_to_end_mocked(self):
        model_dir = self.tmp / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        listing = sorted(p.name for p in model_dir.iterdir())
        short, long_ = self.tmp / "short.wav", self.tmp / "long.wav"
        sf.write(str(short), np.zeros(16000 * 5, np.float32), 16000)
        sf.write(str(long_), np.zeros(44100 * 40, np.float32), 44100)
        state = {}

        def fake_load(path, device=None):
            state["proc"], state["model"] = FakeProcessor(), FakeModel()
            return state["proc"], state["model"], "cpu", torch.float32

        base = ["--model", str(model_dir), str(short), str(long_)]
        code, out = self._main(base, fake_load)
        self.assertEqual(code, 0)
        lines = out.splitlines()
        self.assertEqual(lines, [f"{short}\tchunk0", f"{long_}\tchunk1 chunk2"])
        self.assertTrue(all(kw["temperature"] == wd.FALLBACK_TEMPERATURES
                            for _, kw in state["model"].calls))
        self.assertEqual(set(state["proc"].sampling_rates), {16000})
        self.assertTrue(all(n <= 30 * 16000 for n in state["proc"].lengths))
        self.assertEqual(sorted(state["proc"].lengths), [5 * 16000, 10 * 16000, 30 * 16000])

        code, _ = self._main(base + ["--no_temperature_fallback"], fake_load)
        self.assertEqual(code, 0)
        self.assertTrue(all("temperature" not in kw for _, kw in state["model"].calls))

        csv_path = self.tmp / "sub" / "out.csv"
        code, _ = self._main(base + ["--out", str(csv_path)], fake_load)
        self.assertEqual(code, 0)
        with open(csv_path, encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(list(rows[0]), tx.CSV_COLUMNS)
        self.assertEqual([r["n_chunks"] for r in rows], ["1", "2"])
        self.assertEqual(rows[1]["decoding"], "temperature fallback (default)")
        self.assertAlmostEqual(float(rows[1]["duration_sec"]), 40.0)
        before = csv_path.read_bytes()

        refuse = mock.Mock(side_effect=AssertionError("must not load"))
        code, _ = self._main(base + ["--out", str(csv_path)], refuse)
        self.assertEqual(code, 2)
        refuse.assert_not_called()
        self.assertEqual(csv_path.read_bytes(), before)

        code, _ = self._main(["--model", str(model_dir), str(short), "--out", str(csv_path),
                              "--overwrite"], fake_load)
        self.assertEqual(code, 0)
        with open(csv_path, encoding="utf-8", newline="") as fh:
            self.assertEqual(len(list(csv.DictReader(fh))), 1)

        bad = self.tmp / "bad.wav"
        bad.write_text("not audio", encoding="utf-8")
        code, out = self._main(["--model", str(model_dir), str(bad), str(short)], fake_load)
        self.assertEqual(code, 1)
        self.assertEqual(out.splitlines(), [f"{short}\tchunk0"])

        self.assertEqual(sorted(p.name for p in model_dir.iterdir()), listing)
        code, _ = self._main(["--model", str(self.tmp / "missing"), str(short)], refuse)
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
