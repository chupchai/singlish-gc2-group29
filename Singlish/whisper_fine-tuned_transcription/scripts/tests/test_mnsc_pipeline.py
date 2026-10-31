"""
Offline tests for the MNSC data pipeline (stdlib unittest, temp dirs only).

Run from the project root:
    python -m unittest discover -s scripts/tests -t . -v
"""

import io
import json
import pickle
import shutil
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import preprocess_mnsc_data as pp  # noqa: E402
import verify_mnsc_dataset as vd   # noqa: E402

SR = 16000


def _wav_bytes(audio_int16: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio_int16, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _attrs_part3(spk1, spk2):
    return {"speaker1": {"speaker_id": spk1, "gender": "M"},
            "speaker2": {"speaker_id": spk2, "gender": "F"}}


def build_fake_source(root: Path) -> dict:
    """
    Fake Hub layout: ASR-PART3-Train (2 shards, 6 speakers, 3..20 clips),
    ASR-PART1-Train (1 shard, 'speaker' block), ASR-PART3-Test.
    Returns {clip_id: (raw answer, int16 audio)}.
    """
    rng = np.random.default_rng(0)
    truth = {}
    counts = {"s1": 20, "s2": 12, "s3": 8, "s4": 6, "s5": 4, "s6": 3}
    order = [s for s, n in counts.items() for _ in range(n)]
    rng.shuffle(order)

    def write(config, shard_rows, attrs_fn, n_shards):
        d = root / config
        d.mkdir(parents=True, exist_ok=True)
        per = int(np.ceil(len(shard_rows) / n_shards))
        for k in range(n_shards):
            chunk = shard_rows[k * per:(k + 1) * per]
            ctx, ans, oth = [], [], []
            for j, spk in enumerate(chunk):
                n = int(rng.integers(SR, 3 * SR))
                audio = rng.integers(-8000, 8000, n).astype(np.int16)
                text = f"<Speaker1>: Hello [lah] (um) {spk} clip {k}-{j} #Sentosa#"
                ctx.append({"bytes": _wav_bytes(audio), "path": f"{k}-{j}.wav"})
                ans.append(text)
                oth.append(attrs_fn(spk))
                truth[f"{config}/{k:05d}:{j}"] = (text, audio)
            pq.write_table(pa.table({
                "context": pa.array(ctx),
                "instruction": pa.array(["transcribe"] * len(chunk)),
                "answer": pa.array(ans),
                "other_attributes": pa.array(oth),
            }), str(d / f"train-{k:05d}-of-{n_shards:05d}.parquet"), row_group_size=7)

    write("ASR-PART3-Train", order, lambda s: _attrs_part3(s, s + "-partner"), 2)
    write("ASR-PART1-Train", [f"p1spk{i % 4}" for i in range(16)],
          lambda s: {"speaker": {"speaker_id": s, "gender": "F"}}, 1)
    write("ASR-PART3-Test", ["t1", "t2"], lambda s: _attrs_part3(s, s + "-x"), 1)
    return truth


def run_main(argv):
    with redirect_stdout(io.StringIO()) as out:
        rc = pp.main(argv)
    return rc, out.getvalue()


def _conv_attrs(conv, spk1, spk2):
    return {"speaker1": {"speaker_id": spk1, "gender": "M"},
            "speaker2": {"speaker_id": spk2, "gender": "F"},
            "conversation_id": conv}


def build_heldout_source(root: Path) -> dict:
    """
    Fake layout for held-out builds:
      ASR-PART3-Train: 2 shards, conversations c01..c10 (4-6 rows each, roles swap);
                       c10's partner is c09's second voice, so c09+c10 form one component.
      ASR-PART3-Test : 2-3 rows each for c01..c06 (candidates) and c09 (not a candidate).
      ASR-PART1-Test : one 'speaker' row whose id is c08's first voice.
    Returns {"train": {clip_id: attrs}, "test_conv": [conversation of PART3-Test row i]}.
    """
    rng = np.random.default_rng(7)
    pair = {f"c{i:02d}": (f"c{i:02d}-a", f"c{i:02d}-b") for i in range(1, 11)}
    pair["c10"] = ("c10-a", "c09-b")
    train_rows = []
    for i, conv in enumerate(sorted(pair)):
        a, b = pair[conv]
        for j in range(4 + i % 3):
            train_rows.append(_conv_attrs(conv, *((a, b) if j % 2 == 0 else (b, a))))
    rng.shuffle(train_rows)

    def write(config, rows, n_shards, prefix):
        d = root / config
        d.mkdir(parents=True, exist_ok=True)
        per = int(np.ceil(len(rows) / n_shards))
        out = {}
        for k in range(n_shards):
            chunk = rows[k * per:(k + 1) * per]
            ctx = []
            for j, attrs in enumerate(chunk):
                audio = rng.integers(-8000, 8000, int(rng.integers(SR, 2 * SR))).astype(np.int16)
                ctx.append({"bytes": _wav_bytes(audio), "path": f"{k}-{j}.wav"})
                out[f"{config}/{k:05d}:{j}"] = attrs
            pq.write_table(pa.table({
                "context": pa.array(ctx),
                "instruction": pa.array(["transcribe"] * len(chunk)),
                "answer": pa.array([f"hello {k} {j}" for j in range(len(chunk))]),
                "other_attributes": pa.array(chunk),
            }), str(d / f"{prefix}-{k:05d}-of-{n_shards:05d}.parquet"), row_group_size=7)
        return out

    train = write("ASR-PART3-Train", train_rows, 2, "train")
    test_rows = []
    for i, conv in enumerate(["c01", "c02", "c03", "c04", "c05", "c06", "c09"]):
        a, b = pair[conv]
        test_rows += [_conv_attrs(conv, a, b)] * (2 + i % 2)
    write("ASR-PART3-Test", test_rows, 2, "test")
    write("ASR-PART1-Test", [{"speaker": {"speaker_id": "c08-a", "gender": "F"}},
                             {"speaker": {"speaker_id": "p1-other", "gender": "M"}}], 1, "test")
    return {"train": train, "test_conv": [r["conversation_id"] for r in test_rows]}


class PipelineTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="mnsc_test_"))
        cls.src = cls.tmp / "src"
        cls.data = cls.tmp / "data"
        cls.truth = build_fake_source(cls.src)
        cls.base_args = ["--data_root", str(cls.data), "--source_dir", str(cls.src),
                         "--rows_per_shard", "8", "--row_group_size", "4"]
        rc, cls.log = run_main(["--data_name", "t1", "--max_samples", "40",
                                "--max_clips_per_speaker", "5", "--val_fraction", "0.34"]
                               + cls.base_args)
        assert rc == 0, cls.log

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # ---------------------------------------------------------------
    def test_flac_roundtrip(self):
        rng = np.random.default_rng(1)
        x16 = rng.integers(-32768, 32767, 3 * SR).astype(np.int16)
        xf = x16.astype(np.float32) / 32768.0
        for src in (x16, xf):
            y, sr = pp.decode_flac(pp.encode_flac(src))
            self.assertEqual(sr, SR)
            self.assertEqual(y.ndim, 1)
            self.assertEqual(y.dtype, np.float32)
            self.assertTrue(np.array_equal(y, xf))
        wav_f, _ = sf.read(io.BytesIO(_wav_bytes(x16)), dtype="float32")
        self.assertTrue(np.array_equal(wav_f, pp.decode_flac(pp.encode_flac(x16))[0]))

    def test_normalize_and_speaker_unchanged(self):
        self.assertEqual(
            pp.normalize_transcript("<Speaker1>: Wah [lah]  {HDB} (uh) #Sentosa# OK  <speaker2>:Yes"),
            "wah lah hdb uh sentosa ok yes")
        self.assertEqual(pp.normalize_transcript("[[nested] x]"), "nested x")
        self.assertEqual(pp.normalize_transcript(""), "")
        row = {"other_attributes": _attrs_part3("A1", "B1")}
        self.assertEqual(pp.extract_speaker_id(row), "A1")
        row = {"other_attributes": {"speaker1": {"speaker_id": " "}, "speaker2": {"speaker_id": "B2"}}}
        self.assertEqual(pp.extract_speaker_id(row), "B2")
        self.assertEqual(pp.extract_speaker_id({}), "unknown")
        part1 = {"other_attributes": {"speaker": {"speaker_id": "01820000"}}}
        self.assertEqual(pp.extract_speaker_id(part1), "unknown")       # unchanged function
        self.assertEqual(pp.extract_speaker_id_any(part1), "01820000")  # PART1 fallback
        self.assertEqual(pp.extract_speaker_id_any(row), "B2")
        self.assertEqual(pp.extract_speaker_id_any({}), "unknown")

    def test_build_end_to_end(self):
        out = self.data / "t1"
        self.assertFalse((out / "_staging").exists())
        info = json.loads((out / pp.INFO_NAME).read_text(encoding="utf-8"))
        self.assertEqual(info["format_version"], 2)
        self.assertGreater(len(info["layout"]["train_shards"]), 1)
        rows = {}
        for split in ("train", "val"):
            n = 0
            for name in info["layout"][f"{split}_shards"]:
                t = pq.read_table(str(out / split / name))
                n += t.num_rows
                for r in t.to_pylist():
                    rows[r["clip_id"]] = (split, r)
            self.assertEqual(n, info["split"][f"{split}_samples"])
        tr = {r["speaker_id"] for s, r in rows.values() if s == "train"}
        va = {r["speaker_id"] for s, r in rows.values() if s == "val"}
        self.assertFalse(tr & va)
        self.assertEqual(tr, set(info["split"]["train_speakers"]))
        self.assertEqual(va, set(info["split"]["val_speakers"]))
        counts = Counter(r["speaker_id"] for _, r in rows.values())
        self.assertLessEqual(max(counts.values()), 5)
        self.assertGreater(info["collection"]["skipped_speaker_cap"], 0)
        # 6 speakers capped at 5 -> 5+5+5+5+4+3 = 27 clips available
        self.assertEqual(len(rows), 27)
        self.assertEqual({r["source_config"] for _, r in rows.values()}, {"ASR-PART3-Train"})
        for cid, (_, r) in rows.items():
            answer, _audio = self.truth[cid]
            self.assertEqual(r["text"], pp.normalize_transcript(answer))
            self.assertNotIn("partner", r["speaker_id"])

    def test_lazy_index_and_reader(self):
        train, val, info = pp.load_preprocessed_index("t1", self.data)
        self.assertEqual(len(train) + len(val), info["collection"]["total_collected"])
        self.assertNotIn("audio", train[0])
        cache = {}
        for item in train + val:
            audio = pp.read_clip_audio(item, cache)
            ref = sf.read(io.BytesIO(_wav_bytes(self.truth[item["clip_id"]][1])), dtype="float32")[0]
            self.assertTrue(np.array_equal(audio, ref), item["clip_id"])
            self.assertEqual(len(audio), item["audio_length"])
        self.assertLessEqual(len(cache), 8)
        recs = list(pp.iter_preprocessed_records("t1", "val", self.data))
        self.assertEqual(len(recs), len(val))
        self.assertEqual(set(recs[0]), {"audio", "sampling_rate", "text", "speaker_id", "duration_sec"})
        with redirect_stdout(io.StringIO()):
            etr, eva, _ = pp.load_preprocessed_data(data_name="t1", data_root=self.data)
        self.assertEqual((len(etr), len(eva)), (len(train), len(val)))

    def test_cap_off_and_part1_mix(self):
        rc, log = run_main(["--data_name", "t_nocap", "--max_samples", "40",
                            "--max_clips_per_speaker", "0", "--val_fraction", "0.34"] + self.base_args)
        self.assertEqual(rc, 0, log)
        tr, va, _ = pp.load_preprocessed_index("t_nocap", self.data)
        self.assertGreater(max(Counter(i["speaker_id"] for i in tr + va).values()), 5)

        rc, log = run_main(["--data_name", "t_mix", "--max_samples", "20", "--part1_fraction", "0.5",
                            "--max_clips_per_speaker", "0", "--val_fraction", "0.34"] + self.base_args)
        self.assertEqual(rc, 0, log)
        info = json.loads((self.data / "t_mix" / pp.INFO_NAME).read_text(encoding="utf-8"))
        self.assertEqual(info["collection"]["per_config"], {"ASR-PART3-Train": 10, "ASR-PART1-Train": 10})
        tr, va, _ = pp.load_preprocessed_index("t_mix", self.data)
        p1 = [i for i in tr + va if i["clip_id"].startswith("ASR-PART1-Train")]
        self.assertEqual(len(p1), 10)
        self.assertTrue(all(i["speaker_id"].startswith("p1spk") for i in p1))

    def test_refuses_test_config_and_bad_names(self):
        with self.assertRaises(ValueError):
            pp.list_source_shards("ASR-PART3-Test", self.src)
        with self.assertRaises(ValueError):
            pp.list_source_shards("ASR-PART1-Test", self.src)
        for bad in ("legacy", "..\\x", "../x", "", ".hidden", "a/b"):
            with self.assertRaises(ValueError, msg=bad):
                pp.dataset_dir(bad, self.data)

    def test_overwrite_guard(self):
        args = ["--data_name", "t_ow", "--max_samples", "10", "--val_fraction", "0.34"] + self.base_args
        self.assertEqual(run_main(args)[0], 0)
        info_path = self.data / "t_ow" / pp.INFO_NAME
        mtime = info_path.stat().st_mtime_ns
        rc, _ = run_main(args)
        self.assertNotEqual(rc, 0)
        self.assertEqual(info_path.stat().st_mtime_ns, mtime)
        self.assertEqual(run_main(args + ["--overwrite"])[0], 0)
        # a folder not created by this script is never deleted
        foreign = self.data / "t_foreign"
        foreign.mkdir(parents=True)
        (foreign / "keep.txt").write_text("x")
        rc, _ = run_main(["--data_name", "t_foreign", "--max_samples", "5", "--overwrite"] + self.base_args)
        self.assertNotEqual(rc, 0)
        self.assertTrue((foreign / "keep.txt").exists())

    def test_isolation_of_default_paths(self):
        d = pp.dataset_dir("part3_15k")
        self.assertEqual(d, (pp.DATA_DIR / "part3_15k").resolve())
        for legacy in (pp.TRAIN_PARQUET, pp.VAL_PARQUET, pp.SPLIT_INFO):
            self.assertEqual(legacy.parent.resolve(), pp.DATA_DIR.resolve())
            self.assertNotEqual(legacy.resolve(), (d / pp.INFO_NAME).resolve())
        self.assertEqual(pp.parse_args([]).data_name, "part3_15k")
        a = pp.parse_args([])
        self.assertEqual((a.max_samples, a.val_fraction, a.seed, a.max_clips_per_speaker, a.part1_fraction),
                         (15000, 0.1, 42, 50, 0.0))

    def test_split_deterministic_across_hash_seeds(self):
        import os
        import subprocess
        code = ("import sys, io, contextlib; sys.path.insert(0, r'%s'); import preprocess_mnsc_data as p; "
                "recs=[{'speaker_id': 'spk%%03d' %% i} for i in range(200)]; f=io.StringIO(); "
                "ctx=contextlib.redirect_stdout(f); ctx.__enter__(); "
                "v=p.speaker_split(recs, 0.1, 42)[2]['val_speakers']; ctx.__exit__(None, None, None); "
                "print(','.join(v))" % SCRIPTS)
        outs = set()
        for seed in ("0", "1", "123"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            outs.add(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                                    text=True, check=True).stdout.strip())
        self.assertEqual(len(outs), 1, outs)

    def test_verify_script(self):
        with redirect_stdout(io.StringIO()) as out:
            rc = vd.main(["--data_name", "t1", "--data_root", str(self.data), "--n_decode", "10"])
        self.assertEqual(rc, 0, out.getvalue())
        self.assertIn("PASS", out.getvalue())

    # ---------------------------------------------------------------
    def test_training_helpers(self):
        import train_whisper_speaker_split as tw
        self.assertEqual(tw.derive_run_name({"collection": {"total_collected": 15000}}, -1),
                         "whisper-small-mnsc-sg-15k")
        self.assertEqual(tw.derive_run_name({"collection": {"total_collected": 500}}, 20),
                         "whisper-small-mnsc-sg-500-smoke")
        self.assertEqual(tw.derive_run_name({"collection": {"total_collected": 14918},
                                             "heldout_test": {"n_conversations": 40}}, -1),
                         "whisper-small-mnsc-sg-15k-heldout")
        self.assertEqual(tw.derive_run_name({"collection": {"total_collected": 14918},
                                             "heldout_test": {"n_conversations": 40}}, 10),
                         "whisper-small-mnsc-sg-15k-heldout-smoke")
        conv = {"split": {"split_strategy": "conversation_component"}}
        self.assertEqual(tw.derive_run_name(dict(conv, collection={"total_collected": 15000}), -1),
                         "whisper-small-mnsc-sg-15k-conv")
        self.assertEqual(tw.derive_run_name(dict(conv, collection={"total_collected": 14918},
                                                 heldout_test={"n_conversations": 40}), -1),
                         "whisper-small-mnsc-sg-15k-heldout")
        self.assertEqual(tw.derive_run_name({"collection": {"total_collected": 15000},
                                             "split": {"split_strategy": "speaker_level"}}, -1),
                         "whisper-small-mnsc-sg-15k")
        self.assertEqual(tw.compute_eval_steps(13500, 0), (1688, 844))
        self.assertEqual(tw.compute_eval_steps(13500, 10), (1688, 10))
        a = tw.parse_args([])
        self.assertEqual((a.data_name, a.num_workers, a.max_steps), ("part3_15k", 0, -1))

    def test_whisper_dataset_lazy(self):
        import train_whisper_speaker_split as tw
        from transformers import WhisperProcessor
        try:
            proc = WhisperProcessor.from_pretrained("openai/whisper-small", language="en",
                                                    task="transcribe", local_files_only=True)
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"whisper-small processor not cached: {exc}")
        train, _, _ = pp.load_preprocessed_index("t1", self.data)
        ds = tw.WhisperMNSCDataset(train, proc)
        for idx in (0, len(ds) - 1):
            item = train[idx]
            out = ds[idx]
            ref = sf.read(io.BytesIO(_wav_bytes(self.truth[item["clip_id"]][1])), dtype="float32")[0]
            ref_feat = proc.feature_extractor(ref, sampling_rate=SR).input_features[0]
            self.assertEqual(np.asarray(out["input_features"]).shape, (80, 3000))
            self.assertTrue(np.allclose(out["input_features"], ref_feat))
            self.assertEqual(out["labels"], proc.tokenizer(item["text"]).input_ids)
        ds2 = pickle.loads(pickle.dumps(ds))
        self.assertEqual(ds2._cache, {})
        self.assertTrue(np.allclose(ds2[0]["input_features"], ds[0]["input_features"]))
        # legacy in-memory items still work
        legacy = [{"audio": ref, "sampling_rate": SR, "text": "x"}]
        self.assertEqual(np.asarray(tw.WhisperMNSCDataset(legacy, proc)[0]["input_features"]).shape,
                         (80, 3000))

    def test_eval_helpers(self):
        import pandas as pd
        import evaluate_whisper_speaker_split as ev
        self.assertEqual(ev.output_suffix(0, 0, 0), "")
        self.assertEqual(ev.output_suffix(16, 3, 225), "_rows16_nrng3_mnt225")
        o, s = ev.output_paths("whisper-small-mnsc-sg-15k", "")
        self.assertEqual((o.name, s.name), ("eval_whisper-small-mnsc-sg-15k.csv",
                                            "summary_whisper-small-mnsc-sg-15k.csv"))
        self.assertNotIn(o.name, ("whisper_small_mnsc_extended_evaluation.csv",))
        m = pd.DataFrame({"config": ["ASR-PART1-Test"] * 10 + ["ASR-PART3-Test"] * 10,
                          "sample_id": range(20)})
        sub = ev.subset_manifest(m, 7)
        self.assertEqual(Counter(sub["config"]), {"ASR-PART1-Test": 4, "ASR-PART3-Test": 3})
        df = pd.DataFrame({"reference": ["a b", "a"], "prediction": ["a b c d e f g", "a"]})
        self.assertEqual(ev.n_runaway(df), 1)
        a = ev.parse_args([])
        self.assertEqual((a.no_repeat_ngram_size, a.max_new_tokens, a.batch_size), (0, 0, 8))

    def test_eval_fallback_helpers(self):
        import torch
        from transformers import WhisperForConditionalGeneration
        import evaluate_whisper_speaker_split as ev
        self.assertEqual(ev.output_suffix(0, 5, 0, True), "_nrng5_fb")
        self.assertEqual(ev.output_suffix(0, 0, 0, True), "_fb")
        self.assertEqual(ev.output_suffix(0, 0, 0, True, 1.5), "_fb_cr1.5")
        self.assertEqual(ev.output_suffix(0, 0, 0, True, 1.35, -0.5, 7), "_fb_lp-0.5_seed7")
        self.assertEqual(ev.output_suffix(0, 3, 0, False, 1.5, -0.5, 7), "_nrng3")
        self.assertEqual(ev.output_paths("whisper-small-mnsc-sg-15k", "_fb")[0].name,
                         "eval_whisper-small-mnsc-sg-15k_fb.csv")
        a = ev.parse_args([])
        self.assertEqual((a.temperature_fallback, a.seed), (True, 42))
        self.assertEqual(ev.decoding_kwargs(a), {"temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                                                 "compression_ratio_threshold": 1.35,
                                                 "logprob_threshold": -1.0})
        kw = ev.decoding_kwargs(ev.parse_args(["--temperature_fallback", "--no_repeat_ngram_size", "5"]))
        self.assertEqual(kw, {"no_repeat_ngram_size": 5, "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                              "compression_ratio_threshold": 1.35, "logprob_threshold": -1.0})
        self.assertNotIn("do_sample", kw)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            ev.parse_args(["--no_temperature_fallback", "--compression_ratio_threshold", "1.5"])
        t = torch.tensor([[1, 2]])
        self.assertIs(ev.generated_sequences(t), t)
        self.assertIs(ev.generated_sequences({"sequences": t}), t)
        loop = [100, 200] * 100
        self.assertGreater(ev.token_compression_ratio(loop, 51865, 50257), 1.35)
        self.assertLess(ev.token_compression_ratio([1, 2, 3], 51865, 50257), 1.35)
        self.assertEqual(ev.token_compression_ratio([50258, 50257, 50360, 50364], 51865, 50257), 0.0)
        # HF checks text + one EOS; returned rows may carry EOS padding or no EOS at all
        hf = WhisperForConditionalGeneration._retrieve_compression_ratio(
            torch.tensor([11, 22, 333, 4444, 11, 22, 333, 4444, 50257]), 51865)
        for ids in ([11, 22, 333, 4444, 11, 22, 333, 4444, 50257, 50257, 50257],
                    [11, 22, 333, 4444, 11, 22, 333, 4444]):
            self.assertAlmostEqual(ev.token_compression_ratio(ids, 51865, 50257), hf)

    def test_legacy_parquet_loader(self):
        rng = np.random.default_rng(3)
        audios = [rng.standard_normal(n).astype(np.float32) for n in (100, 250, 50)]
        path = self.tmp / "legacy_val.parquet"
        pq.write_table(pa.table({
            "audio_bytes": pa.array([a.tobytes() for a in audios], type=pa.large_binary()),
            "audio_length": pa.array([len(a) for a in audios], type=pa.int32()),
            "sampling_rate": pa.array([SR] * 3, type=pa.int32()),
            "text": pa.array(["a b", "c", "d e f"]),
            "speaker_id": pa.array(["s1", "s1", "s2"]),
            "duration_sec": pa.array([len(a) / SR for a in audios]),
        }), str(path))
        got = pp.load_legacy_parquet(path)
        want = pp.load_preprocessed_data(train_path=path, val_path=path,
                                         info_path=self.tmp / "missing.json")[1]
        self.assertEqual(len(got), 3)
        for g, w, a in zip(got, want, audios):
            self.assertEqual(set(g), set(w))
            self.assertTrue(np.array_equal(g["audio"], w["audio"]))
            self.assertTrue(np.array_equal(g["audio"], a))
            self.assertEqual((g["text"], g["speaker_id"], g["duration_sec"], g["sampling_rate"]),
                             (w["text"], w["speaker_id"], w["duration_sec"], w["sampling_rate"]))
        with self.assertRaises(FileNotFoundError):
            pp.load_legacy_parquet(self.tmp / "nope.parquet")

    def test_legacy_val_lazy_loader(self):
        import evaluate_whisper_speaker_split as ev
        rng = np.random.default_rng(5)
        audios = [rng.standard_normal(n).astype(np.float32) for n in (100, 250, 50, 0, 7)]
        table = pa.table({
            "audio_bytes": pa.array([a.tobytes() for a in audios], type=pa.large_binary()),
            "audio_length": pa.array([len(a) for a in audios], type=pa.int32()),
            "sampling_rate": pa.array([SR] * 5, type=pa.int32()),
            "text": pa.array(["a b", "c", "d e f", "", "g"]),
            "speaker_id": pa.array(["s1", "s1", "s2", "s3", "s3"]),
            "duration_sec": pa.array([len(a) / SR for a in audios]),
        })

        def write(name, **kw):
            path = self.tmp / name
            pq.write_table(table, str(path), **kw)
            return path

        def load(path, keep_rows=0):
            out = io.StringIO()
            with redirect_stdout(out):
                items = ev.load_legacy_val_lazy(path, keep_rows=keep_rows)
            return items, out.getvalue()

        def check(items, path, rows=5):
            want = pp.load_legacy_parquet(path)
            self.assertEqual(len(items), len(want))
            for item, w in list(zip(items, want))[:rows]:
                audio, sr, text = ev.val_clip(item, {})
                self.assertTrue(np.array_equal(audio, w["audio"]))
                self.assertEqual(audio.tobytes(), w["audio"].tobytes())
                self.assertEqual((audio.dtype, audio.shape), (w["audio"].dtype, w["audio"].shape))
                for flag in ("owndata", "writeable", "c_contiguous"):
                    self.assertEqual(getattr(audio.flags, flag), getattr(w["audio"].flags, flag), flag)
                self.assertIs(type(sr), int)
                self.assertIs(type(item["duration_sec"]), float)
                self.assertEqual((sr, text, item["speaker_id"], item["duration_sec"]),
                                 (w["sampling_rate"], w["text"], w["speaker_id"], w["duration_sec"]))

        # a/b: default writer (dictionary + snappy, the real file's layout) and uncompressed
        for name, kw in (("lazy_snappy.parquet", {}), ("lazy_none.parquet", {"compression": "NONE"})):
            path = write(name, **kw)
            items, out = load(path)
            self.assertNotIn("WARNING", out, name)
            self.assertTrue(all("audio" not in it and "legacy_audio" in it for it in items))
            check(items, path)

        # c: keep_rows trims the audio but keeps every item's metadata
        path = self.tmp / "lazy_snappy.parquet"
        items, _ = load(path, keep_rows=2)
        self.assertEqual(len(items), 5)
        check(items, path, rows=2)
        with self.assertRaises(IndexError):
            ev.val_clip(items[2], {})

        # d: other layouts fall back to the eager loader
        for name, kw in (("lazy_plain.parquet", {"use_dictionary": False}),
                         ("lazy_rg2.parquet", {"row_group_size": 2})):
            path = write(name, **kw)
            items, out = load(path)
            self.assertIn("WARNING", out, name)
            self.assertIn("falling back", out, name)
            self.assertTrue(all("audio" in it for it in items))
            check(items, path)

        # e: missing file
        with self.assertRaises(FileNotFoundError):
            ev.load_legacy_val_lazy(self.tmp / "nope.parquet")

        # f: manifest ids/order unchanged, --max_rows = head of the full manifest
        with mock.patch.object(ev, "VAL_PARQUET", self.tmp / "lazy_snappy.parquet"), \
                mock.patch.object(ev, "SPLIT_INFO", self.tmp / "missing.json"), \
                redirect_stdout(io.StringIO()):
            full, items = ev.load_val_manifest("legacy")
            head, _ = ev.load_val_manifest("legacy", max_rows=2)
        self.assertEqual(full["sample_id"].tolist(), [f"legacy-val_{i:06d}" for i in range(5)])
        self.assertEqual(full["row_index"].tolist(), list(range(5)))
        self.assertTrue((full["config"] == "legacy-val").all())
        self.assertTrue(head.equals(full.head(2)))
        self.assertEqual(len(items), 5)

    def test_eval_split_args(self):
        import evaluate_whisper_speaker_split as ev
        import train_whisper_speaker_split as tw
        a = ev.parse_args([])
        self.assertEqual((a.split, a.data_name, a.data_root), ("test", None, None))
        self.assertEqual(ev.parse_args(["--split", "val"]).data_name, "part3_15k")
        self.assertEqual(ev.parse_args(["--split", "val", "--run_name", "whisper-small-mnsc-sg-4k"]).data_name,
                         "legacy")
        a = ev.parse_args(["--split", "val", "--data_name", "t1", "--data_root", str(self.data)])
        self.assertEqual((a.data_name, a.data_root), ("t1", str(self.data)))
        for bad in (["--data_name", "part3_15k"], ["--data_root", "x"],
                    ["--split", "val", "--data_name", "legacy", "--data_root", "x"],
                    ["--split", "val", "--data_name", "../x"], ["--split", "bogus"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit, msg=bad):
                ev.parse_args(bad)
        r15, r4 = "whisper-small-mnsc-sg-15k", "whisper-small-mnsc-sg-4k"
        # held-out retrain: own val split, --heldout_test is test-only
        self.assertEqual(ev.parse_args(["--split", "val", "--run_name", ev.HELDOUT_RUN_NAME]).data_name,
                         ev.HELDOUT_DATA_NAME)
        self.assertEqual(ev.split_tag("val", ev.HELDOUT_RUN_NAME, ev.HELDOUT_DATA_NAME), "_val")
        self.assertEqual(ev.parse_args(["--heldout_test", "part3_15k_heldout"]).heldout_test, "part3_15k_heldout")
        self.assertIsNone(ev.parse_args([]).heldout_test)
        self.assertEqual(ev.heldout_tag(None), "")
        self.assertEqual(ev.output_suffix(20, 0, 0, True, split_tag=ev.heldout_tag("part3_15k_heldout")),
                         "_ho-part3_15k_heldout_rows20_fb")
        for bad in (["--heldout_test", "x", "--split", "val"], ["--heldout_test", "../x"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit, msg=bad):
                ev.parse_args(bad)
        self.assertEqual(ev.split_tag("test", r15, None), "")
        self.assertEqual(ev.split_tag("val", r15, "part3_15k"), "_val")
        self.assertEqual(ev.split_tag("val", r4, "legacy"), "_val")
        self.assertEqual(ev.split_tag("val", r4, "part3_15k"), "_val-part3_15k")
        self.assertEqual(ev.split_tag("val", r15, "legacy"), "_val-legacy")
        self.assertEqual(ev.output_suffix(0, 5, 0, True, split_tag="_val"), "_val_nrng5_fb")
        self.assertEqual(ev.output_suffix(16, 0, 0, split_tag="_val"), "_val_rows16")
        self.assertEqual(ev.output_suffix(0, 0, 0, True, 1.35, -999, split_tag="_val"), "_val_fb_lp-999")
        self.assertEqual(ev.output_paths(r15, "_val")[0].name, "eval_whisper-small-mnsc-sg-15k_val.csv")
        self.assertEqual(ev.output_paths(r15, "_val_nrng5_fb")[1].name,
                         "summary_whisper-small-mnsc-sg-15k_val_nrng5_fb.csv")
        # val names never collide with test names
        for run in (r15, r4, ev.HELDOUT_RUN_NAME):
            test_names, val_names, ho_names, hx_names = set(), set(), set(), set()
            for rows in (0, 16):
                for nrng in (0, 3, 5, 7, 10):
                    for fb in (False, True):
                        for lp in (-1.0, -999):
                            test_names.update(p.name for p in ev.output_paths(
                                run, ev.output_suffix(rows, nrng, 0, fb, 1.35, lp)))
                            for tag in ("_val", "_val-part3_15k", "_val-legacy"):
                                val_names.update(p.name for p in ev.output_paths(
                                    run, ev.output_suffix(rows, nrng, 0, fb, 1.35, lp, split_tag=tag)))
                            ho_names.update(p.name for p in ev.output_paths(
                                run, ev.output_suffix(rows, nrng, 0, fb, 1.35, lp,
                                                      split_tag="_ho-part3_15k_heldout")))
                            hx_names.update(p.name for p in ev.output_paths(
                                run, ev.output_suffix(rows, nrng, 0, fb, 1.35, lp, split_tag=ev.heldout_extra_tag(
                                    "part3_15k_heldout_extra_test"))))
            self.assertFalse(test_names & val_names)
            self.assertFalse(ho_names & (test_names | val_names))
            self.assertFalse(hx_names & (test_names | val_names | ho_names))
            self.assertTrue(all("_hx-part3_15k_heldout_extra_test" in n for n in hx_names))
            self.assertTrue(all("_val" in n for n in val_names))
            self.assertFalse(any("_val" in n for n in test_names))
        self.assertEqual(ev.summary_groups("test", None), ["ASR-PART1-Test", "ASR-PART3-Test"])
        self.assertEqual(ev.summary_groups("val", "part3_15k"), ["part3_15k-val"])
        self.assertEqual(ev.heldout_extra_tag(None), "")
        self.assertEqual(ev.heldout_extra_tag("x"), "_hx-x")
        self.assertIsNone(ev.parse_args([]).heldout_extra_test)
        self.assertEqual(ev.LEGACY_RUN_NAME, tw.LEGACY_RUN_NAME)
        self.assertEqual(ev.DEFAULT_DATA_NAME, tw.DEFAULT_DATA_NAME)
        # LR / seed sweep runs of the held-out model (and their smoke- versions) use its val split
        for run in ("whisper-small-mnsc-sg-15k-heldout-seed43",
                    "whisper-small-mnsc-sg-15k-heldout-lr1e-5-wu10",
                    "smoke-whisper-small-mnsc-sg-15k-heldout-lr3e-5-wu10",
                    "smoke-" + ev.HELDOUT_RUN_NAME):
            self.assertEqual(ev.default_val_data_name(run), ev.HELDOUT_DATA_NAME, run)
            self.assertEqual(ev.parse_args(["--split", "val", "--run_name", run]).data_name,
                             ev.HELDOUT_DATA_NAME)
            self.assertEqual(ev.split_tag("val", run, ev.HELDOUT_DATA_NAME), "_val")
        for run in (r15, "smoke-whisper-mnsc", "whisper-small-mnsc-sg-15k-heldoutX",
                    "whisper-small-mnsc-sg-15k-conv"):
            self.assertEqual(ev.default_val_data_name(run), "part3_15k", run)
        self.assertEqual(ev.default_val_data_name(r4), "legacy")

    def test_eval_val_loading(self):
        import evaluate_whisper_speaker_split as ev
        m, items = ev.load_val_manifest("t1", str(self.data))
        self.assertEqual(len(m), len(pp.load_preprocessed_index("t1", self.data)[1]))
        self.assertEqual(list(m.columns), ["sample_id", "dataset", "config", "row_index"])
        self.assertEqual(set(m["config"]), {"t1-val"})
        self.assertEqual(set(m["dataset"]), {"MNSC-Extended"})
        self.assertEqual(m["row_index"].tolist(), list(range(len(m))))
        self.assertEqual(m["sample_id"].tolist(), [i["clip_id"] for i in items])
        self.assertTrue(ev.load_val_manifest("t1", str(self.data), 3)[0].equals(m.head(3)))
        cache = {}
        for item in items:
            audio, sr, text = ev.val_clip(item, cache)
            raw, wav = self.truth[item["clip_id"]]
            ref = sf.read(io.BytesIO(_wav_bytes(wav)), dtype="float32")[0]
            self.assertTrue(np.array_equal(audio, ref), item["clip_id"])
            self.assertEqual(sr, SR)
            self.assertEqual(ev.normalize_for_evaluation(text), ev.normalize_for_evaluation(raw))
        self.assertLessEqual(len(cache), 8)
        arr = np.zeros(10, dtype=np.float32)
        self.assertIs(ev.val_clip({"audio": arr, "sampling_rate": SR, "text": "x"}, {})[0], arr)
        self.assertTrue(ev.data_source_label("val", "t1", str(self.data)).endswith("(lazy FLAC)"))

    def test_compare_decoding_runs(self):
        import pandas as pd
        import compare_decoding_runs as cd
        import evaluate_whisper_speaker_split as ev
        cols = ["sample_id", "dataset", "config", "row_index", "reference", "prediction", "WER", "CER"]
        base = pd.DataFrame([
            ["a", "D", "X", 0, "one two three four", "one two", 0.5, 0.4],
            ["b", "D", "X", 1, "five six", "five six", 0.0, 0.0],
            ["c", "D", "X", 2, "seven eight", "", 1.0, 1.0],
            ["d", "D", "Y", 0, "nine", "nine", 0.0, 0.0],
        ], columns=cols)
        new = pd.DataFrame([
            ["a", "D", "X", 0, "one two three four", "one two three four", 0.0, 0.0],
            ["b", "D", "X", 1, "five six", "five", 0.5, 0.3],
            ["c", "D", "X", 2, "seven eight", "", 1.0, 1.0],
        ], columns=cols)
        a, b = self.tmp / "cmp_a.csv", self.tmp / "cmp_b.csv"
        base.to_csv(a, index=False)
        new.to_csv(b, index=False)
        ra = cd.load_run(a)
        self.assertEqual(ra.loc[2, "prediction"], "")
        s = cd.summarize(ra)
        self.assertEqual(s["samples"].tolist(), [4, 3, 1])
        self.assertAlmostEqual(s.loc[0, "corpus_WER"], ev.corpus_metric(ra, "WER"))
        ch = cd.row_changes(ra, cd.load_run(b))
        o = ch["table"].iloc[0]
        self.assertEqual((o["matched"], o["better"], o["worse"], o["same"]), (3, 1, 1, 1))
        self.assertAlmostEqual(o["wer_gain"], 0.5)
        self.assertAlmostEqual(o["wer_loss"], 0.5)
        self.assertEqual((ch["n_unmatched_base"], ch["n_unmatched_new"], ch["n_reference_mismatch"]), (1, 0, 0))
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cd.main([str(a), str(b)]), 0)
        self.assertIn("WARNING", out.getvalue())
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cd.main([str(a)])
        with redirect_stdout(io.StringIO()):
            self.assertEqual(cd.main([str(a), str(self.tmp / "missing.csv")]), 2)
        # new eval CSVs carry conversation_id / speaker_id: same numbers
        new2 = new.assign(conversation_id=["c1", "c1", "c2"], speaker_id=["s1", "s1", "s2"])
        b2 = self.tmp / "cmp_b2.csv"
        new2.to_csv(b2, index=False)
        ch2 = cd.row_changes(ra, cd.load_run(b2))
        pd.testing.assert_frame_equal(ch2["table"], ch["table"])
        self.assertEqual(cd.summarize(cd.load_run(b2))["corpus_WER"].tolist(),
                         cd.summarize(cd.load_run(b))["corpus_WER"].tolist())
        with redirect_stdout(io.StringIO()):
            self.assertEqual(cd.main([str(a), str(b2)]), 0)
            self.assertEqual(cd.main([str(b2), str(a)]), 0)


class HeldoutTests(unittest.TestCase):
    """--heldout_test_conversations / --split_by_conversation and the held-out audit/eval."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="mnsc_heldout_"))
        cls.src = cls.tmp / "src"
        cls.data = cls.tmp / "data"
        cls.fx = build_heldout_source(cls.src)
        cls.base = ["--data_root", str(cls.data), "--source_dir", str(cls.src),
                    "--rows_per_shard", "8", "--row_group_size", "4",
                    "--max_samples", "0", "--max_clips_per_speaker", "0", "--val_fraction", "0.3"]
        rc, cls.log = run_main(["--data_name", "ho", "--heldout_test_conversations", "3",
                                "--max_source_shards", "2"] + cls.base)
        assert rc == 0, cls.log
        cls.out = cls.data / "ho"
        cls.info = json.loads((cls.out / pp.INFO_NAME).read_text(encoding="utf-8"))
        cls.manifest = json.loads((cls.out / pp.HELDOUT_MANIFEST_NAME).read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def default_build(self):
        """Same source, no new flags -> data/ho_default (built once per class)."""
        if not (self.data / "ho_default" / pp.INFO_NAME).exists():
            rc, log = run_main(["--data_name", "ho_default"] + self.base)
            self.assertEqual(rc, 0, log)

    def clips(self, name):
        """{clip_id: split} of a built dataset."""
        tr, va, _ = pp.load_preprocessed_index(name, self.data)
        return {**{i["clip_id"]: "train" for i in tr}, **{i["clip_id"]: "val" for i in va}}

    # ---------------------------------------------------------------
    def test_selection(self):
        with redirect_stdout(io.StringIO()):
            m = pp.select_heldout_conversations(3, 42, self.src)
            m2 = pp.select_heldout_conversations(3, 42, self.src)
        self.assertEqual(m["conversations"], m2["conversations"])
        self.assertEqual(m["n_candidates"], 6)
        self.assertEqual(len(m["conversations"]), 3)
        self.assertNotIn("c09", m["conversations"])
        self.assertTrue(set(m["conversations"]) <= {f"c0{i}" for i in range(1, 7)})
        want = [i for i, c in enumerate(self.fx["test_conv"]) if c in m["conversations"]]
        self.assertEqual(m["part3_test_row_indices"], want)
        self.assertEqual(m["n_part3_test_rows"], len(want))
        self.assertEqual(set(m["voices"]), {f"{c}-{x}" for c in m["conversations"] for x in "ab"})
        self.assertEqual(m["part1_test_voices_count"], 2)
        with self.assertRaises(ValueError):
            pp.select_heldout_conversations(7, 42, self.src)
        with self.assertRaises(ValueError):
            pp.list_test_shards("ASR-PART3-Train", self.src)
        self.assertEqual(pp.row_voices({"speaker1": {"speaker_id": " b "}, "speaker2": '{"speaker_id": "a"}',
                                        "speaker": {"speaker_id": "unknown"}}), ["a", "b"])
        self.assertEqual(pp.row_conversation(json.dumps({"conversation_id": " 7 "})), "7")

    def test_heldout_build_excludes(self):
        held_c, held_v = set(self.manifest["conversations"]), set(self.manifest["voices"])
        clips = self.clips("ho")
        self.assertTrue(clips)
        for cid in clips:
            attrs = self.fx["train"][cid]
            self.assertNotIn(attrs["conversation_id"], held_c, cid)
            self.assertFalse(set(pp.row_voices(attrs)) & held_v, cid)
            self.assertNotEqual(attrs["conversation_id"], "c08", cid)  # PART1-Test voice
        rows = list(self.fx["train"].values())
        n_conv = sum(1 for a in rows if a["conversation_id"] in held_c)
        n_p1 = sum(1 for a in rows if a["conversation_id"] == "c08")
        col = self.info["collection"]
        self.assertEqual(col["skipped_heldout_conversation"], n_conv)
        self.assertEqual(col["skipped_heldout_voice"], 0)
        self.assertEqual(col["skipped_part1_test_voice"], n_p1)
        self.assertEqual(col["excluded_train_conversations"], 4)
        self.assertEqual(len(clips), len(rows) - n_conv - n_p1)
        self.assertEqual(self.info["heldout_test"]["excluded_train_rows"], n_conv + n_p1)
        self.assertEqual((self.info["heldout_test_conversations"], self.info["split_by_conversation"],
                          self.info["max_source_shards"]), (3, True, 2))
        with redirect_stdout(io.StringIO()):
            m = pp.select_heldout_conversations(3, 42, self.src, 2)
        self.assertEqual(self.manifest, {k: v for k, v in m.items() if not k.startswith("_")})
        # parquet columns unchanged by the in-RAM group keys
        t = pq.read_table(str(self.out / "train" / self.info["layout"]["train_shards"][0]))
        self.assertEqual(set(t.column_names), set(pp.META_COLUMNS) | {"audio_flac"})

    def test_heldout_split_disjoint(self):
        sides = {"train": (set(), set()), "val": (set(), set())}
        conv_side = {}
        for cid, side in self.clips("ho").items():
            attrs = self.fx["train"][cid]
            sides[side][0].update(pp.row_voices(attrs))
            sides[side][1].add(attrs["conversation_id"])
            conv_side[attrs["conversation_id"]] = side
        self.assertFalse(sides["train"][0] & sides["val"][0])
        self.assertFalse(sides["train"][1] & sides["val"][1])
        self.assertEqual(conv_side["c09"], conv_side["c10"])
        split = self.info["split"]
        self.assertEqual(split["split_strategy"], "conversation_component")
        self.assertGreaterEqual(split["n_multi_conversation_components"], 1)
        self.assertEqual(set(split["train_voices"]), sides["train"][0])
        self.assertEqual(set(split["val_conversations"]), sides["val"][1])

    def test_conversation_split_deterministic_across_hash_seeds(self):
        import os
        import subprocess
        code = ("import sys, io, contextlib; sys.path.insert(0, r'%s'); import preprocess_mnsc_data as p; "
                "recs=[{'clip_id': str(i), 'speaker_id': 'v%%03d' %% (i %% 60), "
                "'conversation_id': 'c%%02d' %% ((i %% 60) // 2), 'voices': ['v%%03d' %% (i %% 60)]} "
                "for i in range(240)]; f=io.StringIO(); "
                "ctx=contextlib.redirect_stdout(f); ctx.__enter__(); "
                "v=p.conversation_split(recs, 0.1, 42)[2]['val_conversations']; "
                "ctx.__exit__(None, None, None); print(','.join(v))" % SCRIPTS)
        outs = set()
        for seed in ("0", "1", "123"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            outs.add(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                                    text=True, check=True).stdout.strip())
        self.assertEqual(len(outs), 1, outs)
        self.assertTrue(next(iter(outs)))

    def test_default_build_unchanged_keys(self):
        self.default_build()
        info = json.loads((self.data / "ho_default" / pp.INFO_NAME).read_text(encoding="utf-8"))
        for key in ("heldout_test", "heldout_test_conversations", "split_by_conversation",
                    "max_source_shards"):
            self.assertNotIn(key, info)
        self.assertNotIn("skipped_heldout_voice", info["collection"])
        self.assertEqual(info["split"]["split_strategy"], "speaker_level")
        self.assertFalse((self.data / "ho_default" / pp.HELDOUT_MANIFEST_NAME).exists())
        convs = {self.fx["train"][c]["conversation_id"] for c in self.clips("ho_default")}
        self.assertTrue(set(self.manifest["conversations"]) <= convs)

    def test_split_by_conversation_only(self):
        rc, log = run_main(["--data_name", "ho_conv", "--split_by_conversation"] + self.base)
        self.assertEqual(rc, 0, log)
        info = json.loads((self.data / "ho_conv" / pp.INFO_NAME).read_text(encoding="utf-8"))
        self.assertEqual(info["split"]["split_strategy"], "conversation_component")
        self.assertNotIn("heldout_test", info)
        self.assertFalse(set(info["split"]["train_voices"]) & set(info["split"]["val_voices"]))
        # Must never derive (and resume) the speaker-split run of the same size.
        import train_whisper_speaker_split as tw
        self.assertTrue(tw.derive_run_name(info, -1).endswith("-conv"))
        big = dict(info, collection={"total_collected": 15000})
        self.assertEqual(tw.derive_run_name(big, -1), "whisper-small-mnsc-sg-15k-conv")

    def test_remote_attrs_read_never_downloads_full_shard(self):
        # Uncached remote shard -> column read over HfFileSystem, no hf_hub_download.
        shard = sorted((self.src / "ASR-PART3-Test").glob("*.parquet"))[0]
        expected = pp.read_attrs_column([str(shard)], local=True)

        class FakeFS:
            def open(self, path, mode="rb"):
                assert path == f"datasets/{pp.DATASET_ID}/ASR-PART3-Test/x.parquet", path
                return open(shard, mode)

        with mock.patch("huggingface_hub.try_to_load_from_cache", return_value=None), \
                mock.patch("huggingface_hub.HfFileSystem", FakeFS), \
                mock.patch("huggingface_hub.hf_hub_download",
                           side_effect=AssertionError("full download")):
            got = pp.read_attrs_column(["ASR-PART3-Test/x.parquet"], local=False)
        self.assertEqual(got, expected)
        # Cached remote shard -> read from the cache path.
        with mock.patch("huggingface_hub.try_to_load_from_cache", return_value=str(shard)), \
                mock.patch("huggingface_hub.HfFileSystem", side_effect=AssertionError("remote")), \
                mock.patch("huggingface_hub.hf_hub_download",
                           side_effect=AssertionError("full download")):
            self.assertEqual(pp.read_attrs_column(["ASR-PART3-Test/x.parquet"], local=False),
                             expected)

    def test_arg_errors(self):
        for bad in (["--heldout_test_conversations", "-1"], ["--max_source_shards", "-2"],
                    ["--heldout_test_conversations", "5", "--part1_fraction", "0.1"],
                    ["--split_by_conversation", "--part1_fraction", "0.1"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit, msg=bad):
                pp.parse_args(bad)
        rc, log = run_main(["--data_name", "ho_toomany", "--heldout_test_conversations", "9"] + self.base)
        self.assertEqual(rc, 2, log)
        self.assertFalse((self.data / "ho_toomany").exists())

    def test_verify_script(self):
        with redirect_stdout(io.StringIO()) as out:
            rc = vd.main(["--data_name", "ho", "--data_root", str(self.data), "--n_decode", "5"])
        self.assertEqual(rc, 0, out.getvalue())

    def test_audit(self):
        import audit_test_speaker_overlap as au
        res = au.audit_dataset(self.out, self.src)
        self.assertTrue(all(v == 0 for v in res["required"].values()), res["required"])
        self.assertTrue(all(res["consistency"].values()), res["consistency"])
        self.assertEqual(res["info"]["heldout_part3_test_rows"], self.manifest["n_part3_test_rows"])
        # the default build is NOT clean: it contains the held-out conversations
        self.default_build()
        shutil.copy(self.out / pp.HELDOUT_MANIFEST_NAME, self.data / "ho_default" / pp.HELDOUT_MANIFEST_NAME)
        try:
            dirty = au.audit_dataset(self.data / "ho_default", self.src)
        finally:
            (self.data / "ho_default" / pp.HELDOUT_MANIFEST_NAME).unlink()
        self.assertGreater(dirty["required"]["train_conversations_in_heldout"]
                           + dirty["required"]["val_conversations_in_heldout"], 0)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(au.main(["--data_name", "ho", "--data_root", str(self.data),
                                      "--snapshot_dir", str(self.src)]), 0)
            # not a held-out build (no manifest) -> exit 1, nothing written
            self.assertEqual(au.main(["--data_name", "ho_default", "--data_root", str(self.data),
                                      "--snapshot_dir", str(self.src)]), 1)

    def test_eval_heldout_rows(self):
        import pandas as pd
        import evaluate_whisper_speaker_split as ev
        rows = ev.load_heldout_rows("ho", self.data)
        self.assertEqual(rows, self.manifest["part3_test_row_indices"])
        n3 = len(self.fx["test_conv"])
        m = pd.DataFrame({"sample_id": [f"p1_{i}" for i in range(5)] + [f"p3_{i}" for i in range(n3)],
                          "config": ["ASR-PART1-Test"] * 5 + ["ASR-PART3-Test"] * n3,
                          "row_index": list(range(5)) + list(range(n3))})
        f = ev.filter_heldout(m, rows)
        self.assertEqual(int((f["config"] == "ASR-PART1-Test").sum()), 5)
        self.assertEqual(f.loc[f["config"] == "ASR-PART3-Test", "row_index"].tolist(), rows)
        self.assertEqual(f.index.tolist(), list(range(len(f))))
        with self.assertRaises(AssertionError):
            ev.filter_heldout(m, [n3 + 5])
        with self.assertRaises(FileNotFoundError):
            ev.load_heldout_rows("ho_missing", self.data)

    def test_audit_eval_files_skip_and_drop(self):
        import audit_test_speaker_overlap as au
        res = self.tmp / "results_fake"
        res.mkdir(exist_ok=True)
        for name in ("eval_whisper-small-mnsc-sg-15k.csv",
                     "eval_whisper-small-mnsc-sg-15k_hx-x.csv",
                     "eval_whisper-small-mnsc-sg-15k_ho-x.csv"):
            (res / name).write_text("x\n")
        with mock.patch.object(au, "RESULTS_DIR", res):
            files = au.eval_files()
        self.assertEqual([p.name for _, _, p in files], ["eval_whisper-small-mnsc-sg-15k.csv"])


# ============================================================
# EXTRA HELD-OUT TEST SET + BOOTSTRAP
# ============================================================

def _tree_hashes(root: Path) -> dict:
    import hashlib
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def build_extra_source(root: Path) -> dict:
    """
    build_heldout_source() plus, for each of c01..c06, an exact copy (same context bytes,
    answer, other_attributes) of that conversation's first PART3-Train row appended to
    ASR-PART3-Test (still 2 shards). Original test answers are made unique so only the
    copies can match a train row by text.
    Returns the build_heldout_source dict plus "dup_train_clip": {conv: clip_id}.
    """
    fx = build_heldout_source(root)
    train_dir = root / "ASR-PART3-Train"
    first = {}
    for k, shard in enumerate(sorted(train_dir.glob("*.parquet"))):
        for j, row in enumerate(pq.read_table(shard).to_pylist()):
            conv = row["other_attributes"]["conversation_id"]
            if conv in {f"c0{i}" for i in range(1, 7)} and conv not in first:
                first[conv] = (f"ASR-PART3-Train/{k:05d}:{j}", row)
    test_dir = root / "ASR-PART3-Test"
    shards = sorted(test_dir.glob("*.parquet"))
    rows = []
    for shard in shards:
        rows += pq.read_table(shard).to_pylist()
    for i, r in enumerate(rows):
        r["answer"] = f"test utterance number {i}"
    for conv in sorted(first):
        rows.append(first[conv][1])
    for shard in shards:
        shard.unlink()
    per = int(np.ceil(len(rows) / 2))
    for k in range(2):
        chunk = rows[k * per:(k + 1) * per]
        pq.write_table(pa.table({
            "context": pa.array([r["context"] for r in chunk]),
            "instruction": pa.array([r["instruction"] for r in chunk]),
            "answer": pa.array([r["answer"] for r in chunk]),
            "other_attributes": pa.array([r["other_attributes"] for r in chunk]),
        }), str(test_dir / f"test-{k:05d}-of-00002.parquet"), row_group_size=7)
    fx["test_conv"] = [r["other_attributes"]["conversation_id"] for r in rows]
    fx["dup_train_clip"] = {conv: cid for conv, (cid, _) in first.items()}
    return fx


class ExtraTestSetTests(unittest.TestCase):
    """build_heldout_extra_test.py, its audit, and the eval loader."""

    @classmethod
    def setUpClass(cls):
        import build_heldout_extra_test as bx
        cls.bx = bx
        cls.tmp = Path(tempfile.mkdtemp(prefix="mnsc_extra_"))
        cls.src = cls.tmp / "src"
        cls.data = cls.tmp / "data"
        cls.fx = build_extra_source(cls.src)
        cls.base = ["--data_root", str(cls.data), "--source_dir", str(cls.src),
                    "--rows_per_shard", "8", "--row_group_size", "4",
                    "--max_samples", "0", "--max_clips_per_speaker", "0", "--val_fraction", "0.3"]
        rc, log = run_main(["--data_name", "ho", "--heldout_test_conversations", "3",
                            "--max_source_shards", "2"] + cls.base)
        assert rc == 0, log
        cls.ho_hashes = _tree_hashes(cls.data / "ho")
        cls.extra_args = ["--data_name", "ho", "--data_root", str(cls.data), "--source_dir", str(cls.src),
                          "--rows_per_shard", "8", "--row_group_size", "4"]
        with redirect_stdout(io.StringIO()) as out:
            rc = bx.main(cls.extra_args)
        assert rc == 0, out.getvalue()
        cls.extra = cls.data / "ho_extra_test"
        cls.info = json.loads((cls.extra / bx.EXTRA_INFO_NAME).read_text(encoding="utf-8"))
        cls.manifest = json.loads((cls.data / "ho" / pp.HELDOUT_MANIFEST_NAME).read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def bx_main(self, argv):
        with redirect_stdout(io.StringIO()) as out:
            rc = self.bx.main(argv)
        return rc, out.getvalue()

    # ---------------------------------------------------------------
    def test_selection_from_manifest(self):
        info, bx = self.info, self.bx
        held = set(self.manifest["conversations"])
        clips = info["clips"]
        self.assertEqual({c["conversation_id"] for c in clips}, held)
        n_rows = sum(1 for a in self.fx["train"].values() if a["conversation_id"] in held)
        self.assertEqual(info["n_clips"] + info["duplicates"]["n_dropped"], n_rows)  # no cap
        self.assertEqual(info["n_heldout_conversation_rows_seen"], n_rows)
        self.assertEqual(info["n_heldout_conversation_rows_seen"], info["expected_excluded_by_build"])
        tr, va, _ = pp.load_preprocessed_index("ho", self.data)
        self.assertFalse({c["clip_id"] for c in clips} & {i["clip_id"] for i in tr + va})
        self.assertTrue(set(info["voices"]) <= set(self.manifest["voices"]))
        self.assertEqual((info["kind"], info["config"], info["no_speaker_cap"]),
                         (bx.EXTRA_KIND, bx.EXTRA_TEST_CONFIG, True))
        t = pq.read_table(str(self.extra / "test" / info["layout"]["shards"][0]))
        self.assertEqual(t.column_names, pp.META_COLUMNS[:2] + ["audio_flac"] + pp.META_COLUMNS[2:])
        self.assertFalse((self.extra / pp.INFO_NAME).exists())
        items, _ = bx.load_extra_test_index("ho_extra_test", self.data)
        self.assertEqual([i["clip_id"] for i in items], [c["clip_id"] for c in clips])
        self.assertEqual([i["conversation_id"] for i in items], [c["conversation_id"] for c in clips])
        # audio round trip vs the source row
        item = items[-1]
        k, j = (int(x) for x in item["clip_id"].split("/")[1].split(":"))
        shard = sorted((self.src / "ASR-PART3-Train").glob("*.parquet"))[k]
        row = pq.read_table(shard).to_pylist()[j]
        ref = sf.read(io.BytesIO(row["context"]["bytes"]), dtype="float32")[0]
        self.assertTrue(np.array_equal(pp.read_clip_audio(item), ref))
        self.assertEqual(item["text"], pp.normalize_transcript(row["answer"]))

    def test_duplicate_detection(self):
        bx = self.bx
        held = set(self.manifest["conversations"])
        want = {self.fx["dup_train_clip"][c] for c in held}
        dropped = self.info["duplicates"]["dropped"]
        self.assertEqual({d["clip_id"] for d in dropped}, want)
        self.assertEqual(self.info["duplicates"]["n_dropped"], 3)
        for d in dropped:
            self.assertIn("audio_sha256", d["reasons"])
            self.assertIn(d["test_row_index"], self.manifest["part3_test_row_indices"])
        self.assertFalse(want & {c["clip_id"] for c in self.info["clips"]})
        keys = {"segments": {("c1", "1.0", "2.0"): 5}, "audio": {"abc": 6},
                "text": {("hello there", 1.23): 7}, "intervals": {"c1": [(1.0, 2.0, 5)]}}
        self.assertEqual(bx.duplicate_reasons(("c1", "1.0", "2.0"), "zzz", [], keys), ["segment_id"])
        self.assertEqual(bx.duplicate_reasons(None, "zzz", [bx.text_key("Hello, there!", 1.231)], keys),
                         ["text_duration"])
        self.assertEqual(bx.duplicate_reasons(None, "zzz", [bx.text_key("Hello there", 1.30)], keys), [])
        self.assertEqual(bx.duplicate_reasons(None, "abc", [], keys), ["audio_sha256"])
        self.assertIsNone(bx.segment_key({"conversation_id": "c1", "start": "", "end": "2.0"}))
        self.assertEqual(bx.segment_key({"conversation_id": " c1", "start": "1.0 ", "end": "2.0"}),
                         ("c1", "1.0", "2.0"))
        self.assertEqual(bx.overlapping_rows({"conversation_id": "c1", "start": "1.5", "end": "3"}, keys), [5])
        self.assertEqual(bx.overlapping_rows({"conversation_id": "c1", "start": "2.0", "end": "3"}, keys), [])
        self.assertEqual(bx.overlapping_rows({"conversation_id": "c1"}, keys), [])
        a = np.array([1, -2, 3], dtype=np.int16)
        self.assertEqual(bx.pcm_sha256(a), bx.pcm_sha256(a.copy()))
        self.assertNotEqual(bx.pcm_sha256(a), bx.pcm_sha256(a[::-1]))

    def test_guards(self):
        args = self.extra_args
        self.assertEqual(self.bx_main(args + ["--out_name", "ho"])[0], 2)
        self.assertEqual(self.bx_main(args)[0], 2)  # exists, no --overwrite
        tv = self.data / "fake_tv"
        tv.mkdir(exist_ok=True)
        (tv / pp.INFO_NAME).write_text("{}")
        self.assertEqual(self.bx_main(args + ["--out_name", "fake_tv", "--overwrite"])[0], 2)
        self.assertTrue((tv / pp.INFO_NAME).exists())
        foreign = self.data / "fake_foreign"
        foreign.mkdir(exist_ok=True)
        (foreign / "keep.txt").write_text("x")
        self.assertEqual(self.bx_main(args + ["--out_name", "fake_foreign", "--overwrite"])[0], 2)
        self.assertTrue((foreign / "keep.txt").exists())
        self.assertEqual(self.bx_main(args + ["--out_name", "x2", "--max_source_shards", "99"])[0], 2)
        self.assertFalse((self.data / "x2").exists())
        self.assertEqual(self.bx_main(args + ["--out_name", "../x"])[0], 2)
        self.assertEqual(self.bx_main(["--data_name", "ho_missing", "--data_root", str(self.data)])[0], 2)
        rc, log = self.bx_main(args + ["--out_name", "x3"])
        self.assertEqual(rc, 0, log)
        rc, log = self.bx_main(args + ["--out_name", "x3", "--overwrite"])
        self.assertEqual(rc, 0, log)
        info3 = json.loads((self.data / "x3" / self.bx.EXTRA_INFO_NAME).read_text(encoding="utf-8"))
        self.assertEqual(info3["clips"], self.info["clips"])
        self.assertEqual(_tree_hashes(self.data / "ho"), self.ho_hashes)

    def test_audit_extra(self):
        import audit_test_speaker_overlap as au
        res = au.audit_extra_test(self.extra, self.data / "ho", self.src)
        self.assertTrue(all(v == 0 for v in res["required"].values()), res["required"])
        self.assertTrue(all(res["consistency"].values()), res["consistency"])
        self.assertEqual(res["info"]["n_clips"], self.info["n_clips"])
        # a default build contains the held-out conversations -> dirty
        rc, log = run_main(["--data_name", "ho_default"] + self.base)
        self.assertEqual(rc, 0, log)
        shutil.copy(self.data / "ho" / pp.HELDOUT_MANIFEST_NAME,
                    self.data / "ho_default" / pp.HELDOUT_MANIFEST_NAME)
        try:
            dirty = au.audit_extra_test(self.extra, self.data / "ho_default", self.src)
        finally:
            shutil.rmtree(self.data / "ho_default")
        self.assertGreater(dirty["required"]["extra_conversations_in_train_val"], 0)
        self.assertGreater(dirty["required"]["extra_clip_ids_in_train_val"], 0)
        with redirect_stdout(io.StringIO()) as out:
            rc = au.main(["--data_name", "ho", "--data_root", str(self.data),
                          "--extra_test", "ho_extra_test", "--snapshot_dir", str(self.src)])
        self.assertEqual(rc, 0, out.getvalue())
        self.assertIn("EXTRA TEST SET", out.getvalue())
        with redirect_stdout(io.StringIO()):
            self.assertEqual(au.main(["--extra_test", "ho_extra_test"]), 2)

    def test_eval_extra_args_and_manifest(self):
        import evaluate_whisper_speaker_split as ev
        d = str(self.data)
        a = ev.parse_args(["--heldout_extra_test", "ho_extra_test", "--data_root", d])
        self.assertEqual((a.heldout_extra_test, a.data_root, a.split), ("ho_extra_test", d, "test"))
        for bad in (["--heldout_extra_test", "x", "--split", "val"],
                    ["--heldout_extra_test", "x", "--heldout_test", "y"],
                    ["--heldout_extra_test", "../x"],
                    ["--heldout_extra_test", "x", "--data_name", "y"],
                    ["--data_root", "x"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit, msg=bad):
                ev.parse_args(bad)
        m, items = ev.load_extra_test_manifest("ho_extra_test", d)
        self.assertEqual(list(m.columns), ["sample_id", "dataset", "config", "row_index"])
        self.assertEqual(set(m["config"]), {"PART3-heldout-train-clips"})
        self.assertEqual(m["row_index"].tolist(), list(range(self.info["n_clips"])))
        self.assertEqual(m["sample_id"].tolist(), [i["clip_id"] for i in items])
        self.assertTrue(ev.load_extra_test_manifest("ho_extra_test", d, 2)[0].equals(m.head(2)))
        self.assertEqual(ev.summary_groups("test", None, "ho_extra_test"), ["PART3-heldout-train-clips"])
        self.assertIn("not PART3-Test", ev.data_source_label("test", None, d, extra_test="ho_extra_test"))
        audio, sr, text = ev.val_clip(items[0], {})
        self.assertEqual((sr, text), (SR, items[0]["text"]))
        lookup = {("ASR-PART3-Test", 0): {"other_attributes": _conv_attrs("c01", "c01-a", "c01-b")},
                  ("ASR-PART1-Test", 0): {"other_attributes": {"speaker": {"speaker_id": "p1"}}}}
        self.assertEqual(ev.test_cluster_ids(lookup, {"config": "ASR-PART3-Test", "row_index": 0}),
                         ("c01", "c01-a"))
        self.assertEqual(ev.test_cluster_ids(lookup, {"config": "ASR-PART1-Test", "row_index": 0}),
                         ("", "p1"))


class BootstrapTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import pandas as pd
        import bootstrap_wer_ci as bw
        cls.bw, cls.pd = bw, pd
        cls.tmp = Path(tempfile.mkdtemp(prefix="mnsc_boot_"))
        rng = np.random.default_rng(11)
        words = "one two three four five six seven eight nine ten".split()
        rows = []

        def add(config, n, cluster_col, n_clusters):
            for i in range(n):
                ref = " ".join(rng.choice(words, int(rng.integers(3, 9))))
                hyp = ref.split()
                for _ in range(int(rng.integers(1, 4))):  # every row has >= 1 error
                    hyp[int(rng.integers(0, len(hyp)))] = "x"
                cid = f"{config[:4]}{i % n_clusters}"
                rows.append({"sample_id": f"{config}_{i}", "config": config, "row_index": i,
                             "reference": ref, "prediction": " ".join(hyp),
                             "conversation_id": cid if cluster_col == "conv" else "",
                             "speaker_id": cid if cluster_col == "spk" else f"s{i}"})

        add("ASR-PART1-Test", 8, "spk", 4)
        add("ASR-PART3-Test", 6, "conv", 3)
        add("PART3-heldout-train-clips", 4, "conv", 2)
        cls.raw = pd.DataFrame(rows)
        cls.df = bw.attach_clusters(bw.row_counts(cls.raw))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def frame(self, rows):
        return self.pd.DataFrame(rows).set_index(["group", "metric"])

    def test_point_estimate_matches_corpus_metric(self):
        import evaluate_whisper_speaker_split as ev
        out = self.frame(self.bw.bootstrap_file(self.df, 200))
        for g in ("ASR-PART1-Test", "ASR-PART3-Test", "PART3-heldout-train-clips"):
            sub = self.df[self.df["config"] == g]
            for metric in ("WER", "CER"):
                self.assertAlmostEqual(out.loc[(g, metric), "estimate"], ev.corpus_metric(sub, metric), 12)
        for metric in ("WER", "CER"):
            self.assertAlmostEqual(out.loc[("Overall", metric), "estimate"],
                                   ev.corpus_metric(self.df, metric), 12)
        self.assertEqual(out.loc[("ASR-PART1-Test", "WER"), "cluster_unit"], "speaker")
        self.assertEqual(out.loc[("ASR-PART3-Test", "WER"), "cluster_unit"], "conversation")
        self.assertEqual(out.loc[("Overall", "WER"), "cluster_unit"], "stratified")
        self.assertEqual([g for g, _ in out.index[::2]],
                         ["ASR-PART1-Test", "ASR-PART3-Test", "PART3-heldout-train-clips", "Overall"])

    def test_deterministic_and_ci_contains_estimate(self):
        bw, pd = self.bw, self.pd
        a = pd.DataFrame(bw.bootstrap_file(self.df, 300, 0.95, 42))
        b = pd.DataFrame(bw.bootstrap_file(self.df, 300, 0.95, 42))
        c = pd.DataFrame(bw.bootstrap_file(self.df, 300, 0.95, 43))
        pd.testing.assert_frame_equal(a, b)
        self.assertFalse(np.allclose(a["ci_low"], c["ci_low"]) and np.allclose(a["ci_high"], c["ci_high"]))
        self.assertTrue(((a["ci_low"] <= a["estimate"]) & (a["estimate"] <= a["ci_high"])).all(), a)
        n = a.set_index(["group", "metric"])["n_clusters"]
        self.assertEqual((n[("ASR-PART1-Test", "WER")], n[("ASR-PART3-Test", "WER")],
                          n[("PART3-heldout-train-clips", "WER")], n[("Overall", "WER")]), (4, 3, 2, 9))
        # row order of the input does not change the result
        shuffled = self.df.sample(frac=1, random_state=0)
        pd.testing.assert_frame_equal(a, pd.DataFrame(bw.bootstrap_file(shuffled, 300, 0.95, 42)))
        one = self.df[self.df["cluster"] == "conv:ASR-0"]
        r = pd.DataFrame(bw.bootstrap_file(one, 50))
        for _, row in r.iterrows():
            self.assertAlmostEqual(row["ci_low"], row["estimate"], 12)
            self.assertAlmostEqual(row["ci_high"], row["estimate"], 12)

    def test_paired(self):
        bw, pd = self.bw, self.pd
        same = pd.DataFrame(bw.bootstrap_paired(self.df, self.df.copy(), 200))
        self.assertTrue((same["diff"] == 0).all() and (same["ci_low"] == 0).all() and (same["ci_high"] == 0).all())
        perfect = bw.attach_clusters(bw.row_counts(self.raw.assign(prediction=self.raw["reference"])))
        p = pd.DataFrame(bw.bootstrap_paired(self.df, perfect, 200))
        self.assertTrue((p["ci_high"] < 0).all(), p)
        self.assertTrue((p["p_boot_b_not_better"] == 0).all())
        self.assertTrue((p["b"] == 0).all())
        bad = perfect.copy()
        bad.loc[0, "reference"] = "something else"
        with self.assertRaises(ValueError):
            bw.bootstrap_paired(self.df, bad, 10)
        with self.assertRaises(ValueError):
            bw.bootstrap_paired(self.df, perfect.assign(config="other-val"), 10)
        # same draws for A and B: a + diff == b for every group
        self.assertTrue(np.allclose(p["a"] + p["diff"], p["b"]))

    def test_old_csv_cluster_join_and_cli(self):
        bw, pd = self.bw, self.pd
        src = self.tmp / "src"
        fx = build_heldout_source(src)
        n3 = len(fx["test_conv"])
        old = pd.DataFrame({
            "sample_id": [f"p3_{i}" for i in range(n3)] + ["p1_0", "p1_1"],
            "dataset": "MNSC-Extended",
            "config": ["ASR-PART3-Test"] * n3 + ["ASR-PART1-Test"] * 2,
            "row_index": list(range(n3)) + [0, 1],
            "reference": ["a b c"] * (n3 + 2), "prediction": ["a b"] * (n3 + 2),
            "WER": 1 / 3, "CER": 0.4,
        })
        path = self.tmp / "eval_old.csv"
        old.to_csv(path, index=False)
        got = bw.attach_clusters(bw.row_counts(bw.load_eval_csv(path)), str(src))
        p3 = got[got["config"] == "ASR-PART3-Test"]
        self.assertEqual(p3["conversation_id"].tolist(), fx["test_conv"])
        self.assertEqual(got.loc[got["config"] == "ASR-PART1-Test", "cluster"].tolist(),
                         ["spk:c08-a", "spk:p1-other"])
        with self.assertRaises(ValueError):  # val rows without ids cannot be joined
            bw.attach_clusters(bw.row_counts(old.assign(config="x-val")), str(src))
        out = self.tmp / "boot.csv"
        argv = [str(path), "--snapshot_dir", str(src), "--n_boot", "50", "--out", str(out)]
        with redirect_stdout(io.StringIO()) as log:
            self.assertEqual(bw.main(argv), 0, log.getvalue())
        res = pd.read_csv(out)
        self.assertEqual(set(res["group"]), {"ASR-PART1-Test", "ASR-PART3-Test", "Overall"})
        self.assertEqual(set(res["file"]), {"eval_old.csv"})
        with redirect_stdout(io.StringIO()):
            self.assertEqual(bw.main(argv), 2)                       # exists, no --overwrite
            self.assertEqual(bw.main(argv + ["--overwrite"]), 0)
            self.assertEqual(bw.main([str(path), "--out", str(path), "--overwrite"]), 2)  # out == input
            old.drop(columns=["prediction"]).to_csv(self.tmp / "eval_bad.csv", index=False)
            self.assertEqual(bw.main([str(self.tmp / "eval_bad.csv"), "--out", str(self.tmp / "b2.csv")]), 2)
            pout = self.tmp / "paired.csv"
            self.assertEqual(bw.main(["--paired", str(path), str(path), "--snapshot_dir", str(src),
                                      "--n_boot", "20", "--out", str(pout)]), 0)
        self.assertTrue((pd.read_csv(pout)["diff"] == 0).all())
        self.assertEqual(bw.default_out(["results/eval_a.csv"], False, 1000, 42).name,
                         "bootstrap_a_n1000_s42.csv")
        self.assertEqual(bw.default_out(["eval_a.csv", "eval_b.csv"], True, 1000, 42).name,
                         "bootstrap_paired_a__vs__b_n1000_s42.csv")
        self.assertEqual(bw.default_out(["eval_a.csv", "eval_b.csv"], False, 10, 1).name,
                         "bootstrap_a_plus1_n10_s1.csv")
        self.assertEqual(bw.default_out(["x"], False, 1000, 42).parent, bw.RESULTS_DIR)
        with redirect_stderr(io.StringIO()):
            for bad in (["--paired", "a"], ["a", "--confidence", "1.5"], ["a", "--n_boot", "0"]):
                with self.assertRaises(SystemExit, msg=bad):
                    bw.parse_args(bad)

    def test_val_conversation_join(self):
        bw, pd = self.bw, self.pd
        src = self.tmp / "src_val"
        fx = build_heldout_source(src)
        clip_ids = sorted(fx["train"])[::2]
        raw = pd.DataFrame({
            "sample_id": clip_ids, "dataset": "MNSC-Extended", "config": "dsx-val",
            "row_index": range(len(clip_ids)), "reference": "a b c", "prediction": "a b",
            "conversation_id": "",
            "speaker_id": [fx["train"][c]["speaker1"]["speaker_id"] for c in clip_ids],
        })
        path = self.tmp / "eval_val.csv"
        raw.to_csv(path, index=False)
        df = bw.row_counts(bw.load_eval_csv(path))
        snapshot = df.copy()
        got = bw.attach_clusters(df, str(src), val_conversations=True)
        self.assertEqual(got["cluster"].tolist(),
                         [f"conv:{fx['train'][c]['conversation_id']}" for c in clip_ids])
        self.assertEqual(set(got["cluster_unit"]), {"conversation"})
        pd.testing.assert_frame_equal(df, snapshot)                      # input unchanged
        plain = bw.attach_clusters(df, str(src))
        self.assertTrue(plain["cluster"].str.startswith("spk:").all())  # default: speaker
        wrong = df.copy()
        wrong.loc[0, "speaker_id"] = "someone-else"
        with self.assertRaises(ValueError):
            bw.attach_clusters(wrong, str(src), val_conversations=True)
        bad_id = df.copy()
        bad_id.loc[0, "sample_id"] = "ASR-PART3-Train/00099:0"           # shard not in snapshot
        with self.assertRaises(ValueError):
            bw.attach_clusters(bad_id, str(src), val_conversations=True)
        self.assertEqual(bw.val_shard_indices(clip_ids + ["x"]), [0, 1])
        out = self.tmp / "boot_val.csv"
        before = path.read_bytes()
        with redirect_stdout(io.StringIO()) as log:
            self.assertEqual(bw.main(["--paired", str(path), str(path), "--val_conversations",
                                      "--snapshot_dir", str(src), "--n_boot", "20",
                                      "--out", str(out)]), 0, log.getvalue())
        res = pd.read_csv(out)
        self.assertEqual(set(res.loc[res["group"] == "dsx-val", "cluster_unit"]), {"conversation"})
        self.assertEqual(path.read_bytes(), before)                     # CSV untouched


if __name__ == "__main__":
    unittest.main()
