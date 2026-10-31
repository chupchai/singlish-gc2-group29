"""
Offline tests for the learning-rate / seed sweep: the training flags (--learning_rate, --warmup_ratio, --seed), the sweep runner
and the comparison script (no training, no model loading, no downloads).

Run from the project root:
    python -m unittest discover -s scripts/tests -t . -v
"""

import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pandas as pd
import torch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import compare_lr_sweep as cs              # noqa: E402
import evaluate_whisper_speaker_split as ev  # noqa: E402
import run_lr_sweep as rs                  # noqa: E402
import train_whisper_speaker_split as tw   # noqa: E402
from transformers import Seq2SeqTrainingArguments  # noqa: E402


def original_kwargs(output_dir, eval_steps, epochs=3, max_steps=-1, num_workers=0):
    """TrainingArguments kwargs exactly as the script passed them before the new flags
    (copied from the pre-change script)."""
    return dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=epochs,
        max_steps=max_steps,
        fp16=True,
        gradient_checkpointing=True,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=eval_steps,
        logging_steps=min(25, max(1, eval_steps // 2)),
        dataloader_num_workers=num_workers,
        dataloader_persistent_workers=num_workers > 0,
        predict_with_generate=False,
        report_to="none",
        remove_unused_columns=False,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )


def cpu_args(kw):
    """TrainingArguments that can be built without CUDA (fp16 needs a GPU)."""
    return Seq2SeqTrainingArguments(**dict(kw, fp16=False))


class _Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="lr_sweep_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


# ============================================================
# TRAINING FLAGS
# ============================================================

class TrainArgsTests(_Tmp):

    def test_defaults_and_parsing(self):
        a = tw.parse_args([])
        self.assertEqual((a.learning_rate, a.warmup_ratio, a.seed), (2e-5, 0.0, 42))
        self.assertEqual((a.data_name, a.run_name, a.max_steps, a.eval_steps, a.max_eval_samples,
                          a.num_workers, a.epochs, a.overwrite_model),
                         ("part3_15k", None, -1, 0, 0, 0, 3, False))
        a = tw.parse_args(["--learning_rate", "3e-5", "--warmup_ratio", "0.1", "--seed", "43"])
        self.assertEqual((a.learning_rate, a.warmup_ratio, a.seed), (3e-5, 0.1, 43))
        for bad in (["--warmup_ratio", "1.0"], ["--warmup_ratio", "-0.1"], ["--learning_rate", "0"],
                    ["--seed", "x"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit, msg=bad):
                tw.parse_args(bad)

    def test_default_kwargs_reproduce_old_training_args(self):
        out = self.tmp / "ckpt"
        a = tw.parse_args([])
        self.assertEqual(tw.training_args_kwargs(a, out, 844),
                         dict(original_kwargs(out, 844), seed=42, warmup_steps=0))
        a = tw.parse_args(["--max_steps", "20", "--eval_steps", "10", "--num_workers", "2"])
        self.assertEqual(tw.training_args_kwargs(a, out, 10),
                         dict(original_kwargs(out, 10, max_steps=20, num_workers=2),
                              seed=42, warmup_steps=0))
        # seed=42 / warmup_steps=0 are the HF defaults, so the objects are identical
        old = cpu_args(original_kwargs(out, 844)).to_dict()
        new = cpu_args(tw.training_args_kwargs(tw.parse_args([]), out, 844)).to_dict()
        self.assertEqual(old, new)

    @unittest.skipUnless(torch.cuda.is_available(), "fp16 TrainingArguments need CUDA")
    def test_build_training_args_equals_old_on_gpu(self):
        out = self.tmp / "ckpt"
        self.assertEqual(Seq2SeqTrainingArguments(**original_kwargs(out, 844)).to_dict(),
                         tw.build_training_args(tw.parse_args([]), out, 844).to_dict())

    def test_warmup_resolution_and_record(self):
        out = self.tmp / "ckpt"
        self.assertEqual(tw.total_training_steps(1689, 3, -1), 5067)
        self.assertEqual(tw.total_training_steps(1689, 3, 20), 20)
        a = tw.parse_args(["--data_name", "part3_15k_heldout", "--learning_rate", "1e-5",
                           "--warmup_ratio", "0.1", "--seed", "43"])
        targs = cpu_args(tw.training_args_kwargs(a, out, 844))
        self.assertEqual(targs.get_warmup_steps(5067), 507)
        self.assertEqual(targs.get_warmup_steps(20), 2)
        self.assertEqual(cpu_args(tw.training_args_kwargs(tw.parse_args([]), out, 844))
                         .get_warmup_steps(5067), 0)
        rec = tw.hyperparameters_record(a, targs, 5067, "run-x")
        self.assertEqual({k: rec[k] for k in ("run_name", "data_name", "learning_rate", "warmup_ratio",
                                              "warmup_steps_resolved", "seed", "total_steps",
                                              "batch_size", "gradient_accumulation",
                                              "lr_scheduler_type", "weight_decay")},
                         {"run_name": "run-x", "data_name": "part3_15k_heldout", "learning_rate": 1e-5,
                          "warmup_ratio": 0.1, "warmup_steps_resolved": 507, "seed": 43,
                          "total_steps": 5067, "batch_size": 2, "gradient_accumulation": 4,
                          "lr_scheduler_type": "linear", "weight_decay": 0.0})
        json.dumps(rec)  # must be JSON-serializable for resource_usage.json

    def test_resume_check(self):
        rec = {"learning_rate": 2e-5, "warmup_ratio": 0.0, "seed": 42, "data_name": "d"}
        p = self.tmp / "resource_usage.json"
        self.assertIsNone(tw.check_resume_hyperparameters(p, rec))           # no file
        p.write_text(json.dumps({"median_s_per_step": 1.3}), encoding="utf-8")
        self.assertIsNone(tw.check_resume_hyperparameters(p, rec))           # old file, no block
        p.write_text(json.dumps({"hyperparameters": dict(rec)}), encoding="utf-8")
        self.assertIsNone(tw.check_resume_hyperparameters(p, rec))           # same values
        for key, val in (("learning_rate", 3e-5), ("seed", 44), ("warmup_ratio", 0.1)):
            msg = tw.check_resume_hyperparameters(p, dict(rec, **{key: val}))
            self.assertIn(key, msg)
            self.assertIn("ERROR", msg)

    def test_resource_monitor_records_hyperparameters(self):
        cb = tw.ResourceMonitorCallback(self.tmp / "ru.json", 10, 3, 5, 16, 16,
                                        hyperparameters={"seed": 43})

        class S:
            global_step, epoch = 5, 0.5
        cb.on_evaluate(None, S(), None, metrics={"eval_loss": 0.7, "eval_runtime": 1.0})
        rep = json.loads((self.tmp / "ru.json").read_text(encoding="utf-8"))
        self.assertEqual(rep["hyperparameters"], {"seed": 43})
        self.assertEqual(rep["eval_history"], [{"step": 5, "epoch": 0.5, "eval_loss": 0.7}])
        self.assertIn("median_s_per_step", rep)


# ============================================================
# RUNNER
# ============================================================

def make_model(root: Path, name: str, complete: bool = True):
    d = root / "models" / name
    d.mkdir(parents=True, exist_ok=True)
    files = rs.MODEL_FILES + ["processor_config.json", "training_args.bin"]
    for f in (files if complete else ["config.json", "model.safetensors"]):
        (d / f).write_text("x", encoding="utf-8")


class FakeRunner:
    """Records commands; 'trains' by writing the model files and 'evaluates' by writing the
    summary + eval CSV. fail = set of (run name, 'train'|'eval') that return rc 1."""

    def __init__(self, root: Path, smoke: bool, fail=()):
        self.root, self.smoke, self.fail, self.calls = root, smoke, set(fail), []

    def __call__(self, cmd, log_path, cwd):
        name = cmd[cmd.index("--run_name") + 1]
        kind = "train" if cmd[2].endswith("train_whisper_speaker_split.py") else "eval"
        self.calls.append((name, kind))
        if (name, kind) in self.fail:
            return 1, 0.5
        if kind == "train":
            make_model(self.root, name)
        else:
            for p in (rs.val_summary_path(name, self.smoke, self.root),
                      rs.val_eval_path(name, self.smoke, self.root)):
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("x", encoding="utf-8")
        return 0, 1.0


class RunnerTests(_Tmp):

    def test_runs_and_commands(self):
        self.assertEqual([r["name"] for r in rs.SWEEP_RUNS],
                         ["whisper-small-mnsc-sg-15k-heldout-seed43",
                          "whisper-small-mnsc-sg-15k-heldout-lr1e-5-wu10",
                          "whisper-small-mnsc-sg-15k-heldout-lr3e-5-wu10"])
        self.assertEqual([(r["learning_rate"], r["warmup_ratio"], r["seed"]) for r in rs.SWEEP_RUNS],
                         [(2e-5, 0.0, 43), (1e-5, 0.1, 42), (3e-5, 0.1, 42)])
        r = rs.SWEEP_RUNS[1]
        t = rs.train_command(r, False)
        self.assertTrue(t[2].endswith("train_whisper_speaker_split.py"))
        self.assertEqual(t[3:], ["--data_name", "part3_15k_heldout", "--run_name", r["name"],
                                 "--learning_rate", "1e-05", "--warmup_ratio", "0.1", "--seed", "42"])
        self.assertEqual(rs.train_command(r, True)[3:],
                         ["--data_name", "part3_15k_heldout", "--run_name", "smoke-" + r["name"],
                          "--learning_rate", "1e-05", "--warmup_ratio", "0.1", "--seed", "42",
                          "--max_steps", "20", "--eval_steps", "10", "--max_eval_samples", "16"])
        # the training script accepts the generated flags
        a = tw.parse_args(rs.train_command(r, True)[3:])
        self.assertEqual((a.learning_rate, a.warmup_ratio, a.seed, a.max_steps), (1e-5, 0.1, 42, 20))
        e = rs.eval_command(r, False)
        self.assertTrue(e[2].endswith("evaluate_whisper_speaker_split.py"))
        self.assertEqual(e[3:], ["--run_name", r["name"], "--split", "val", "--data_name",
                                 "part3_15k_heldout"])
        self.assertEqual(rs.eval_command(r, True)[-2:], ["--max_rows", "64"])
        self.assertNotIn("--heldout_test", " ".join(e + rs.eval_command(r, True)))

    def test_summary_path_matches_eval_naming(self):
        # needs models/<run>/ to not exist only for --model_id; plain parse is fine
        for smoke in (False, True):
            for r in rs.SWEEP_RUNS:
                name = rs.run_name(r, smoke)
                args = ev.parse_args(rs.eval_command(r, smoke)[3:])
                self.assertEqual(args.data_name, "part3_15k_heldout")
                eval_p, summ_p = ev.output_paths(name, ev.run_suffix(args))
                self.assertEqual(rs.val_summary_path(name, smoke).name, summ_p.name)
                self.assertEqual(rs.val_eval_path(name, smoke).name, eval_p.name)
                self.assertEqual(rs.val_summary_path(name, smoke).parent, ev.RESULTS_DIR)
        self.assertEqual(rs.val_summary_path(rs.BASELINE_RUN, False).name,
                         "summary_whisper-small-mnsc-sg-15k-heldout_val_fb.csv")

    def test_model_complete(self):
        make_model(self.tmp, "a")
        self.assertTrue(rs.model_complete(self.tmp / "models" / "a"))
        make_model(self.tmp, "b", complete=False)
        self.assertFalse(rs.model_complete(self.tmp / "models" / "b"))
        self.assertFalse(rs.model_complete(self.tmp / "models" / "nope"))
        self.assertFalse(rs.has_checkpoint(self.tmp / "checkpoints" / "a"))
        (self.tmp / "checkpoints" / "a" / "checkpoint-10").mkdir(parents=True)
        self.assertTrue(rs.has_checkpoint(self.tmp / "checkpoints" / "a"))

    def test_sweep_skips_and_continues(self):
        names = [rs.run_name(r, False) for r in rs.SWEEP_RUNS]
        make_model(self.tmp, names[1])                                  # run 2 already trained
        s = rs.val_summary_path(names[2], False, self.tmp)              # run 3 eval already done
        s.parent.mkdir(parents=True)
        s.write_text("x", encoding="utf-8")
        fake = FakeRunner(self.tmp, False, fail={(names[0], "train")})  # run 1 fails
        with redirect_stdout(io.StringIO()):
            rows = rs.run_sweep(False, False, self.tmp, fake)
        self.assertEqual(fake.calls, [(names[0], "train"), (names[1], "eval"), (names[2], "train")])
        self.assertEqual([(r["train"], r["eval"]) for r in rows],
                         [("failed (rc 1)", "skipped (no model)"),
                          ("skipped (complete)", "done"),
                          ("done", "skipped (summary exists)")])
        self.assertTrue(rs.any_failed(rows))
        table = rs.format_status_table(rows)
        for n in names:
            self.assertIn(n, table)
        self.assertTrue((self.tmp / "logs" / "sweep" / f"{names[0]}.log").exists())
        # second pass: everything that succeeded is skipped; the failed run is retried
        fake2 = FakeRunner(self.tmp, False)
        with redirect_stdout(io.StringIO()):
            rows = rs.run_sweep(False, False, self.tmp, fake2)
        self.assertEqual(fake2.calls, [(names[0], "train"), (names[0], "eval")])
        self.assertFalse(rs.any_failed(rows))

    def test_dry_run_creates_nothing(self):
        fake = FakeRunner(self.tmp, True)
        with redirect_stdout(io.StringIO()) as out:
            rows = rs.run_sweep(True, True, self.tmp, fake)
            self.assertEqual(rs.main(["-DryRun"], root=self.tmp, runner=fake), 0)
            self.assertEqual(rs.main(["--dry_run", "-SmokeTest"], root=self.tmp, runner=fake), 0)
        self.assertEqual(fake.calls, [])
        self.assertEqual(len(rows), 3)
        self.assertEqual(list(self.tmp.iterdir()), [])
        self.assertEqual(out.getvalue().count("[dry run] TRAIN"), 9)
        self.assertEqual(out.getvalue().count("[dry run] EVAL"), 9)

    def test_main_real_with_fake_runner(self):
        fake = FakeRunner(self.tmp, True)
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(rs.main(["-SmokeTest"], root=self.tmp, runner=fake), 0)
        self.assertEqual(len(fake.calls), 6)
        self.assertTrue(all(n.startswith("smoke-") for n, _ in fake.calls))
        status = self.tmp / "logs" / "sweep" / "smoke-sweep_status.log"
        self.assertIn("done", status.read_text(encoding="utf-8"))
        self.assertIn("compare_lr_sweep.py --smoke", out.getvalue())
        fail = FakeRunner(self.tmp, False, fail={(rs.SWEEP_RUNS[0]["name"], "train")})
        with redirect_stdout(io.StringIO()):
            self.assertEqual(rs.main([], root=self.tmp, runner=fail), 1)


# ============================================================
# COMPARISON
# ============================================================

def decide_table(noise, cands, cur_ok=True):
    """noise: (diff, ci_low, ci_high) or None (pending); cands: list of (name, diff, lo, hi, wer)."""
    rows = [{"run": "cur", "role": "current", "status": "ok" if cur_ok else "pending",
             "val_WER": 0.237, "wer_diff_vs_current": None, "ci_low": None, "ci_high": None}]
    if noise is None:
        rows.append({"run": "seed", "role": "noise", "status": "pending", "val_WER": None,
                     "wer_diff_vs_current": None, "ci_low": None, "ci_high": None})
    else:
        rows.append({"run": "seed", "role": "noise", "status": "ok", "val_WER": 0.237 + noise[0],
                     "wer_diff_vs_current": noise[0], "ci_low": noise[1], "ci_high": noise[2]})
    for name, d, lo, hi, wer in cands:
        rows.append({"run": name, "role": "candidate", "status": "ok", "val_WER": wer,
                     "wer_diff_vs_current": d, "ci_low": lo, "ci_high": hi})
    t = pd.DataFrame(rows)
    for c in cs.COLUMNS:
        if c not in t.columns:
            t[c] = None
    return t


WORDS = "alpha bravo charlie delta echo".split()


def write_val(root: Path, name: str, suffix: str, n_rows: int, wrong_rows=(), with_summary=True):
    """Per-row val CSV + summary; rows in wrong_rows get one wrong word out of 5."""
    import bootstrap_wer_ci as bw
    rows = []
    for i in range(n_rows):
        ref = " ".join(WORDS)
        hyp = ref.replace("alpha", "xx") if i in set(wrong_rows) else ref
        rows.append({"sample_id": f"ASR-PART3-Train/00000:{i}", "dataset": "MNSC-Extended",
                     "config": "part3_15k_heldout-val", "row_index": i, "reference": ref,
                     "prediction": hyp, "WER": 0.0, "CER": 0.0, "conversation_id": "",
                     "speaker_id": f"spk{i // 2}", "compression_ratio": 1.0})
    res = root / "results"
    res.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(res / f"eval_{name}{suffix}.csv", index=False)
    c = bw.row_counts(df)
    wer = c["w_err"].sum() / c["w_ref"].sum()
    if with_summary:
        pd.DataFrame([{"group": g, "samples": n_rows, "mean_WER": wer, "mean_CER": wer / 2,
                       "corpus_WER": wer, "corpus_CER": wer / 2, "n_pred_gt3x_ref": 0,
                       "n_cr_gt_threshold": 0} for g in ("Overall", "part3_15k_heldout-val")]
                     ).to_csv(res / f"summary_{name}{suffix}.csv", index=False)
    return wer


def write_ckpt(root: Path, name: str, hparams=None):
    d = root / "checkpoints" / name
    for step, best in ((10, 10), (20, 10)):
        (d / f"checkpoint-{step}").mkdir(parents=True, exist_ok=True)
        st = {"global_step": step, "best_global_step": best, "best_metric": 0.61234,
              "log_history": [{"step": 10, "epoch": 0.5, "eval_loss": 0.61234},
                              {"step": 20, "epoch": 1.0, "eval_loss": 0.7}][: step // 10]}
        (d / f"checkpoint-{step}" / "trainer_state.json").write_text(json.dumps(st), encoding="utf-8")
    if hparams:
        (d / "resource_usage.json").write_text(json.dumps({"hyperparameters": hparams}), encoding="utf-8")


class CompareTests(_Tmp):

    def test_decide(self):
        w, reasons, t = cs.decide(decide_table((0.005, -0.01, 0.02), [("a", -0.02, -0.03, -0.01, 0.217)]))
        self.assertEqual(w, "a")
        self.assertTrue(t.loc[t["run"] == "a", "wins"].iloc[0])
        self.assertAlmostEqual(t.loc[t["run"] == "a", "noise_floor"].iloc[0], 0.005)
        # CI below 0 but improvement not above the noise floor
        w, _, _ = cs.decide(decide_table((0.005, -0.01, 0.02), [("a", -0.004, -0.008, -0.001, 0.233)]))
        self.assertIsNone(w)
        # CI contains 0
        w, _, t = cs.decide(decide_table((0.001, -0.01, 0.01), [("a", -0.02, -0.04, 0.003, 0.217)]))
        self.assertIsNone(w)
        self.assertFalse(t.loc[t["run"] == "a", "ci_excludes_0_better"].iloc[0])
        # seed run missing: no decision possible
        w, reasons, _ = cs.decide(decide_table(None, [("a", -0.05, -0.07, -0.03, 0.187)]))
        self.assertIsNone(w)
        self.assertIn("noise floor not available", " ".join(reasons))
        self.assertIn("noise floor not available", cs.recommendation_text(w, reasons))
        # the seed run itself is never a candidate, even with a CI below 0
        w, _, t = cs.decide(decide_table((-0.05, -0.07, -0.03), [("a", -0.01, -0.02, -0.005, 0.227)]))
        self.assertIsNone(w)
        self.assertFalse(t.loc[t["run"] == "seed", "wins"].iloc[0])
        # two winners: lower val WER
        w, reasons, _ = cs.decide(decide_table((0.001, -0.01, 0.01),
                                               [("a", -0.02, -0.03, -0.01, 0.217),
                                                ("b", -0.03, -0.04, -0.02, 0.207)]))
        self.assertEqual(w, "b")
        # current model missing
        w, reasons, _ = cs.decide(decide_table((0.0, 0.0, 0.0), [], cur_ok=False))
        self.assertIsNone(w)

    def test_recommendation_commands(self):
        run = "whisper-small-mnsc-sg-15k-heldout-lr1e-5-wu10"
        text = cs.recommendation_text(run, [])
        self.assertIn(f"test {run} once", text)
        for c in (
            f"python scripts\\evaluate_whisper_speaker_split.py --run_name {run} --heldout_test part3_15k_heldout",
            f"python scripts\\evaluate_whisper_speaker_split.py --run_name {run} --heldout_extra_test part3_15k_heldout_extra_test",
            "python scripts\\bootstrap_wer_ci.py --paired results\\eval_whisper-small-mnsc-sg-15k-heldout_ho-part3_15k_heldout_fb.csv "
            f"results\\eval_{run}_ho-part3_15k_heldout_fb.csv",
            "python scripts\\bootstrap_wer_ci.py --paired results\\eval_whisper-small-mnsc-sg-15k-heldout_hx-part3_15k_heldout_extra_test_fb.csv "
            f"results\\eval_{run}_hx-part3_15k_heldout_extra_test_fb.csv",
        ):
            self.assertIn(c, text)
        # the printed test commands parse and produce the files the bootstrap lines name
        a = ev.parse_args(["--run_name", run, "--heldout_test", "part3_15k_heldout"])
        self.assertEqual(ev.output_paths(run, ev.run_suffix(a))[0].name,
                         f"eval_{run}_ho-part3_15k_heldout_fb.csv")
        self.assertIn("keep the current model", cs.recommendation_text(None, ["x: no"]))

    def test_inputs(self):
        write_val(self.tmp, "m", "_val_fb", 4, wrong_rows=[0])
        s = cs.read_summary(self.tmp / "results" / "summary_m_val_fb.csv")
        self.assertAlmostEqual(s["val_WER"], 0.05)
        self.assertEqual((s["loop_rows"], s["samples"]), (0, 4))
        self.assertIsNone(cs.read_summary(self.tmp / "nope.csv"))
        write_ckpt(self.tmp, "m", {"learning_rate": 1e-5, "warmup_ratio": 0.1, "seed": 42})
        self.assertEqual(cs.best_from_trainer_state(self.tmp / "checkpoints" / "m"),
                         {"best_step": 10, "best_epoch": 0.5, "best_eval_loss": 0.6123})
        self.assertIsNone(cs.best_from_trainer_state(self.tmp / "checkpoints" / "nope"))
        self.assertEqual(cs.run_hparams(self.tmp / "checkpoints" / "m", rs.BASELINE_HPARAMS),
                         ({"learning_rate": 1e-5, "warmup_ratio": 0.1, "seed": 42}, "logged"))
        self.assertEqual(cs.run_hparams(self.tmp / "checkpoints" / "x", rs.BASELINE_HPARAMS)[1], "planned")

    def _fixture(self, smoke=False, with_all=True):
        root = self.tmp
        info = root / "data" / "part3_15k_heldout"
        info.mkdir(parents=True)
        (info / "speaker_split_info.json").write_text(json.dumps({"split": {"val_conversations": []}}),
                                                      encoding="utf-8")
        n = 40
        base_wrong = list(range(0, 40, 2))                 # 20 wrong rows
        write_val(root, rs.BASELINE_RUN, "_val_fb", n, base_wrong)
        write_ckpt(root, rs.BASELINE_RUN)
        suffix = rs.val_eval_suffix(smoke)
        names = [rs.run_name(r, smoke) for r in rs.SWEEP_RUNS]
        m = 16 if smoke else n
        write_val(root, names[0], suffix, m, [i for i in base_wrong if i < m] + [1])  # seed: 1 more error
        write_val(root, names[1], suffix, m, [])                                       # perfect: wins
        if with_all:
            write_val(root, names[2], suffix, m, [i for i in base_wrong if i < m])   # same as current
        for nm, r in zip(names, rs.SWEEP_RUNS):
            write_ckpt(root, nm, {k: r[k] for k in ("learning_rate", "warmup_ratio", "seed")})
        return names

    def test_main_end_to_end(self):
        names = self._fixture()
        argv = ["--cluster", "speaker", "--n_boot", "200"]
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cs.main(argv, root=self.tmp), 0, out.getvalue())
        log = out.getvalue()
        self.assertIn(f"Recommendation: test {names[1]} once.", log)
        csv = cs.default_out(False, self.tmp)
        t = pd.read_csv(csv)
        self.assertEqual(t["run"].tolist(), [rs.BASELINE_RUN] + names)
        self.assertEqual(t.loc[t["run"] == names[1], "wins"].iloc[0], True)
        self.assertEqual(t.loc[t["run"] == names[2], "wins"].iloc[0], False)
        self.assertEqual(t.loc[t["run"] == names[1], "cluster_unit"].iloc[0], "speaker")
        self.assertEqual(int(t.loc[t["run"] == names[1], "n_clusters"].iloc[0]), 20)
        self.assertEqual(t.loc[t["run"] == names[1], "hparams_source"].iloc[0], "logged")
        self.assertEqual(int(t.loc[t["run"] == rs.BASELINE_RUN, "best_step"].iloc[0]), 10)
        before = csv.read_bytes()
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cs.main(argv, root=self.tmp), 2)   # never overwritten
        self.assertIn("never overwritten", out.getvalue())
        self.assertEqual(csv.read_bytes(), before)
        inp = self.tmp / "results" / f"eval_{rs.BASELINE_RUN}_val_fb.csv"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(cs.main(argv + ["--out", str(inp)], root=self.tmp), 2)

    def test_pending_writes_nothing_and_smoke_pairs_subset(self):
        names = self._fixture(smoke=True, with_all=False)
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cs.main(["--smoke", "--cluster", "speaker", "--n_boot", "50"],
                                     root=self.tmp), 0)
        log = out.getvalue()
        self.assertIn("SMOKE MODE", log)
        self.assertIn("pending", log)
        self.assertIn("nothing written", log)
        self.assertFalse(cs.default_out(True, self.tmp).exists())
        t, _ = cs.build_table(True, self.tmp, "speaker", None, 50, 42)
        self.assertEqual(int(t.loc[t["run"] == names[1], "n_rows"].iloc[0]), 16)   # matched rows only
        self.assertEqual(t["status"].tolist(), ["ok", "ok", "ok", "pending"])


if __name__ == "__main__":
    unittest.main()
