# Hyperparameters and settings: Whisper-small MNSC fine-tuning

This file lists every setting that affects our results: what it does, the value we used, and what we learned from it.

## Terminology

| Kind | Meaning | Examples |
|---|---|---|
| Training hyperparameters | Control how the model learns. They are baked into the saved weights. | learning rate, batch size, epochs |
| Data hyperparameters | Control what the model learns from. | dataset size, speaker cap, PART1 share |
| Decoding (inference) hyperparameters | Control how text is generated from a trained model. They are not saved in the weights, so they can be changed at any time without retraining. | `no_repeat_ngram_size`, `temperature_fallback`, thresholds |
| Reproducibility / engineering settings | Change speed, memory use or randomness, not what the model learns (in principle). | `seed`, `fp16`, `batch_size` at eval |

The decoding settings (`no_repeat_ngram_size`, `temperature_fallback`, `compression_ratio_threshold`, `logprob_threshold`) are commonly called decoding hyperparameters or generation parameters. `seed` is not a hyperparameter in the usual sense. It only makes random choices repeatable.

Caveat: the first decoding comparisons (the n-gram rows in "Results so far") were made on the test set, which makes those numbers slightly optimistic, so they are exploratory. Decoding settings are chosen on each model's own validation split (`--split val`), and the test split is run once with the chosen setting. Validation was also used for early stopping and best-checkpoint choice, which is standard and fine for ranking decoding settings. Commands and the per-model validation splits: README "Choosing decoding settings: validation first, test once".

---

## 1. Data hyperparameters (`scripts\preprocess_mnsc_data.py`)

| Setting | Value used | Purpose | What happened |
|---|---|---|---|
| `--max_samples` | 15000 (4k model: about 5,000 clips, 4,010 train / 990 val) | Number of PART3 training clips to collect | Going from 4k to 15k improved PART3 plain WER from 0.328 to 0.281. This was the main training-side gain. |
| `--val_fraction` | 0.1 | Share of speakers held out for validation | 15k: 13,584 train / 1,416 val clips, 618 / 68 speakers |
| `--max_clips_per_speaker` | 50 | Stops a few talkative speakers from dominating the data | Keeps speaker diversity high (686 speakers) |
| `--min_duration` / `--max_duration` | off | Filter clips by length | Not used |
| `--part1_fraction` | 0.0 (off) | Mixes some PART1 (read speech) into training | Kept off for a fair comparison with 4k. 15k PART1 plain WER got slightly worse than 4k (0.120 to 0.123), so 0.1–0.2 is a candidate for the next retrain. |
| `--seed` | 42 | Makes the speaker split reproducible | Same split every run |
| Speaker-aware split | always on | No speaker appears in both train and val | Prevents optimistic validation loss |
| `--rows_per_shard`, `--row_group_size` | 512, 8 | Storage layout for lazy FLAC loading | Engineering only. It lets 15k clips fit on a 15.8 GB RAM machine. |
| `--heldout_test_conversations` | 0 (off); 40 for the held-out run | Keeps N randomly chosen PART3-Test conversations, all of their voices (either role) and all PART1-Test voices out of train and val. The rows are listed in `heldout_test_manifest.json`. | Held-out build: 1,969 train rows of 40 conversations / 80 voices dropped, 0 dropped for PART1 voices. Gives PART3-Test, 122 held-out rows (unseen speakers); the other 878 PART3-Test rows remain seen-speaker (partial hold-out). Excluding all 322 test conversations was not feasible (only 21 conversations, ~1k clips, would remain). |
| `--split_by_conversation` | off (implied by the flag above) | Train/val split by conversation-voice components instead of speaker1 id | Held-out build: 303 components = 303 conversations, 273 train / 30 val, 0 voices shared (15k: 108 of 122 val voices were in train) |
| `--max_source_shards` | 0 = all; 30 for the held-out run | Reads only the first K source shards | Held-out build uses only the cached shards: 14,918 clips instead of 15,000 |
| `build_heldout_extra_test.py --data_name part3_15k_heldout` | no cap, same 30 shards | Extra set (`PART3-heldout-train-clips`, 1,969 clips): every PART3-Train clip of the 40 held-out conversations, minus exact duplicates of the 122 held-out rows (segment id, audio hash, or text + duration) | 1,969 clips, 40 conversations, 80 voices, 15.19 h, 0 duplicates dropped, 2 boundary overlaps (< 0.6 s) kept. Separate group `PART3-heldout-train-clips`, never pooled with PART3-Test. Unseen-speaker only for the held-out model (`whisper-small-mnsc-sg-15k-heldout`; 15k trained on 1,607 of these clips, 4k train+val held 559). |

**ASR-PART2 was not added.** PART3 training already contains 122 of 128 checked Singapore terms (missing: NDP, LRT, OCBC, kaya / kaya toast, Bukit Timah, Bukit Batok). PART2 spells acronyms letter by letter ("m r t") while PART3 writes "mrt". PART2 is read speech like PART1, and 514 of the 810 PART1-Test speakers already appear in two sampled PART2 files, so adding it would leak test speakers into training. Details and term recall: [docs/part2_terms_analysis.md](docs/part2_terms_analysis.md).

## 2. Training hyperparameters (`scripts\train_whisper_speaker_split.py`)

| Setting | Value | Purpose | Notes |
|---|---|---|---|
| Base model | `openai/whisper-small` | Pretrained starting point | whisper-medium is a possible next step (more VRAM needed) |
| `BATCH_SIZE` | 2 per step | Clips per forward pass | Limited by 8 GB VRAM |
| `GRADIENT_ACCUMULATION` | 4 | Adds up gradients over 4 steps | Effective batch = 2 × 4 = 8 |
| `--learning_rate` (`LEARNING_RATE`) | 2e-5 | Step size of weight updates (peak of the linear schedule) | Standard range for Whisper fine-tuning (1e-5 to 3e-5). 1e-5 and 3e-5 are tried in the sweep below. |
| `--warmup_ratio` | 0.0 | Fraction of all optimizer steps over which the LR rises linearly from 0 to its peak | Passed as `TrainingArguments.warmup_steps`; in transformers 5.16.1 a value below 1 there is a ratio (0.1 of 5,067 steps = 507 steps). The sweep tries 0.1. |
| `--seed` | 42 | Training randomness: data order and dropout | 42 is the HF default that all existing models used. The sweep's seed-43 run measures how much WER moves from the seed alone. |
| LR scheduler / weight decay | HF defaults (linear decay, 0 weight decay, AdamW) | How the learning rate changes over time | Not changed. Warmup is set with `--warmup_ratio` (above); see "Learning-rate sweep". |
| `--epochs` | 3 | Maximum passes over the training data | Early stopping can end training sooner |
| `--eval_steps` | 0 = every half epoch | How often validation loss is checked and a checkpoint saved | Gives early stopping more points to choose from |
| Early stopping patience | 2 evaluations (min_delta 1e-4) | Stops when validation loss stops improving | Prevents overfitting and saves time |
| `load_best_model_at_end` | True, by `eval_loss` | Keeps the best checkpoint, not the last one | |
| `fp16` | True | Half-precision maths | Engineering: less memory, faster |
| `gradient_checkpointing` | True | Recomputes activations to save memory | Engineering: fits on 8 GB |
| `predict_with_generate` | False | Skips slow text generation during validation | Validation uses loss, not WER |
| `--max_steps` | -1 (full run) | Cap on steps, for smoke tests only | |

`--learning_rate`, `--warmup_ratio` and `--seed` default to the values used for all models before the
sweep, so existing commands build the same training arguments (checked by a unit test). The values used are printed at start and saved under
`hyperparameters` in `checkpoints/<run>/resource_usage.json`. A resume with different values is
refused.

### Learning-rate sweep

Purpose: check whether the training hyperparameters, not only the decoding settings, can improve the
held-out model (`whisper-small-mnsc-sg-15k-heldout`, LR 2e-5, no warmup, seed 42). Three extra runs,
run overnight by `python scripts\run_lr_sweep.py` in this order:

| Run name | LR | Warmup | Seed | Purpose |
|---|---|---|---|---|
| `whisper-small-mnsc-sg-15k-heldout-seed43` | 2e-5 | 0 | 43 | Noise floor: same config, only the seed differs |
| `whisper-small-mnsc-sg-15k-heldout-lr1e-5-wu10` | 1e-5 | 0.1 (507 steps) | 42 | Lower LR |
| `whisper-small-mnsc-sg-15k-heldout-lr3e-5-wu10` | 3e-5 | 0.1 (507 steps) | 42 | Higher LR |

Fixed for all runs: data `part3_15k_heldout` (not rebuilt), 3 epochs, eval every half epoch, early
stopping patience 2, best checkpoint by eval loss, batch 2 x 4, fp16, linear decay, AdamW, no weight
decay. Each run is scored on its own val split (1,411 clips) with temperature fallback decoding.

Selection uses val only. `python scripts\compare_lr_sweep.py` compares each run with the held-out
model on the same val rows, using a paired cluster bootstrap (1000 resamples, 95 % CI, seed 42; the
30 val conversations are the clusters). Decision rule, fixed before the sweep:

- Noise floor = |val WER of the seed-43 run − val WER of the held-out model|.
- A run wins only if the 95 % CI of its WER difference (run − held-out model) lies entirely below 0 AND
  its improvement is larger than the noise floor.
- The seed run is the noise reference, never a candidate. If both LR runs win, the lower val WER is
  picked. If none wins, the held-out model stays.

Only a winner is then evaluated on the test sets, once (PART3-Test, 122 held-out rows and the extra set,
temperature fallback decoding), and compared with the held-out model by paired bootstrap. Caveats: 30 val
conversations give wide CIs, and one extra seed is a single sample of the seed noise, not its
distribution.

Results (sweep run 2026-10-29 17:45 to 2026-10-30 00:54; about 2 h 05 min training and 16–18 min val eval per run; source `results\lr_sweep_comparison_val_fb.csv`):

| Run | Best step / epoch | Best eval loss | Val WER | Val CER | Loop rows | WER diff vs held-out model [95 % CI] | Wins? |
|---|---|---|---|---|---|---|---|
| held-out model (`whisper-small-mnsc-sg-15k-heldout`) | 3376 / 2.0 | 0.6654 | 0.2369 | 0.1573 | 1 | (reference) | |
| seed43 | 3376 / 2.0 | 0.6627 | 0.2353 | 0.1566 | 1 | −0.0016 [−0.0041, +0.0006] | (noise floor) |
| lr1e-5-wu10 | 3376 / 2.0 | 0.6608 | 0.2361 | 0.1567 | 1 | −0.0008 [−0.0026, +0.0010] | no |
| lr3e-5-wu10 | 3376 / 2.0 | 0.6739 | 0.2405 | 0.1599 | 1 | +0.0036 [+0.0007, +0.0062] | no (worse) |

Eval loss per half epoch for the sweep runs (steps 844 / 1688 / 2532 / 3376 / 4220 / 5064, from `docs/training_logs/<run>/resource_usage.json`, `eval_history`):

| Run | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 |
|---|---|---|---|---|---|---|
| seed43 | 0.7213 | 0.6806 | 0.6757 | **0.6627** | 0.6955 | 0.6912 |
| lr1e-5-wu10 | 0.7327 | 0.6821 | 0.6754 | **0.6608** | 0.6737 | 0.6726 |
| lr3e-5-wu10 | 0.7588 | 0.7054 | 0.6964 | **0.6739** | 0.7099 | 0.7040 |

Decision: "**keep the current model**" (the held-out model). Changing only the seed moved val WER by 0.0016 (the noise floor). LR 1e-5 with warmup was within that noise and its CI contains 0. LR 3e-5 with warmup was worse, and its CI lies entirely above 0. No sweep run was scored on the test sets. In every run, eval loss was lowest at epoch 2 and rose at epochs 2.5 and 3, so early stopping kept the epoch-2 checkpoint and more epochs would not help. (The held-out model's own [`trainer_state.json`](docs/training_logs/whisper-small-mnsc-sg-15k-heldout/trainer_state.json) only records evals up to its best step, so its epoch 2.5 and 3 losses are not available.)

## 3. Decoding hyperparameters (`scripts\evaluate_whisper_speaker_split.py`, `scripts\transcribe.py`; shared code in `scripts\whisper_decoding.py`)

Temperature fallback is the default decoding in our scripts (since 2026-10-11), chosen by the validation sweeps on both 15k models (below). `--no_temperature_fallback` gives plain greedy decoding. `no_repeat_ngram_size` stays off. Nothing is written to the model's `generation_config.json`, so a raw `generate()` / HF pipeline call on a saved model is still plain greedy unless the caller passes these options (see README "Using the model in an app").

| Setting | Default | Purpose |
|---|---|---|
| `--no_repeat_ngram_size N` | off | Forbids any N-token sequence from appearing twice in one transcript. Breaks repetition loops, but can also block real repeats. |
| `--temperature_fallback` | on (default; the flag is accepted and has no effect) | Decodes greedily first. If the output looks like a loop or low-confidence output, it re-decodes that clip with sampling at temperature 0.2, 0.4 … 1.0. This is OpenAI's own anti-loop method. |
| `--no_temperature_fallback` | off | Plain greedy decoding, as in all "plain" rows below; eval output files keep the untagged names |
| `--overwrite` | off | Existing eval/summary files are never replaced without it |
| `--compression_ratio_threshold` | 1.35 | Used with temperature fallback. Highly repetitive output compresses well (high zlib ratio), so a ratio above the threshold triggers a retry. HF measures it on token IDs, so 1.35 here ≈ OpenAI's 2.4 on text. |
| `--logprob_threshold` | -1.0 | Used with temperature fallback. Retries when the model's average confidence is too low. It can also retry hard non-loop clips. Set it to -999 to test only the loop check. |
| `--seed` | 42 | Makes sampled temperature fallback retries reproducible |
| `--max_new_tokens` | off | Hard limit on output length |
| `--batch_size` | 8 | Clips per `generate()` call. Engineering only. |
| `language="en"`, `task="transcribe"` | fixed | Forces English transcription |

### Results so far (corpus WER, 4,000 test clips: 3,000 PART1 + 1,000 PART3)

Scoring: references and predictions are normalised the same way before scoring. `<SpeakerN>:` labels are removed, `[]`, `{}`, `()` and `#…#` markers are removed but their contents kept, text is lowercased, all punctuation except apostrophes is removed, and whitespace is collapsed. WER and CER are corpus-level (total errors / total reference words or characters in a group).

| Model / decoding | Overall | PART1 | PART3 | Loop rows* |
|---|---|---|---|---|
| 4k plain | 0.2541 | 0.1197 | 0.3275 | – |
| 4k nrng3 | 0.2132 | 0.1138 | 0.2675 | 1 |
| 4k nrng5 | 0.2008 | 0.1138 | 0.2483 | 1 |
| 15k plain | 0.2250 | 0.1230 | 0.2807 | 10 |
| 15k nrng3 | 0.2054 | 0.1150 | 0.2547 | 1 |
| 15k nrng5 | 0.1926 | 0.1149 | 0.2351 | 1 |
| 4k / 15k temperature fallback | not run on test (chosen on val first, see below) | | | |
| 4k / 15k temperature fallback + nrng5 | not run on test | | | |
| base (zero-shot) plain | 0.2915 | 0.1072 | 0.3921 | 10 |
| base (zero-shot) temperature fallback | 0.2620 | 0.1071 | 0.3466 | 0 |

Base (zero-shot) = un-fine-tuned `openai/whisper-small` at Hub commit `973afd24965f72e36ca33b3055d56a652f456b4d`, scored with `--model_id openai/whisper-small` (same normalisation, language/task forcing and decoding code; files `results\summary_whisper-small-zeroshot.csv` and `_fb.csv`). Its PART3 rows are not seen-speaker rows, unlike the 4k / 15k PART3 numbers. Paired old 15k plain − base plain: Overall −0.0664 [−0.0888, −0.0466], PART3 −0.1113 [−0.1416, −0.0807], PART1 +0.0158 [+0.0040, +0.0354].

*Loop row = prediction has more than 3 × the reference word count (`n_pred_gt3x_ref` in the summary CSVs). The single remaining flagged row is a truncated label, not a real loop.

### Evaluation statistics (bootstrap CIs, `scripts\bootstrap_wer_ci.py`)

| Setting | Value | Meaning |
|---|---|---|
| `--n_boot` | 1000 | Bootstrap resamples |
| `--confidence` | 0.95 | Percentile interval (2.5th to 97.5th percentile of the resampled values) |
| Cluster unit | PART3-Test and extra set: conversation; PART1-Test: speaker; val: conversation if known, else speaker; Overall: stratified (each group resampled within itself, then summed) | Rows of one conversation/speaker are correlated, so whole clusters are resampled with replacement |
| `--seed` | 42 | Fresh `numpy.random.default_rng(seed)` per file and group, so results are repeatable and independent of file order |
| Statistic | corpus WER = Σ word errors / Σ reference words over the resampled rows (CER: characters) | Same numbers as the summary CSVs (checked to 1e-9) |
| `--paired` | off | Two runs on identical rows, the same cluster draws for both; CI of B − A |

Demo on the existing 15k nrng5 test CSV: Overall WER 0.1926, 95 % CI [0.1845, 0.2005]; PART1 0.1149 [0.1074, 0.1225] (810 speakers); PART3 0.2351 [0.2234, 0.2475] (322 conversations). Paired nrng5 − plain on 15k: Overall WER −0.0324 [−0.0479, −0.0177], PART3 −0.0457 [−0.0678, −0.0255], PART1 −0.0081 [−0.0254, +0.0003]. So nrng5's gain is real overall and on PART3; on PART1 the CI just touches 0 (the gain there comes from a few loop rows).

### Validation sweep (corpus WER/CER on each model's own validation split)

| Model / decoding | WER | CER | Loop rows* | Notes |
|---|---|---|---|---|
| 15k plain | 0.2896 | 0.1960 | 20 | |
| 15k nrng5 | 0.2277 | 0.1516 | 2 | |
| 15k nrng10 | 0.2235 | 0.1489 | 2 | |
| 15k temperature fallback | **0.2211** | **0.1472** | 2 | 4 rows still above the compression-ratio threshold |
| 15k temperature fallback + nrng5 | 0.2255 | 0.1497 | 2 | 1 row above the threshold |
| 4k plain | 0.3092 | 0.2087 | 16 | other 4k settings not run yet |
| held-out model plain | 0.2966 | 0.2035 | 16 | own val: 1,411 clips, 30 conversations not used in training |
| held-out model nrng5 | 0.2421 | 0.1609 | 1 | |
| held-out model nrng10 | 0.2414 | 0.1607 | 2 | |
| held-out model temperature fallback | **0.2369** | **0.1573** | 1 | 4 rows still above the compression-ratio threshold |
| held-out model temperature fallback + nrng5 | 0.2408 | 0.1598 | 1 | 1 row above the threshold |

On 15k validation, temperature fallback alone is best, with nrng10 close behind. Adding nrng5 on top of temperature fallback made it slightly worse, so the two don't stack. The chosen setting for 15k is temperature fallback alone; the test split should be run once with it.

The held-out model's val gives the same ranking (temperature fallback > temperature fallback + nrng5 ≈ nrng10 ≈ nrng5 >> plain), so temperature fallback was used for all its test runs. Its val WER is higher than 15k's (0.237 vs 0.221) because its val split holds out whole conversations and their voices, while 15k's val shares 108 of 122 voices with its training set. The two val numbers are not comparable.

### Caveat: PART3-Test speakers were seen in training

A speaker-overlap audit (`scripts/audit_test_speaker_overlap.py`, output `results/test_speaker_overlap.csv`) found:

- PART3-Test (1,000 rows) comes from 322 conversations, and all 322 also appear in ASR-PART3-Train. 976 test rows (15k model) and 965 (4k model) are spoken by a voice the model heard in training. No PART3-Test row is unseen by both models.
- PART1-Test (3,000 rows, 810 speakers) has zero overlap with training.
- The train/val split groups only by `speaker1`. Roles swap within a conversation, so 108 of 122 15k-val voices also appear in 15k training.

So PART3-Test WER means "new speech from seen speakers". PART1-Test is the unseen-speaker result, though it is read speech, so it also involves a domain shift from the conversational training data. Small subsets of unseen-voice rows (24 and 35 rows) gave mixed results and are too small to measure the size of the effect. Excluding all 322 conversations would leave only ~1k training clips, so instead 40 PART3-Test conversations were held out and the 15k model was retrained without them (`whisper-small-mnsc-sg-15k-heldout`). Results are below.

### Held-out results: unseen-speaker PART3 (temperature fallback, 1000 cluster-bootstrap resamples, 95 % CI, seed 42)

Training of the held-out model (`whisper-small-mnsc-sg-15k-heldout`): data `part3_15k_heldout` (13,507 train / 1,411 val clips; 273 / 30 conversations). Eval loss at each half-epoch: 0.721, 0.681, 0.680, 0.665. The best checkpoint, which is the saved model, is step 3376 (epoch 2.0 of 3, 5,067 max steps). The old 15k's best is also at epoch 2.0 (step 3396; eval loss 0.689, 0.650, 0.646, 0.632). Sources: [held-out model `trainer_state.json`](docs/training_logs/whisper-small-mnsc-sg-15k-heldout/trainer_state.json), [old 15k `trainer_state.json`](docs/training_logs/whisper-small-mnsc-sg-15k/trainer_state.json). Only the best checkpoint is kept, and its `trainer_state.json` records history up to that step only, so evals after epoch 2 are not visible there. Loss flattened between epochs 1 and 1.5 for both runs.

Held-out model, final numbers:

| Test set | Clips | Clusters | WER [95 % CI] | CER [95 % CI] |
|---|---|---|---|---|
| PART1-Test | 3,000 | 810 speakers | 0.116 [0.108, 0.124] | 0.053 [0.049, 0.058] |
| PART3-Test, 122 held-out rows | 122 | 40 conversations | 0.232 [0.203, 0.260] | 0.153 [0.133, 0.173] |
| `PART3-heldout-train-clips` (extra set) | 1,969 | 40 conversations | 0.225 [0.207, 0.242] | 0.147 [0.134, 0.159] |
| Overall (PART1 + 122 held-out rows) | 3,122 | 850 | 0.136 [0.128, 0.145] | 0.069 [0.064, 0.074] |

The plain run of the held-out model on the same 3,122 rows (`--no_temperature_fallback`, `summary_whisper-small-mnsc-sg-15k-heldout_ho-part3_15k_heldout.csv`, no CI computed) gives Overall 0.1592, PART1 0.1212, PART3 0.3355.

Base model (zero-shot `openai/whisper-small`, commit `973afd2…`), same rows:

| Test set | Decoding | Overall WER | PART1 WER | PART3 / extra-set WER | PART3 / extra-set CER | Loop rows |
|---|---|---|---|---|---|---|
| 3,122 rows (PART1 + PART3-Test, 122 held-out rows) | plain | 0.1585 | 0.1072 | 0.3960 | 0.2923 | 1 |
| 3,122 rows | temperature fallback | 0.1480 [0.1380, 0.1574] | 0.1071 [0.0997, 0.1153] | 0.3375 [0.3030, 0.3705] | 0.2405 [0.2131, 0.2669] | 0 |
| `PART3-heldout-train-clips` (extra set, 1,969) | temperature fallback | | | 0.3350 [0.3100, 0.3590] | 0.2386 [0.2194, 0.2573] | 0 |

Paired held-out model − base (temperature fallback, same rows): PART3-Test, 122 held-out rows −0.1057 [−0.1255, −0.0855], extra set −0.1100 [−0.1199, −0.1003], PART1 +0.0086 [+0.0041, +0.0138], Overall −0.0117 [−0.0170, −0.0064]. Fine-tuning lowered unseen-speaker PART3 WER by about 0.11; on PART1 read speech the base model was slightly better. Part of the base model's error is written-style formatting (digits, dropped fillers, US spelling); see [docs/zeroshot_formatting_analysis.md](docs/zeroshot_formatting_analysis.md).

Paired comparison on the same 3,122 rows (diff = held-out model − old 15k; the old 15k heard these PART3 speakers, but not these clips):

| Group | Old 15k WER | Held-out model WER | Diff [95 % CI] | CER diff [95 % CI] |
|---|---|---|---|---|
| PART1-Test | 0.1146 | 0.1157 | +0.0011 [−0.0020, +0.0044] | +0.0003 [−0.0011, +0.0017] |
| PART3-Test, 122 held-out rows | 0.2301 | 0.2318 | +0.0017 [−0.0076, +0.0110] | +0.0025 [−0.0038, +0.0093] |
| Overall | 0.1351 | 0.1363 | +0.0012 [−0.0016, +0.0042] | +0.0006 [−0.0008, +0.0022] |

Old 15k CER on the same rows: PART1-Test 0.0529, PART3-Test, 122 held-out rows 0.1504.

What this shows:

- Unseen-speaker PART3 WER is about 0.23 (0.225 on the extra set, 0.232 on PART3-Test, 122 held-out rows). The two sets agree, and the extra set's CI is about half as wide.
- All paired CIs contain 0, so the old and held-out models can't be told apart. Having heard the PART3-Test speakers in training did not measurably lower the old 15k's WER. Its full-test PART3 number (0.235 with nrng5, CI [0.223, 0.248]) is in the same range as the unseen-speaker number.
- PART3 is still about twice as hard as PART1 (0.23 vs 0.12). This comes from the conversational audio, not from leakage.

Caveats:

- Both PART3 sets come from the same 40 conversations and 80 voices, so they are not independent confirmations. The CI reflects variation across those 40 conversations only.
- 122 rows is small, so the PART3 diff CI (±~0.009) can only rule out speaker-familiarity effects larger than about 1 WER point.
- The extra set is a fair test only for the held-out model. The old 15k trained on 1,607 of its clips (and had 131 more in val), and the 4k model's train+val held 559. Old-model numbers on it are not reported.
- The two models had slightly different data: 14,918 vs 15,000 clips, and a conversation-disjoint vs speaker1-only val split.
- The 4k model was not compared on the held-out rows. Its `_ho-` run is optional (README step 5).

### What we learned

- Loops are the main cause of very high PART3 errors. Plain 15k has 8 PART3 loop rows and 2 PART1 loop rows ("and australia" repeated). A few loops can add thousands of wrong words to the corpus WER.
- nrng3 removes the loops but over-blocks. On 15k PART3 it made 623 rows slightly worse and 103 better. It blocks real speech such as "sadly sadly" and spelled-out "v i b e … v i b e".
- nrng5 is a better tradeoff. It gives the same loop removal with about 4–5× less collateral damage (15k PART3: 197 rows worse, 74 better, losing 5.3 WER points against 46.3 gained). Real repeats survive. Its weak spot is longer-phrase loops ("multi multi multi divide divide…"), which a 5-gram block can't catch.
- Temperature fallback: in 20-row smoke tests it fixed the sampled loops (15k: 204 → 76 words, 225 → 114 words) and left PART1 unchanged. On the full 15k validation split it was the best setting (WER 0.2211 vs 0.2896 plain and 0.2277 nrng5). Compare runs row by row, not just on totals, because `logprob_threshold` may also re-sample clips that aren't loops.
- nrng10 beat nrng5 on 15k val (0.2235 vs 0.2277) with the same loop count, so a larger N protects real repeats without losing loop control here.
- Production note: plain numbers are what a raw `model.generate()` / HF pipeline call gives, because `generation_config.json` is unchanged. Our scripts (evaluation and `transcribe.py`) default to temperature fallback. Always label which decoding a number comes from: headline the temperature fallback number (e.g. "PART3 unseen-speaker WER 0.232 [0.203, 0.260], temperature fallback decoding") with the plain number next to it (held-out val: plain 0.2966 vs temperature fallback 0.2369; PART3-Test, 122 held-out rows: plain 0.3355 vs temperature fallback 0.232).

## 4. Candidate next experiments

| Experiment | Kind | Expected effect |
|---|---|---|
| 4k on the held-out rows (`--run_name whisper-small-mnsc-sg-4k --heldout_test part3_15k_heldout --temperature_fallback`) | Methodology | Completes the three-model comparison on identical rows |
| Bootstrap CIs for the plain held-out model run (`bootstrap_wer_ci.py results\eval_whisper-small-mnsc-sg-15k-heldout_ho-part3_15k_heldout.csv`) | Methodology | CI for the plain number next to the temperature fallback one |
| Out-of-domain test on another MNSC part (e.g. PART4), after a speaker audit | Methodology | Generalisation beyond PART3-style conversations. Report it as a separate group. |
| Full 1,000-row test-split run of 15k with temperature fallback (the val-chosen setting) | Decoding | Final temperature fallback number on the original test split (current full-test number is nrng5) |
| 4k val sweep | Decoding | Choose 4k's setting the same way |
| `--logprob_threshold -999` with temperature fallback | Decoding | Separates the effect of the loop check from the confidence check |
| nrng 7 / 10 | Decoding | Fewer real repeats blocked, possibly weaker loop control |
| Decode-time prompting with ~30 local terms (`prompt_ids`) | Decoding | May raise term recall without retraining; check WER and hallucination on the term subset (147 clips, 240 named-term occurrences) |
| `--part1_fraction 0.1–0.2` retrain | Data | Recover PART1 accuracy |
| Small labelled test set from our own microphones | Methodology | Measures the domain gap to real app audio |
| whisper-medium | Model size | Likely a larger gain, but needs more VRAM and time |
