# Whisper-small fine-tuning on Singaporean English (MNSC PART3)

This project fine-tunes `openai/whisper-small` on conversational Singapore English from
[AudioLLMs/Multitask-National-Speech-Corpus-v1-extend](https://huggingface.co/datasets/AudioLLMs/Multitask-National-Speech-Corpus-v1-extend)
(`ASR-PART3-Train`). It evaluates the models on a fixed 4,000-clip test manifest
(3,000 `ASR-PART1-Test` + 1,000 `ASR-PART3-Test`). Training never uses the test configs.

## Layout

```
scripts/
  preprocess_mnsc_data.py          build data/<DATA_NAME>/ (download, cap, speaker split, FLAC shards)
  verify_mnsc_dataset.py           sanity-check a built dataset
  train_whisper_speaker_split.py   fine-tune (lazy audio loading) -> models/<RUN_NAME>/
  evaluate_whisper_speaker_split.py  WER/CER on the fixed manifest (or --split val) -> results/
  whisper_decoding.py              shared decoding (temperature fallback default) used by eval + transcribe
  transcribe.py                    transcribe your own audio files (app-style CLI)
  compare_decoding_runs.py         compare eval CSVs row by row (read-only)
  build_heldout_extra_test.py      extra set: PART3-Train clips of the held-out conversations
  bootstrap_wer_ci.py              cluster bootstrap CIs for corpus WER/CER of eval CSVs
  audit_test_speaker_overlap.py    speaker-overlap audit / held-out build check (read-only)
  create_mnsc_extended_manifest.py how data/mnsc_extended_evaluation_manifest.csv was made
  tests/                           offline unit tests (python -m unittest)
data/
  part3_15k/                       15k-clip dataset (default)
    train/shard-*.parquet          FLAC (16 kHz mono PCM_16) + text, speaker_id, duration
    val/shard-*.parquet
    speaker_split_info.json        build settings, timings, speaker distribution, split
  part3_15k_heldout/               held-out build (same layout + heldout_test_manifest.json)
  part3_15k_heldout_extra_test/    extra set: test/shard-*.parquet + extra_test_info.json
  preprocessed_train.parquet       legacy 4k-baseline data (float32), kept unchanged
  preprocessed_val.parquet
  speaker_split_info.json          legacy split; read by ../accent_classifier
                                   (prepare_mnsc_singaporean.py --exclude_speakers_json default)
  mnsc_extended_evaluation_manifest.csv  fixed 4,000-row test manifest
models/<RUN_NAME>/                 final models (whisper-small-mnsc-sg-4k = baseline)
checkpoints/<RUN_NAME>/            resumable training state + resource_usage.json
results/                           eval_<RUN_NAME>[_val].csv, summary_<RUN_NAME>[_val].csv
docs/                              reference notes: part2_terms_analysis.md, zeroshot_formatting_analysis.md,
                                   training_logs/<run>/ (trainer_state.json / resource_usage.json copies)
```

The file `data/part3_15k/speaker_split_info.json` lists the training speakers of the 15k model
under `split.train_speakers`.

## Using the Google Drive data (skip the downloads)

Our data files for the Transcription Model are on Google Drive: `https://drive.google.com/file/d/1N7H1w3fPltvVgubhAngiabqmWm_ZHmkj/view?usp=drive_link`

Put the Drive folders and files into `data\` as shown below. If Google Drive gives you a .zip file,
extract it first.

```
data\part3_15k_heldout\              held-out model: train\ + val\ shards, speaker_split_info.json, heldout_test_manifest.json
data\part3_15k_heldout_extra_test\   extra set (1,969 clips): test\ shards + extra_test_info.json
data\mnsc_extended_evaluation_manifest.csv   required for every PART1-Test / PART3-Test run
optional: data\part3_15k\ (old 15k model, same layout)
optional: data\preprocessed_train.parquet, data\preprocessed_val.parquet, data\speaker_split_info.json (4k build)
```

Then follow "Held-out run" below with these changes:
- Skip the `preprocess_mnsc_data.py` command in step 1. Skip the `build_heldout_extra_test.py` command in step 6.
- `verify_mnsc_dataset.py` and steps 2, 3 and 7 read their data only from `data\`.
- On its first run, training downloads the base model `openai/whisper-small` (~0.97 GB). Step 9 uses it too.
- Steps 4, 5 and 9 still need Hugging Face. On the first run, they download the two test configs (~1.4 GB).
  This includes PART3-Test, 122 held-out rows.
- The two `audit_test_speaker_overlap.py` commands never download. They need the raw cache below.
- If you have `data\part3_15k\`, skip step 1 of "From scratch" too.

If you also downloaded the raw Hugging Face cache folder, the scripts reuse its MNSC files. Copy the folder
`datasets--AudioLLMs--Multitask-National-Speech-Corpus-v1-extend` into `%USERPROFILE%\.cache\huggingface\hub\`.
Keep the folder name and all its subfolders. For another location, set `$env:HF_HUB_CACHE` to its parent folder.
To block all downloads, run `$env:HF_HUB_OFFLINE = "1"` in the same PowerShell window.

Offline mode also needs `openai/whisper-small` in the cache. If it is not there, run training once
without offline mode.

## From scratch (run from the project root, PowerShell or bash)

All timings come from an RTX 3070 Laptop (8 GB) with 16 GB RAM and a ~10-13 MB/s download.

```powershell
# 0. Install (Python 3.14 tested). Install the CUDA build of torch first:
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt

# 1. Build the 15k dataset -> data/part3_15k/  (~6.3 GB on disk)
#    Downloads the first 27 of 171 PART3-Train shards (~13 GB) into the HF cache:
#    ~20-25 min download + ~5 min processing. Re-runs reuse the cache (~5 min).
#    Peak RAM ~1.7 GB.
python scripts/preprocess_mnsc_data.py
#    = --data_name part3_15k --max_samples 15000 --val_fraction 0.1 --max_clips_per_speaker 50 --seed 42

# 2. Check it (~1 min)
python scripts/verify_mnsc_dataset.py --data_name part3_15k

# 3. Train -> models/whisper-small-mnsc-sg-15k/   (see the estimate below)
#    Resumes automatically from checkpoints/whisper-small-mnsc-sg-15k/ if interrupted.
python scripts/train_whisper_speaker_split.py

# 4. Evaluate -> results/eval_whisper-small-mnsc-sg-15k.csv + summary_whisper-small-mnsc-sg-15k.csv
#    (first run downloads the two test configs, ~1.4 GB). --no_temperature_fallback reproduces the
#    reported plain test numbers (untagged files); without it the script uses the default
#    temperature fallback and writes *_fb.csv instead.
python scripts/evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-15k --no_temperature_fallback

# 5. (Optional) choose decoding settings on the validation split first, then run the test split
#    once with the chosen setting - see "Choosing decoding settings" below.
```

Training estimate for 13,584 train clips: 1,698 optimizer steps/epoch (batch 2 x
grad-accum 4) and 3 epochs. Training runs an eval and saves a checkpoint every 849 steps (~half epoch).
Early stopping ends training after 2 evals without improvement. The full 3 epochs take ≈ 2 hours
(less if early stopping triggers).

A 20-step smoke run measured ~1.29 s per optimizer step and ~64 s per full validation pass
(1,416 clips). A checkpoint save took ~5 s. Peak VRAM was ~5.0 GB reserved, and peak RAM was ~3 GB.
Keep `--num_workers 0` (the default). On Windows, worker processes are re-spawned for every
evaluation (~15 s each). The lazy loader is not a bottleneck.

With `--no_temperature_fallback`, a full evaluation on 4,000 clips takes about the same time as the
4k baseline evaluation (same decoding). The default temperature fallback is slower. The extra time
depends on how many clips need re-decoding (up to 5 extra passes for those clips).

## Held-out run (40 PART3-Test conversations unseen)

ASR-PART3-Test is cut from the same 343 conversations as ASR-PART3-Train (322 of them are test
conversations). So the 15k/4k PART3-Test WER is a "seen speakers" number. If all 322 are excluded,
only 21 conversations (~1k clips) are left. So this run holds out 40 randomly chosen PART3-Test
conversations instead (seed 42, chosen at runtime).

The build removes their 80 voices (either role) and any PART1-Test voice from train AND val.
It splits train/val by conversation, so train and val share no voice either.
Result: `data/part3_15k_heldout/`, 14,918 clips (13,507 train / 1,411 val,
273 / 30 conversations, 546 / 60 voices). That is 82 fewer than 15,000, because the build reads
only the 30 cached shards.

Its `heldout_test_manifest.json` lists PART3-Test, 122 held-out rows.
Only these 122 rows and PART1-Test are unseen-speaker for the held-out model
(`whisper-small-mnsc-sg-15k-heldout`).
The other 878 PART3-Test rows are still seen-speaker rows.

```powershell
# 1. Build (~5 min, reads only the 30 cached PART3-Train shards; ~6.3 GB)
python scripts\preprocess_mnsc_data.py --data_name part3_15k_heldout --heldout_test_conversations 40 --max_source_shards 30
python scripts\verify_mnsc_dataset.py --data_name part3_15k_heldout
python scripts\audit_test_speaker_overlap.py --data_name part3_15k_heldout   # must print RESULT: PASS (all zeros)

# 2. Train -> models/whisper-small-mnsc-sg-15k-heldout/ (~2 h: 1,689 steps/epoch x 3 at ~1.31 s/step,
#    7 evals of ~67 s; resumes from checkpoints/whisper-small-mnsc-sg-15k-heldout/ if interrupted)
python scripts\train_whisper_speaker_split.py --data_name part3_15k_heldout --run_name whisper-small-mnsc-sg-15k-heldout

# 3. Validation sweep on its own val split (data/part3_15k_heldout/val, 1,411 clips)
python scripts\evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-15k-heldout --split val --no_temperature_fallback
python scripts\evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-15k-heldout --split val --no_repeat_ngram_size 5 --no_temperature_fallback
python scripts\evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-15k-heldout --split val --no_repeat_ngram_size 10 --no_temperature_fallback
python scripts\evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-15k-heldout --split val --temperature_fallback
python scripts\evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-15k-heldout --split val --temperature_fallback --no_repeat_ngram_size 5
python scripts\compare_decoding_runs.py results\eval_whisper-small-mnsc-sg-15k-heldout_val.csv results\eval_whisper-small-mnsc-sg-15k-heldout_val_nrng5.csv results\eval_whisper-small-mnsc-sg-15k-heldout_val_nrng10.csv results\eval_whisper-small-mnsc-sg-15k-heldout_val_fb.csv results\eval_whisper-small-mnsc-sg-15k-heldout_val_nrng5_fb.csv

# 4. Test ONCE with the val-chosen setting (--temperature_fallback won on both 15k val and this val)
#    Optional: its full-test PART3 number includes 878 seen-speaker rows; steps 5-8 give the unseen-speaker numbers.
python scripts\evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-15k-heldout --temperature_fallback

# 5. Fair comparison on identical rows (PART1-Test 3000 + the 122 held-out PART3-Test rows),
#    same decoding for all three models -> *_ho-part3_15k_heldout_fb.csv
python scripts\evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-15k-heldout --heldout_test part3_15k_heldout --temperature_fallback
python scripts\evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-15k --heldout_test part3_15k_heldout --temperature_fallback
python scripts\evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-4k --heldout_test part3_15k_heldout --temperature_fallback
```

Caveats:
- On the 122 rows, the held-out model has never heard the speakers. The 15k and 4k models have
  heard them (seen-speaker numbers). The gap between them estimates the speaker-familiarity effect.
- 122 rows from 40 conversations is a small sample, so expect a wide spread. The other 878
  PART3-Test rows are still "seen" by the held-out model. So its full-test PART3 number (step 4)
  is not unseen-speaker.
- PART1-Test voices are excluded by string id only. PART1 and PART3 use different id schemes, so
  0 rows were dropped. A person who recorded both parts cannot be detected.
- 14,918 clips is 0.5 % fewer than 15k's 15,000. The val split is conversation-disjoint, unlike
  15k's speaker1-only split. So val numbers are not directly comparable between the two models.

### Extra set and confidence intervals

122 held-out rows is a small sample, so two additions help. The first addition is the extra set
(`PART3-heldout-train-clips`, 1,969 clips). It holds all ASR-PART3-Train clips of the same 40
held-out conversations (80 voices, 15.19 h; no 50-clip cap).

These are the clips the held-out build excluded from training. They were checked against
PART3-Test, 122 held-out rows by segment id, audio hash, and text + duration. The check found
0 exact duplicates. 2 clips overlap a test row by under 0.6 s at a segment boundary and are kept.
The extra set is reported as its own group `PART3-heldout-train-clips` in separate `_hx-` files.
It is never pooled into the official PART3-Test number.

The second addition is cluster bootstrap 95 % confidence intervals for corpus WER/CER
(`scripts\bootstrap_wer_ci.py`).

Caveats for the extra set:
- The extra set has the same conversations and voices as the 122 held-out rows, so it is not an
  independent sample. It adds clips, not conversations. The CI is still driven by 40 conversations.
- The extra set is unseen-speaker only for the held-out model. The 15k model trained on 1,607 of
  these exact clips, and 131 more were in its val split. The 4k model's train+val contained 559 of
  them. So 15k/4k numbers on this set are training-data numbers, not a fair comparison. For a fair
  comparison, use `--heldout_test` (step 5).
- The extra set references come from the PART3-Train `answer` field (punctuated, mixed case).
  They are normalized like the train/val text. PART3-Test references come from its raw
  transcription field. So the label styles of the two groups differ slightly before normalization.

```powershell
# 6. Build the extra set (~1-2 min, reads the same 30 cached shards; ~0.83 GB) -> data/part3_15k_heldout_extra_test/
python scripts\build_heldout_extra_test.py --data_name part3_15k_heldout
python scripts\audit_test_speaker_overlap.py --data_name part3_15k_heldout --extra_test part3_15k_heldout_extra_test   # RESULT: PASS

# 7. Evaluate the extra set (the 122 held-out rows are step 5), same decoding.
#    step 5 -> eval_<RUN>_ho-part3_15k_heldout_fb.csv   extra set -> eval_<RUN>_hx-part3_15k_heldout_extra_test_fb.csv
python scripts\evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-15k-heldout --heldout_extra_test part3_15k_heldout_extra_test --temperature_fallback
python scripts\evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-15k --heldout_extra_test part3_15k_heldout_extra_test --temperature_fallback

# 8. Confidence intervals (1000 resamples, seed 42) -> results\bootstrap_*.csv
#    single: one CI per group (PART1, PART3 held-out, extra set, Overall) for each file
python scripts\bootstrap_wer_ci.py results\eval_whisper-small-mnsc-sg-15k-heldout_ho-part3_15k_heldout_fb.csv results\eval_whisper-small-mnsc-sg-15k-heldout_hx-part3_15k_heldout_extra_test_fb.csv
#    paired: held-out model (B) minus 15k (A) on the same 3,122 rows, same resampled clusters for both
python scripts\bootstrap_wer_ci.py --paired results\eval_whisper-small-mnsc-sg-15k_ho-part3_15k_heldout_fb.csv results\eval_whisper-small-mnsc-sg-15k-heldout_ho-part3_15k_heldout_fb.csv

# 9. (Optional) zero-shot baseline: un-fine-tuned openai/whisper-small on the same rows (HF cache,
#    nothing copied to models/; outputs named whisper-small-zeroshot). RTX 3070 Laptop: 4-16 min per run.
python scripts\evaluate_whisper_speaker_split.py --model_id openai/whisper-small --no_temperature_fallback   # -> eval_whisper-small-zeroshot.csv
python scripts\evaluate_whisper_speaker_split.py --model_id openai/whisper-small                              # -> eval_whisper-small-zeroshot_fb.csv
python scripts\evaluate_whisper_speaker_split.py --model_id openai/whisper-small --heldout_test part3_15k_heldout --no_temperature_fallback
python scripts\evaluate_whisper_speaker_split.py --model_id openai/whisper-small --heldout_test part3_15k_heldout   # -> *_ho-part3_15k_heldout_fb.csv
python scripts\evaluate_whisper_speaker_split.py --model_id openai/whisper-small --heldout_extra_test part3_15k_heldout_extra_test   # -> *_hx-..._fb.csv
python scripts\bootstrap_wer_ci.py results\eval_whisper-small-zeroshot_ho-part3_15k_heldout_fb.csv
python scripts\bootstrap_wer_ci.py results\eval_whisper-small-zeroshot_hx-part3_15k_heldout_extra_test_fb.csv
#    paired: fine-tuned (B) minus base (A), same rows and decoding
python scripts\bootstrap_wer_ci.py --paired results\eval_whisper-small-zeroshot_ho-part3_15k_heldout_fb.csv results\eval_whisper-small-mnsc-sg-15k-heldout_ho-part3_15k_heldout_fb.csv
python scripts\bootstrap_wer_ci.py --paired results\eval_whisper-small-zeroshot_hx-part3_15k_heldout_extra_test_fb.csv results\eval_whisper-small-mnsc-sg-15k-heldout_hx-part3_15k_heldout_extra_test_fb.csv
python scripts\bootstrap_wer_ci.py --paired results\eval_whisper-small-zeroshot.csv results\eval_whisper-small-mnsc-sg-15k.csv   # plain; old 15k PART3 = seen speakers
#    read-only formatting diagnostic (digits, fillers, spelling; Whisper-normaliser WER as a sensitivity check)
python scripts\analyze_zeroshot_formatting.py --pair "held-out rows" results\eval_whisper-small-zeroshot_ho-part3_15k_heldout_fb.csv results\eval_whisper-small-mnsc-sg-15k-heldout_ho-part3_15k_heldout_fb.csv
```

Reading the output:
- The bootstrap resamples clusters, not rows. A cluster is a conversation for PART3 and the extra
  set, and a speaker for PART1. Overall resamples each group separately and adds them up.
- In paired mode, `diff` = B − A. A CI that excludes 0 means a real difference. A CI that contains 0
  means the runs cannot be told apart on this data.
- With only 40 held-out conversations, expect wide PART3 intervals (several WER points).

### Held-out results

On the held-out val split, temperature fallback won again (WER 0.2369, vs nrng10 0.2414, nrng5
0.2421, temperature fallback + nrng5 0.2408 and plain 0.2966). So all test runs of the held-out
model use it. `HYPERPARAMETERS.md` ("Held-out results") has the results with 95 % CIs, the paired
comparison with the old 15k model and the caveats. When you quote a number, always state the decoding.

The plain run on the same rows is
`python scripts\evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-15k-heldout --heldout_test part3_15k_heldout --no_temperature_fallback`.
Its result files already exist, and the script never overwrites result files. To redo the run,
add `--overwrite`.

## Optional: overnight learning-rate / seed sweep

This step retrains the held-out model three more times on the same data (`data/part3_15k_heldout`,
no rebuild). Only the learning rate, warmup and seed change. The runner then scores each run on its
own val split with temperature fallback decoding. It does not touch the test splits.

The three runs are `whisper-small-mnsc-sg-15k-heldout-seed43` (seed 43, the noise floor; runs first),
`…-lr1e-5-wu10` and `…-lr3e-5-wu10`. The last two use LR 1e-5 / 3e-5 with 10 % warmup (507 steps).
`HYPERPARAMETERS.md` ("Learning-rate sweep") has the settings and results.

The seed run goes first because the comparison cannot decide anything without it. The expected
time is about 6.8 h (3 x ~2.05 h training + 3 x ~12 min val eval). Each run uses about 3.7 GB of
disk (best checkpoint 2.8 GB + model 0.9 GB). The total is about 11.1 GB at the end. It is up to
16.6 GB while the last run saves a checkpoint. If less than 20 GB is free, the runner refuses to start.

```powershell
# Check the plan first (prints the 6 commands, runs and creates nothing)
python scripts\run_lr_sweep.py -DryRun
# Start the sweep (from the project root)
python scripts\run_lr_sweep.py
```

Before leaving it overnight:

- Keep the laptop plugged in.
- Set Windows sleep to Never while the sweep runs. Go to Settings > System > Power & battery >
  Screen and sleep. Set "When plugged in, put my device to sleep after" to Never. Set the lid-close
  action to "Do nothing", or keep the lid open. This setting is in Control Panel > Power Options >
  Choose what closing the lid does.
- Training needs about 5.3 GB of the 8 GB VRAM.

Logs: each run writes `logs\sweep\<run>.log` (every line timestamped). The runner appends the final
status table to `logs\sweep\sweep_status.log`. If a run fails, the runner logs it and starts the
next run.

If the sweep was interrupted (crash, power cut, Ctrl+C), run the same command again. The runner
does not train a run again if its `models\<run>\` is complete. It does not recompute an existing
val summary. An interrupted run resumes from its last checkpoint. At most half an epoch (about
20 min) is lost.

Early-stopping patience is not saved in checkpoints. So a resumed run may train one or two
evaluations longer than an uninterrupted one. The runner refuses to resume with different settings
(exit 2).

If a run failed, read its log first. If a crash happened during the final save, `models\<run>\` can
be incomplete. In that case, delete only that run's `models\<run>\` folder and run the command again.
An eval that stopped half way can leave `results\eval_<run>_val_fb.csv` without a summary. In that
case, delete only that file.

Do not run the training command of a finished run again by hand. Its best checkpoint is still
there, so the command resumes and trains further. The runner checks this for you.

In the morning:

```powershell
python scripts\compare_lr_sweep.py
```

The script prints one row per model. Each row shows LR, warmup, seed, best step/epoch, best eval
loss, val WER/CER and loop rows. It also shows the paired bootstrap WER diff vs the held-out model
with its 95 % CI (30 conversation clusters). The script then applies the decision rule from
`HYPERPARAMETERS.md` ("Learning-rate sweep"). It prints either "keep the current model" or
"test <run> once" with the exact test commands.

When all runs have results, the script writes `results\lr_sweep_comparison_val_fb.csv`. It never
overwrites this file. Run the printed test commands only if the script says "test <run> once".

## Smoke tests (minutes)

```powershell
python -m unittest discover -s scripts/tests -t . -v
python scripts/preprocess_mnsc_data.py --data_name part3_smoke500 --max_samples 500
python scripts/train_whisper_speaker_split.py --data_name part3_15k --run_name smoke-whisper-mnsc --max_steps 20 --eval_steps 10 --max_eval_samples 16
python scripts/evaluate_whisper_speaker_split.py --run_name smoke-whisper-mnsc --max_rows 16 --no_temperature_fallback
python scripts/evaluate_whisper_speaker_split.py --run_name smoke-whisper-mnsc --split val --max_rows 16 --no_temperature_fallback
```

Smoke outputs go to `models/smoke-*`, `checkpoints/smoke-*`, `results/*_rows16.csv` and
`results/*_val_rows16.csv`. Delete them afterwards.

The sweep runner smoke test takes about 7 min and needs ~12 GB free temporarily. It runs
3 x 20-step trainings under `smoke-` names plus 64-row val evals. Then it runs the comparison on them.

```powershell
python scripts\run_lr_sweep.py -SmokeTest
python scripts\compare_lr_sweep.py --smoke
```

Its outputs are `models\smoke-*`, `checkpoints\smoke-*`, `results\*smoke*` and
`logs\sweep\smoke-*`. Delete them afterwards.

## Options (data/training defaults keep results comparable to the 4k baseline)

- `preprocess_mnsc_data.py --max_clips_per_speaker 50`: sets the cap per speaker (0 = off). At 15k
  clips the mean is ~22 clips/speaker, so 50 only trims outliers.
- `preprocess_mnsc_data.py --part1_fraction 0.1`: OFF by default. Mixes in a share of
  `ASR-PART1-Train` (read speech). Avoid it for the main comparison. PART1-Test is already
  easy for the model, and the mix dilutes the conversational target.
- `preprocess_mnsc_data.py --data_name NAME`: every build gets its own folder. The script never
  replaces an existing folder unless you add `--overwrite`. Even then, it replaces only a folder
  that this script built.
- `preprocess_mnsc_data.py --heldout_test_conversations N`: OFF by default (0). At runtime, the
  script picks N PART3-Test conversations whose voices occur in no other conversation (sorted,
  shuffled with `--seed`). Before the speaker cap, it drops every training row of those
  conversations. It also drops every row with one of their voices (either role) or with a
  PART1-Test voice. The flag implies `--split_by_conversation` and writes
  `heldout_test_manifest.json`. Do not combine it with `--part1_fraction`.
- `preprocess_mnsc_data.py --split_by_conversation`: OFF by default. Splits train/val by connected
  components of the conversation-voice graph, so no conversation or voice is in both. If you train
  on such a build without `--run_name`, the training script derives `...-<N>k-conv`. So it never
  resumes the speaker-split checkpoint of the same size.
- `preprocess_mnsc_data.py --max_source_shards K`: OFF by default (0 = all). Reads only the first K
  source shards (e.g. 30 = the cached ones).
- `evaluate_whisper_speaker_split.py --heldout_test DATA_NAME`: test split only, with any `--run_name`.
  Scores PART1-Test 3000 plus the PART3-Test rows listed in `data/<DATA_NAME>/heldout_test_manifest.json`.
  Output files get `_ho-<DATA_NAME>`, so they never replace test or val files.
- `evaluate_whisper_speaker_split.py --model_id REPO_ID [--model_revision REV]`: OFF by default.
  Evaluates an un-fine-tuned Hub model (zero-shot baseline, e.g. `openai/whisper-small`) loaded
  through the HF cache. Nothing is copied into `models/`, and no `generation_config.json` is written.
  The output run name is `<repo name>-zeroshot` (e.g. `whisper-small-zeroshot`) unless you give
  `--run_name`. If you give `--run_name`, it must not name a folder under `models/`. Test split only
  (with `--heldout_test` / `--heldout_extra_test` as usual).

  `--model_revision` (default `main`) takes a branch, tag or commit. The script prints the resolved
  commit. It also writes it to the summary CSV columns `model_id`, `model_revision`, `model_commit`
  (fine-tuned summaries keep their columns). Loading, fp16, English/transcribe forcing and decoding
  use the same code as for the fine-tuned models.
- `analyze_zeroshot_formatting.py --pair LABEL BASE_CSV FT_CSV [--out_md PATH]`: read-only diagnostic of
  formatting mismatches (digits, fillers, contractions, British/US spelling). It also gives a WER with
  Whisper's English normaliser on both sides (sensitivity check, not an official number). It never
  writes to `results/`.
- `audit_test_speaker_overlap.py --data_name NAME`: read-only check of a held-out build (0 shared
  conversations/voices with its held-out rows and PART1-Test, 0 voices shared by train and val).
  It writes nothing and exits 0 only if the build is clean. To also check the extra set, add
  `--extra_test EXTRA_NAME`. The extra-set check needs all its conversations/voices in the manifest
  and none in train/val or PART1-Test. No clip may equal a held-out test row. The flag also prints
  how many of its clips the 15k/4k models saw.
- `build_heldout_extra_test.py --data_name NAME [--out_name NAME_extra_test]`: writes all PART3-Train
  clips of the held-out conversations of `data/NAME/` to `data/<out_name>/test/`. It also writes
  `extra_test_info.json` (counts, dropped duplicates, per-clip conversation/voices). It reads only
  the manifest's `train_shards_scanned` source shards. It never touches `data/NAME/`.
- `evaluate_whisper_speaker_split.py --heldout_extra_test DATA_NAME`: test split only, with any
  `--run_name`. Do not combine it with `--heldout_test`. Scores the extra set as the group
  `PART3-heldout-train-clips`. Output files get `_hx-<DATA_NAME>`.
- Per-row eval CSVs include `conversation_id` (PART3 rows, '' for PART1 and val) and `speaker_id`
  (PART1 speaker, PART3 speaker1, val speaker). These columns come after `CER`.
  `compare_decoding_runs.py` ignores them.
- `bootstrap_wer_ci.py FILE [FILE ...]` / `--paired A B`: cluster bootstrap CIs of corpus WER/CER
  (`--n_boot 1000 --confidence 0.95 --seed 42`, `--out`, `--overwrite`). Old CSVs without the cluster
  columns get them from the cached test parquet files. To use another location, set `--snapshot_dir`.
- `bootstrap_wer_ci.py --val_conversations`: OFF by default. Val CSVs have an empty
  `conversation_id`, so val rows cluster by speaker. With this flag, the script looks up the
  conversation of each val clip by its `sample_id`. It reads the cached `ASR-PART3-Train` shards for
  this. The lookup is in memory, and the CSVs are not changed. Val rows then cluster by conversation (30 for
  `part3_15k_heldout`).
- `train_whisper_speaker_split.py --run_name NAME --max_steps N --eval_steps N --num_workers N`:
  `--data_name legacy` trains on the old 4k files. The script never overwrites an existing
  `models/<RUN_NAME>/` unless you add `--overwrite_model`.
- `train_whisper_speaker_split.py --learning_rate 2e-5 --warmup_ratio 0.0 --seed 42`: the defaults
  give exactly the training arguments used for all existing models. `--warmup_ratio` is a fraction
  of the total optimizer steps (0.1 of 5,067 = 507 steps). The script prints the values at start. It
  saves them under `hyperparameters` in `checkpoints/<RUN_NAME>/resource_usage.json`.
- `evaluate_whisper_speaker_split.py --no_repeat_ngram_size 3` / `--max_new_tokens 225`: OFF by
  default. Use them to measure the repetition-loop effect separately. Output files get a suffix
  (`_nrng3`, `_mnt225`). They combine with the default temperature fallback (`_nrng3_fb`). To
  reproduce the plain-decoding `_nrng3` / `_nrng5` numbers, add `--no_temperature_fallback`. The
  summary column `n_pred_gt3x_ref` counts predictions with more than 3x the reference word count.
- Temperature fallback is the default decoding in our scripts (`evaluate_whisper_speaker_split.py`,
  `transcribe.py`). `--no_temperature_fallback` gives plain greedy decoding (untagged file names, as
  in all plain results). `--temperature_fallback` is still accepted and has no effect. Do not give
  both flags together; that is an error. Do not give the thresholds with `--no_temperature_fallback`;
  that is also an error.

  Each clip is first decoded greedily. It is re-decoded if the output looks like a repetition loop
  (token compression ratio > `--compression_ratio_threshold`, default 1.35). It is also re-decoded
  if it has low confidence (average log-probability < `--logprob_threshold`, default -1.0).
  Re-decoding uses sampling at temperature 0.2, 0.4, ... 1.0 until the output passes (the last
  attempt is kept). `HYPERPARAMETERS.md` section 3 gives the thresholds and how they were chosen.

  Temperature fallback can be combined with `--no_repeat_ngram_size`. Output files get `_fb` (plus
  `_cr<x>` / `_lp<x>` / `_seed<N>` for non-default values), e.g. `eval_whisper-small-mnsc-sg-15k_fb.csv`,
  `eval_whisper-small-mnsc-sg-15k_nrng5_fb.csv`. Temperature fallback runs add a per-row
  `compression_ratio` column (≈ the value HF checks). They also add a summary column
  `n_cr_gt_threshold` (predictions still above the threshold after all temperatures). Sampled
  re-decodes depend on `--seed` (default 42). So results can change slightly with a different seed,
  GPU or library version.

  ```powershell
  python scripts\evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-15k --temperature_fallback
  python scripts\evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-4k --temperature_fallback
  # combined with the n-gram block:
  python scripts\evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-15k --no_repeat_ngram_size 5 --temperature_fallback
  # plain greedy (untagged eval_whisper-small-mnsc-sg-15k.csv):
  python scripts\evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-15k --no_temperature_fallback
  ```
- `evaluate_whisper_speaker_split.py --overwrite`: without it, existing eval/summary files are never
  replaced. The run stops (exit code 2) before it loads any data or model.

### Choosing decoding settings: validation first, test once

Choose the decoding options (`--no_repeat_ngram_size`, `--temperature_fallback`, the thresholds) on
validation data, not on the test set. Protocol: sweep the options with `--split val`. Pick one
setting per model. Then run the test split ONCE with that setting. Report that number.

The `_nrng3` / `_nrng5` test results are exploratory comparisons. They were made on the test set
before this protocol was used. The validation sweeps chose temperature fallback, which is the
default. So the plain and nrng-only sweep lines below pass `--no_temperature_fallback`.

- By default, each model uses its own validation split. `whisper-small-mnsc-sg-15k` ->
  `data/part3_15k/val` (1,416 clips, 68 speakers, read lazily one clip at a time).
  `whisper-small-mnsc-sg-4k` -> the legacy `data/preprocessed_val.parquet` (990 clips, 136 speakers,
  read lazily). Its audio is one compressed block, decompressed once (~1.8 GB) and converted to
  float32 one clip at a time. A run peaks at ~3.6 GB RAM. Other run names default to `part3_15k`.
- Do not share one validation set across the two models. 57 of the 68 15k-val speakers (1,228 of
  1,416 clips) are in the 4k training set. 125 of the 136 legacy-val speakers (924 of 990 clips)
  are in the 15k training set. A non-default `--data_name` prints a warning and writes
  `_val-<data_name>` files. Compare the two models with each other on the test set only.
- Validation was already used for early stopping and best-checkpoint selection (eval loss). That is
  standard practice. It makes the absolute val WER slightly optimistic. It is still fine for ranking
  decoding settings. Validation is PART3 (conversational) only, so PART1 behaviour cannot be tuned
  there.
- Output names: `_val` comes first in the suffix, e.g. `eval_whisper-small-mnsc-sg-15k_val.csv`,
  `summary_whisper-small-mnsc-sg-15k_val_nrng5_fb.csv`, `eval_whisper-small-mnsc-sg-4k_val_fb_lp-999.csv`.
  Val files never replace test files. The `config` column is `part3_15k-val` / `legacy-val`. This is
  also the summary group next to `Overall`, with the same numbers. `row_index` is the clip's position
  in the validation set, so runs can be joined on `[config, row_index]`. `--max_rows N` takes the
  first N clips.
- `compare_decoding_runs.py` takes two or more eval CSVs. For each file, it prints corpus WER/CER per
  group and the loop-row count (`n_pred_gt3x_ref`). For each file after the first, it also prints the
  rows that got better / worse / stayed the same. It also prints the summed per-row WER gain / loss
  against the first file. It only prints and writes nothing.

```powershell
# 15k model: sweep on its validation split (1,416 clips per run)
python scripts\evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-15k --split val --no_temperature_fallback
python scripts\evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-15k --split val --no_repeat_ngram_size 3 --no_temperature_fallback
python scripts\evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-15k --split val --no_repeat_ngram_size 5 --no_temperature_fallback
python scripts\evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-15k --split val --no_repeat_ngram_size 7 --no_temperature_fallback
python scripts\evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-15k --split val --no_repeat_ngram_size 10 --no_temperature_fallback
python scripts\evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-15k --split val --temperature_fallback
python scripts\evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-15k --split val --temperature_fallback --no_repeat_ngram_size 5
python scripts\evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-15k --split val --temperature_fallback --logprob_threshold -999
python scripts\compare_decoding_runs.py results\eval_whisper-small-mnsc-sg-15k_val.csv results\eval_whisper-small-mnsc-sg-15k_val_nrng3.csv results\eval_whisper-small-mnsc-sg-15k_val_nrng5.csv results\eval_whisper-small-mnsc-sg-15k_val_nrng7.csv results\eval_whisper-small-mnsc-sg-15k_val_nrng10.csv results\eval_whisper-small-mnsc-sg-15k_val_fb.csv results\eval_whisper-small-mnsc-sg-15k_val_nrng5_fb.csv results\eval_whisper-small-mnsc-sg-15k_val_fb_lp-999.csv

# 4k model: the same sweep on its own (legacy) validation split (990 clips per run)
python scripts\evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-4k --split val --no_temperature_fallback
python scripts\evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-4k --split val --no_repeat_ngram_size 3 --no_temperature_fallback
python scripts\evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-4k --split val --no_repeat_ngram_size 5 --no_temperature_fallback
python scripts\evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-4k --split val --no_repeat_ngram_size 7 --no_temperature_fallback
python scripts\evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-4k --split val --no_repeat_ngram_size 10 --no_temperature_fallback
python scripts\evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-4k --split val --temperature_fallback
python scripts\evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-4k --split val --temperature_fallback --no_repeat_ngram_size 5
python scripts\evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-4k --split val --temperature_fallback --logprob_threshold -999
python scripts\compare_decoding_runs.py results\eval_whisper-small-mnsc-sg-4k_val.csv results\eval_whisper-small-mnsc-sg-4k_val_nrng3.csv results\eval_whisper-small-mnsc-sg-4k_val_nrng5.csv results\eval_whisper-small-mnsc-sg-4k_val_nrng7.csv results\eval_whisper-small-mnsc-sg-4k_val_nrng10.csv results\eval_whisper-small-mnsc-sg-4k_val_fb.csv results\eval_whisper-small-mnsc-sg-4k_val_nrng5_fb.csv results\eval_whisper-small-mnsc-sg-4k_val_fb_lp-999.csv

# Then ONE final test run per model with the setting that won on its validation split
# (replace the flags below with that setting):
python scripts\evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-15k --no_repeat_ngram_size 5 --temperature_fallback
python scripts\evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-4k --no_repeat_ngram_size 5 --temperature_fallback
```

## Using the model in an app

`scripts\transcribe.py` transcribes your own files. It uses the same decoding code as the evaluation
(`scripts\whisper_decoding.py`). That means temperature fallback by default, `language="en"`,
`task="transcribe"`, and fp16 on GPU if available, else CPU.

```powershell
python scripts\transcribe.py --model models\whisper-small-mnsc-sg-15k-heldout path\to\audio1.wav [more files]
python scripts\transcribe.py path\to\audio1.wav --out transcripts.csv    # add --overwrite to replace transcripts.csv
python scripts\transcribe.py --no_temperature_fallback path\to\audio1.wav  # plain greedy
```

- The default `--model` is `models\whisper-small-mnsc-sg-15k-heldout`, the held-out model. It was
  validated on unseen speakers, which is closest to what an app faces. It is statistically tied with
  the old 15k model. Relative paths are resolved against the project root.
- stdout gets one line per file, `<path><TAB><transcript>`. Progress and errors go to stderr. An
  unreadable file is reported and skipped (exit code 1). `--out` writes
  `path,transcript,duration_sec,n_chunks,decoding`. If the output file exists, add `--overwrite` to
  replace it. Without `--overwrite`, the script never replaces an existing file.
- Formats: WAV, FLAC, OGG, MP3 (what libsndfile reads). Convert other formats, e.g. m4a, with
  ffmpeg. Audio is mixed down to mono and resampled to 16 kHz. Files over 30 s are cut into fixed
  30 s windows, transcribed separately and joined. The seed / threshold / `--no_repeat_ngram_size`
  options and defaults are the same as in the evaluation. Nothing is written to the model folder.

Calling the model directly: the saved `generation_config.json` is unchanged, so a plain
`model.generate()` is greedy decoding. To get the decoding that our numbers use, pass the
temperature fallback options yourself. `scripts\whisper_decoding.py` is the reference
implementation, incl. fp16 on GPU, batching and the seed:

```python
import torch, soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration

model_dir = "models/whisper-small-mnsc-sg-15k-heldout"
processor = WhisperProcessor.from_pretrained(model_dir)
model = WhisperForConditionalGeneration.from_pretrained(model_dir).eval()

audio, sr = sf.read("clip_16k_mono.wav", dtype="float32")   # 16 kHz mono, <= 30 s
features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features
with torch.no_grad():
    ids = model.generate(features, language="en", task="transcribe",
                         temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                         compression_ratio_threshold=1.35, logprob_threshold=-1.0)
print(processor.batch_decode(ids, skip_special_tokens=True)[0])
```

Advice:
- Before you transcribe, trim leading/trailing silence with a voice activity detector (VAD). Long
  silences invite hallucinated text. Our scripts do not do this.
- Whisper reads at most 30 s at a time, so longer audio needs chunking. `transcribe.py` uses fixed
  windows, which can cut a word at a boundary. VAD-based or overlapping chunks do better.
- Temperature fallback adds latency only for clips that fail the check and get re-decoded (up to
  5 extra passes). Most clips decode once.
- Re-decoded clips are sampled, so their text can vary slightly with the seed, GPU or library version.
- Real-world audio (microphones, noise, accents, topics) differs from MNSC, so the benchmark WER is
  not a promise for your app. Test on a sample of your own audio.

## Sharing without re-downloading

To use shared data, copy the data files into the same location:
- For the held-out model: `data/part3_15k_heldout/`. Training needs nothing else.
- To evaluate it: `data/part3_15k_heldout_extra_test/` and `data/mnsc_extended_evaluation_manifest.csv`.
- For the old 15k model only: `data/part3_15k/`.

Then run the preprocessing step again. It reads the cache and does not download. In "From scratch", step 1
then only re-encodes (~5 min).

To use our Google Drive files and skip the downloads, see "Using the Google Drive data (skip the downloads)".
