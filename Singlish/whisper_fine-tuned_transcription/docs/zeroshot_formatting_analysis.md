# Zero-shot formatting-mismatch diagnostic

This diagnostic asks how much of the base model's (un-fine-tuned `openai/whisper-small`) WER comes from written-style formatting rather than mishearing. The numbers here are not official results.

- "official" is the corpus WER recomputed from the per-row CSV. It equals the eval summary.
- "diagnostic" is the corpus WER after Whisper's EnglishTextNormalizer is applied to the reference and the prediction of both models. It is a sensitivity check only.
- The CSV text is already `normalize_for_evaluation()` output, so punctuation and symbols such as '%' were removed before this diagnostic runs.
- Fillers counted: uh, um, uhm, er, erm, hmm, mm, mhm.
- Spelling pairs: colour/color, favourite/favorite, realise/realize, organise/organize, centre/center, theatre/theater, neighbour/neighbor, behaviour/behavior, travelled/traveled, programme/program, metres/meters, metre/meter, licence/license.

## Full test (4,000 rows), plain: base vs old 15k (PART3 seen-speaker for old 15k)

- base: `results/eval_whisper-small-zeroshot.csv`
- fine-tuned: `results/eval_whisper-small-mnsc-sg-15k.csv`

### WER: official vs diagnostic (Whisper EnglishTextNormalizer on both sides)

| group | rows | base official | base diagnostic | fine-tuned official | fine-tuned diagnostic | empty refs after normalizer |
|---|---|---|---|---|---|---|
| Overall | 4000 | 0.2915 | 0.2782 | 0.2250 | 0.2176 | 0 |
| ASR-PART1-Test | 3000 | 0.1072 | 0.1004 | 0.1230 | 0.1179 | 0 |
| ASR-PART3-Test | 1000 | 0.3921 | 0.3750 | 0.2807 | 0.2718 | 0 |

### Formatting counts (on the CSV text)

| group | model | rows | pred_digit_ref_none | ref_digit_pred_none | ref_numword_pred_digit | ref_digit_tokens | pred_digit_tokens | ref_fillers | pred_fillers | fillers_dropped | fillers_added | ref_apostrophe_tokens | pred_apostrophe_tokens | ref_uk_pred_us | ref_us_pred_uk |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Overall | base | 4000 | 224 | 1 | 190 | 2 | 474 | 1056 | 32 | 1030 | 6 | 3068 | 3672 | 42 | 6 |
| Overall | fine-tuned | 4000 | 0 | 1 | 0 | 2 | 0 | 1056 | 1397 | 268 | 609 | 3068 | 3980 | 16 | 17 |
| ASR-PART1-Test | base | 3000 | 46 | 0 | 29 | 0 | 47 | 0 | 0 | 0 | 0 | 525 | 513 | 18 | 0 |
| ASR-PART1-Test | fine-tuned | 3000 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 3 | 525 | 546 | 8 | 3 |
| ASR-PART3-Test | base | 1000 | 178 | 1 | 161 | 2 | 427 | 1056 | 32 | 1030 | 6 | 2543 | 3159 | 24 | 6 |
| ASR-PART3-Test | fine-tuned | 1000 | 0 | 1 | 0 | 2 | 0 | 1056 | 1394 | 268 | 606 | 2543 | 3434 | 8 | 14 |

Column meanings: `pred_digit_ref_none` = rows where the prediction has a digit and the reference none; `ref_digit_pred_none` = the reverse; `ref_numword_pred_digit` = rows where the reference has a spelled number word and the prediction a digit (and the reference no digit); `ref_digit_tokens` / `pred_digit_tokens` = words containing a digit; `ref_fillers` / `pred_fillers` = filler words; `fillers_dropped` / `fillers_added` = per-row shortfall / excess of fillers in the prediction; `ref_apostrophe_tokens` / `pred_apostrophe_tokens` = words with an apostrophe (contractions); `ref_uk_pred_us` = rows where the reference uses the British form of a listed word and the prediction the US form; `ref_us_pred_uk` = the reverse.

## Held-out rows (3,122), temperature fallback: base vs held-out model

- base: `results/eval_whisper-small-zeroshot_ho-part3_15k_heldout_fb.csv`
- fine-tuned: `results/eval_whisper-small-mnsc-sg-15k-heldout_ho-part3_15k_heldout_fb.csv`

The ASR-PART3-Test group here is PART3-Test, 122 held-out rows.

### WER: official vs diagnostic (Whisper EnglishTextNormalizer on both sides)

| group | rows | base official | base diagnostic | fine-tuned official | fine-tuned diagnostic | empty refs after normalizer |
|---|---|---|---|---|---|---|
| Overall | 3122 | 0.1480 | 0.1394 | 0.1363 | 0.1314 | 0 |
| ASR-PART1-Test | 3000 | 0.1071 | 0.1002 | 0.1157 | 0.1119 | 0 |
| ASR-PART3-Test | 122 | 0.3375 | 0.3198 | 0.2318 | 0.2216 | 0 |

### Formatting counts (on the CSV text)

| group | model | rows | pred_digit_ref_none | ref_digit_pred_none | ref_numword_pred_digit | ref_digit_tokens | pred_digit_tokens | ref_fillers | pred_fillers | fillers_dropped | fillers_added | ref_apostrophe_tokens | pred_apostrophe_tokens | ref_uk_pred_us | ref_us_pred_uk |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Overall | base | 3122 | 61 | 0 | 43 | 0 | 79 | 112 | 1 | 112 | 1 | 819 | 844 | 23 | 1 |
| Overall | fine-tuned | 3122 | 0 | 0 | 0 | 0 | 0 | 112 | 134 | 28 | 50 | 819 | 863 | 7 | 4 |
| ASR-PART1-Test | base | 3000 | 46 | 0 | 29 | 0 | 47 | 0 | 0 | 0 | 0 | 525 | 511 | 18 | 0 |
| ASR-PART1-Test | fine-tuned | 3000 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 7 | 525 | 526 | 6 | 3 |
| ASR-PART3-Test | base | 122 | 15 | 0 | 14 | 0 | 32 | 112 | 1 | 112 | 1 | 294 | 333 | 5 | 1 |
| ASR-PART3-Test | fine-tuned | 122 | 0 | 0 | 0 | 0 | 0 | 112 | 127 | 28 | 43 | 294 | 337 | 1 | 1 |

## Extra set (1,969), temperature fallback: base vs held-out model

- base: `results/eval_whisper-small-zeroshot_hx-part3_15k_heldout_extra_test_fb.csv`
- fine-tuned: `results/eval_whisper-small-mnsc-sg-15k-heldout_hx-part3_15k_heldout_extra_test_fb.csv`

### WER: official vs diagnostic (Whisper EnglishTextNormalizer on both sides)

| group | rows | base official | base diagnostic | fine-tuned official | fine-tuned diagnostic | empty refs after normalizer |
|---|---|---|---|---|---|---|
| Overall | 1969 | 0.3350 | 0.3129 | 0.2250 | 0.2155 | 0 |
| PART3-heldout-train-clips | 1969 | 0.3350 | 0.3129 | 0.2250 | 0.2155 | 0 |

### Formatting counts (on the CSV text)

| group | model | rows | pred_digit_ref_none | ref_digit_pred_none | ref_numword_pred_digit | ref_digit_tokens | pred_digit_tokens | ref_fillers | pred_fillers | fillers_dropped | fillers_added | ref_apostrophe_tokens | pred_apostrophe_tokens | ref_uk_pred_us | ref_us_pred_uk |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Overall | base | 1969 | 325 | 0 | 304 | 2 | 827 | 2046 | 77 | 1990 | 21 | 5052 | 5701 | 45 | 26 |
| Overall | fine-tuned | 1969 | 0 | 1 | 0 | 2 | 0 | 2046 | 2264 | 397 | 615 | 5052 | 5663 | 9 | 38 |
| PART3-heldout-train-clips | base | 1969 | 325 | 0 | 304 | 2 | 827 | 2046 | 77 | 1990 | 21 | 5052 | 5701 | 45 | 26 |
| PART3-heldout-train-clips | fine-tuned | 1969 | 0 | 1 | 0 | 2 | 0 | 2046 | 2264 | 397 | 615 | 5052 | 5663 | 9 | 38 |

## Example

Extra-set row 36 (word errors of the base model: 35 official, 11 diagnostic):

- reference: "fifty a week for food so that's two hundred …"
- base model: "50 a week for food so that's 200 …"

## Reproduce

The command is read-only and writes nothing to `results/`. Add `--out_md PATH` to save the report.

```powershell
python scripts\analyze_zeroshot_formatting.py `
  --pair "Full test (4,000 rows), plain" results\eval_whisper-small-zeroshot.csv results\eval_whisper-small-mnsc-sg-15k.csv `
  --pair "Held-out rows (3,122), temperature fallback" results\eval_whisper-small-zeroshot_ho-part3_15k_heldout_fb.csv results\eval_whisper-small-mnsc-sg-15k-heldout_ho-part3_15k_heldout_fb.csv `
  --pair "Extra set (1,969), temperature fallback" results\eval_whisper-small-zeroshot_hx-part3_15k_heldout_extra_test_fb.csv results\eval_whisper-small-mnsc-sg-15k-heldout_hx-part3_15k_heldout_extra_test_fb.csv
```
