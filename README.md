# Out-of-State Tag Holder Challenge — submission

Decision-support scorer for the 12,000 challenge cases: a review classification, three
class probabilities, and a review-priority value, at T0 and T1.

**Start with [SUMMARY.md](SUMMARY.md)** — one page.
**[METHOD.md](METHOD.md)** is the full write-up: what the data turned out to be, how
identities are resolved, why the model is ordinal, the validation results, and the eight
ideas that were tested and rejected.

## Submission file

`case_predictions.csv` — 24,000 rows in the template's exact schema and row order. Every
row's three probabilities sum to exactly 1 in decimal; `review_priority` is in [0, 1].

## Supporting files

| file | what it is |
|---|---|
| `case_audit.csv` | the same 24,000 rows plus the evidence behind each score: newest record per feed with its date, evidence volume and freshness, direction of the T1 update, and the signed contribution of each feature block. Intended for the staff member working the queue. |
| `model_card.json` | fitted thresholds and standardised coefficients, plus the cross-validated metrics |
| `diagnostics_output.txt` | output of `diagnostics.py` — the evidence for every structural claim in METHOD.md |
| `model_selection_output.txt` | output of `model_selection.py` — model family, feature set, and aggregation comparisons |
| `FORM_ANSWERS.md` | prepared copy-paste text for the submission form (approach, limitations, at three lengths) |
| `dist/*.pdf` | the summary and full write-up as PDFs; rebuild with `./build_pdfs.sh` |

## Running it

The challenge data is not in this repo — it is the organisers' to distribute. Point the
code at your own copy.

```bash
pip install -r requirements.txt

# the challenge package must be findable: put the Identify_Out_of_State_Tag_Holders
# folder next to this code, or point OOS_DATA_DIR at it
export OOS_DATA_DIR=/path/to/Identify_Out_of_State_Tag_Holders

python predict.py       # ~35 s: resolve -> featurise -> cross-validate -> score -> write
                        # deterministic: reproduces case_predictions.csv byte for byte
python validate.py      # schema, row order, probability sums, ranges
python diagnostics.py   # reproduces the structural findings
python model_selection.py
```

## Code

| module | responsibility |
|---|---|
| `er.py` | name normalisation, edit distance, the fuzzy-lookup index |
| `resolve.py` | entity resolution — name matching, and vehicle-keyed title grouping |
| `pipeline.py` | data loading; builds the tidy evidence table |
| `featurize.py` | per-candidate, per-phase features |
| `model.py` | proportional-odds ordinal logistic model |
| `predict.py` | end-to-end run; writes the submission, audit file and model card |
| `validate.py` | submission-format checks |
| `diagnostics.py`, `model_selection.py` | reproduce the claims in METHOD.md |
| `experiments.py` | the ideas that were tested and rejected (METHOD.md section 8) |

## Headline numbers

Grouped 5-fold cross-validation (20 repeats) on the 300 development candidates, scored at
both phases (600 rows):

| | model | baseline |
|---|---:|---:|
| log loss | 0.934 | 1.096 |
| accuracy | 0.527 | 0.355 |
| macro F1 | 0.528 | 0.175 |
| macro AUC | 0.719 | 0.500 |
| priority AUC for `review_warranted` | 0.781 | 0.500 |

T1 scores better than T0 (log loss 0.914 vs 0.954), i.e. the model does move on the later
evidence rather than ignoring it.

The output supports staff review. It is not a residency, fee, or enforcement
determination.
