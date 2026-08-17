# Out-of-State Tag Holder Review — one-page summary

**Deliverable:** `case_predictions.csv` — 24,000 rows (12,000 cases × T0/T1) in the
template's exact schema. Full write-up in `METHOD.md`.

## The problem behind the problem

The package gives no key linking `candidate_records.csv` to the six evidence feeds — only
names, and the names are deliberately corrupted. Getting the identity layer right mattered
more than the choice of classifier, so that is where the work went.

Using `vehicle_ref` on the title feed as an independent check, the corruption pattern is
**3 good rows plus 1 replaced outright**: cluster the owner names inside each vehicle's
four rows and 98% split three-and-one, with the fourth name unrelated to the other three.
Grouping the licence feed by date of birth reproduces the same pattern, so it is the
generator's noise model, not a quirk of one table.

Two consequences:

- The title feed **can** recover its fourth row, because `vehicle_ref` says which vehicle
  it belongs to. Doing so lifts that feed from 38,582 to 47,581 linked rows and its signal
  from rho +0.25 to +0.37.
- The other feeds **cannot**. Forcing an assignment there made the signal *worse* — an
  unrelated name carries an unrelated state. Accepting ~80% recall is the better trade,
  and 9,369 of 12,000 candidates land on exactly three address rows as predicted.

## The classes are ordered, so the model is

In the development labels no case ever jumps from `review_not_warranted` to
`review_warranted` between phases — every move passes through `insufficient_evidence`.
That is one latent score with two thresholds, so the model is a **proportional-odds
ordinal logistic**: one direction vector and two cut-points instead of three independent
classifiers, which matters at 300 labelled candidates. On identical features it beats
multinomial logistic (0.953), random forest (0.981) and gradient boosting (1.038).

The direction is **recency**: cases whose recent records point to Delaware are the ones
warranting review. A 150-day exponential half-life beats every alternative aggregation
tried, including a change-point fit. The work-location feed is complete, uncorrupted and
correlates −0.07 with the answer — it is excluded from the pooled score.

## Results

Grouped 5-fold cross-validation, 20 repeats, 600 labelled rows:

| | model | baseline |
|---|---:|---:|
| log loss | **0.934** | 1.096 |
| accuracy | **0.527** | 0.355 |
| macro F1 | **0.528** | 0.175 |
| macro AUC | **0.719** | 0.500 |
| priority ranking AUC | **0.781** | 0.500 |

T1 scores better than T0 (0.914 vs 0.954) — the model moves on the later evidence.
Probabilities are calibrated with no post-hoc fitting (predicted vs observed agree within
±0.05 across quintiles). Errors sit on the adjacent class: only 39 of 600 rows cross the
whole scale.

`review_priority = p(warranted) + 0.5 × p(insufficient)` — a case that warrants review
takes full priority, a case the evidence cannot decide still needs a person and takes half.

## Honest ceiling

Eight further ideas were tested and rejected, including semi-supervised self-training
(worse at every setting) and three routes to recovering the unattributable rows. Each
candidate has roughly fourteen state-bearing records split between two states, so the
latent direction can only be estimated so precisely. Macro AUC ~0.72 looks close to what
this data supports.

## For the reviewer

`case_audit.csv` carries the same 24,000 rows plus the evidence behind each score — newest
record per feed with its date, evidence freshness, T1 update direction, and the signed
contribution of each feature block — so a case can be checked without reading the code.
`diagnostics.py` and `model_selection.py` reproduce every number quoted above.

The output supports staff review. It is not a residency, fee, or enforcement determination.
