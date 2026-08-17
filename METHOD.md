# Potential Out-of-State Tag Holder Review — Method

A decision-support scorer for the 12,000 challenge cases. For each case it produces a
review classification, three class probabilities, and a review-priority value, at both
T0 and T1.

Everything below is reproducible from the code in this folder:
`diagnostics.py` prints the evidence for each structural claim,
`model_selection.py` prints the feature-set comparison, and
`predict.py` runs the whole thing end to end.

---

## 1. What the data actually is

Nothing in the package names a join key between `candidate_records.csv` and the six
evidence feeds. The feeds carry only names (and a date of birth on the licence feed).
Three properties of the data drive every design decision that follows.

**Each feed holds one record set per candidate.** The address, licence, title and
external feeds each hold 4.01 rows per candidate; the work-location feed and the T1
update stream each hold ~2.0. These are not coincidences across five tables — the
evidence was generated per candidate, so linking is an *assignment* problem, not an
independent per-row lookup.

**Names are corrupted on a 3-good-plus-1-garbled pattern.** The title feed is the one
place this can be verified without assuming an answer, because `vehicle_ref` groups a
vehicle's history independently of who is named on it. Clustering owner names inside
each 4-row vehicle gives a 3 + 1 split for 98% of vehicles: three rows carrying one
name, lightly corrupted (truncated to an initial, one substituted character, an
adjacent transposition, a stray space), and one row carrying a name with no
relationship to the other three. Grouping the licence feed by date of birth reproduces
the same 3 + 1 pattern, so it is the generator's noise model rather than a quirk of one
table.

**The three classes are ordered.** In the 300 development cases, no case moves from
`review_not_warranted` to `review_warranted` between phases (0 cases), and one moves the
other way. Every other move passes through `insufficient_evidence`. That is the
signature of a single latent score with two thresholds, not three unrelated categories.

| T0 \ T1 | insufficient | not warranted | warranted |
|---|---:|---:|---:|
| insufficient_evidence | 71 | 22 | 12 |
| review_not_warranted | 20 | 85 | **0** |
| review_warranted | 12 | **1** | 77 |

Reading the axis against the evidence: cases whose *recent* records point to Delaware
are labelled `review_warranted`; cases whose recent records point to another state are
`review_not_warranted`; cases in between are `insufficient_evidence`. That matches the
operational question — a vehicle whose owner now appears to be operating in Delaware is
the one a staff member should look at.

---

## 2. Identity resolution

Resolution runs in two modes, because only one feed has a key that survives name
corruption.

**Title feed — grouped by `vehicle_ref`.** Rows are clustered by name inside each
vehicle; each cluster is matched to the candidate it agrees with best; the garbled
singleton row is attached to the vehicle's dominant owner at a reduced weight. This
recovers the fourth row that name matching alone cannot reach, and it roughly doubles
the title feed's signal: linked rows go from 38,582 to 47,581 of 48,108, and Spearman rho
against the ordered label rises from +0.25 to +0.37 (printed by `diagnostics.py`,
section 5). T1 title updates carry `vehicle_ref` too and are attached the same way.

**Address, licence, external, work feeds — confidence-weighted name matching.** The
matcher normalises the `SYN…-` prefixes, blocks on a SymSpell-style delete neighbourhood
(edit distance ≤ 2, which covers substitution, insertion, deletion and transposition)
plus an explicit prefix index (which covers truncation to an initial), and scores first
and last name jointly. A date-of-birth agreement adds a bonus on the licence feed, never
a penalty. Matches are softened into weights that sum to 1 across the competing
candidates, so an ambiguous name splits its evidence rather than committing to a guess.

The garbled row in these feeds is deliberately **not** forced onto a candidate. An
earlier version did force it, using the 4-rows-per-candidate structure as a capacity
constraint. Coverage went to ~99%, but the address, licence and external signals all got
*weaker* — an unrelated name carries an unrelated state, and inserting it corrupts the
candidate's evidence. Accepting ~80% recall on those feeds is the better trade.

Resulting link rates:

| feed | rows linked | of | note |
|---|---:|---:|---|
| address | 38,740 | 48,121 | name matching; garbled row not recoverable |
| licence | 38,726 | 48,124 | name + DOB |
| external | 38,631 | 48,116 | name matching |
| work | 24,131 | 24,131 | this feed is not corrupted |
| title | 47,581 | 48,108 | via `vehicle_ref` |
| T1 updates | 20,093 | 24,000 | 4,713 via `vehicle_ref` |

---

## 3. Signal, and what carries none

Spearman rho of a recency-weighted DE share against the ordered T0 label:

| feed | rho |
|---|---:|
| title | +0.37 |
| address | +0.29 |
| external | +0.27 |
| licence | +0.14 |
| **work location** | **−0.07** |
| four feeds pooled | **+0.52** |

The work-location feed is complete, uncorrupted, and unrelated to the labels. It is
excluded from the pooled score and kept only as its own near-zero feature block, so the
model can confirm that rather than being forced to assume it.

Recency dominates. Aggregations compared on the same resolved evidence (`model_selection.py`,
section C):

| aggregation | rho |
|---|---:|
| exponential decay, 150-day half-life | **+0.517** |
| rank decay (0.5 per step back) | +0.507 |
| newest record per feed, averaged | +0.480 |
| hard window, last 180 days | +0.461 |
| change-point, post-segment mean | +0.410 |
| linear-in-time extrapolation | +0.387 |

The half-life sits on a broad plateau between 90 and 240 days; 150 days is used. That a
change-point fit does *worse* says the evidence behaves like a recency-weighted mixture
rather than a clean move date.

The observed tag state contributes independently and more weakly (rho +0.11): an
out-of-state tag raises the score a little at any level of residency evidence. There is
no interaction worth modelling.

---

## 4. Model

A **proportional-odds (ordinal) logistic model** over

```
review_not_warranted (0)  <  insufficient_evidence (1)  <  review_warranted (2)
P(y ≤ k) = sigmoid(theta_k − x·beta),  theta_0 < theta_1
```

fitted by L-BFGS with an L2 penalty (C = 0.03), implemented in `model.py`.

Why ordinal rather than a 3-way classifier: it encodes the banded transition structure
found in section 1, and it estimates one direction vector plus two thresholds instead of
three independent ones — which matters when there are only 300 labelled candidates. On
identical features and identical folds:

| model | log loss | accuracy | macro AUC | macro F1 |
|---|---:|---:|---:|---:|
| **ordinal logit** | **0.939** | **0.530** | **0.718** | **0.531** |
| multinomial logistic | 0.953 | 0.512 | 0.703 | 0.505 |
| random forest | 0.981 | 0.495 | 0.672 | 0.488 |
| hist gradient boosting | 1.038 | 0.467 | 0.655 | 0.466 |
| class prior | 1.096 | 0.355 | 0.500 | 0.175 |

**Training rows.** T0 and T1 are pooled into 600 rows with a phase indicator, so both
phases train one model. Cross-validation groups by candidate so a candidate's T0 and T1
rows never straddle a fold.

**34 features in four blocks:**

- *evidence direction* — recency-weighted DE share of the four signal feeds at 90/150/365-day
  half-lives and unweighted; the state on the newest record; how many feeds end in DE;
  disagreement between feeds; per-feed direction and newest state
- *evidence sufficiency* — volume of linked evidence, records in the last 6 and 12 months,
  age of the newest record, share of records with no state, share of limited-quality
  records, share of expired/superseded records, identity-match confidence, number of
  distinct states seen
- *T1 updates* — volume, direction, direction of the newest update, direction of the title
  and address updates, and the change against prior evidence. Encoded as signed deviations
  that are exactly zero in the T0 phase, so the block contributes nothing before the
  updates exist rather than relying on imputation.
- *observed tag* — whether the tag on the candidate record is out of state

A 22-feature version is more readable but measurably worse (CV log loss 0.949 vs 0.938 on
the same folds), so the fuller set is kept and interpretability is handled in the audit
file instead (section 7). The penalty curve is flat between C = 0.02 and C = 0.05.

---

## 5. Results

Grouped 5-fold cross-validation, 20 repeats, on all 600 labelled rows:

| metric | model | baseline |
|---|---:|---:|
| multi-class log loss | **0.934** | 1.096 (class prior) |
| accuracy | **0.527** | 0.355 (majority class) |
| macro F1 | **0.528** | 0.175 |
| macro AUC | **0.719** | 0.500 |
| priority AUC for `review_warranted` | **0.781** | 0.500 |

By phase — the model responds to the later evidence, as intended:

| | log loss | accuracy | macro AUC | priority AUC |
|---|---:|---:|---:|---:|
| T0 | 0.954 | 0.520 | 0.704 | 0.768 |
| T1 | **0.914** | **0.533** | **0.737** | **0.799** |

Confusion matrix (rows true, columns predicted, order not-warranted / insufficient /
warranted):

```
[[130  67  16]
 [ 64  88  56]
 [ 23  58  98]]
```

The off-diagonal mass sits where the ordinal structure says it should: confusion is
mostly with the adjacent class, and only 39 of 600 rows cross the whole scale.

**Calibration.** Predicted vs observed frequency in quintiles of predicted probability:

| quintile | p(warranted) pred / obs | p(not warranted) pred / obs |
|---|---|---|
| 1 | 0.06 / 0.05 | 0.08 / 0.09 |
| 2 | 0.14 / 0.19 | 0.18 / 0.17 |
| 3 | 0.25 / 0.20 | 0.31 / 0.34 |
| 4 | 0.40 / 0.42 | 0.48 / 0.44 |
| 5 | 0.64 / 0.63 | 0.71 / 0.73 |

No post-hoc recalibration is applied; the ordinal likelihood is already well calibrated
on held-out folds, and with 300 labelled candidates a second fitted calibration layer
would cost more in variance than it gains.

**Class prior.** A Kolmogorov–Smirnov test of the strongest feature, labelled subset vs
all 12,000 candidates, gives D = 0.047, p = 0.52 — the development set behaves like a
random sample, so its 35/35/30 class mix is used as the population prior rather than
being reweighted.

---

## 6. Review priority

```
review_priority = p(review_warranted) + 0.5 × p(insufficient_evidence)
```

A case that warrants review takes full priority. A case the evidence cannot decide still
needs a person to look at it — that is what `insufficient_evidence` means operationally —
so it takes half. A case that clearly does not warrant review takes almost none. The
value lies in [0, 1] by construction.

This also happens to rank slightly better than `p(review_warranted)` alone
(AUC 0.7814 vs 0.7804), but the gap is inside the noise on 600 rows; the definition is
chosen for what it means to a queue, not for the third decimal place.

---

## 7. Using the output

`case_predictions.csv` is the submission file, in the template's exact schema and row
order.

`case_audit.csv` is a supporting file with the same 24,000 rows plus the evidence behind
each score, so a reviewer can see why a case is where it is without reading the code:

- `latest_{address,licence,title,external}_{state,date}` — the newest record per feed
- `de_share_recent`, `de_share_all_time` — the direction of the evidence
- `evidence_records_linked`, `days_since_newest_record` — how much there is and how fresh
- `t1_update_direction` — `toward DE` / `away from DE` / `mixed` / `no update`
- `driver_summary` — signed contribution of each of the four feature blocks, largest first
- `top_drivers` — the three individual features with the largest contribution

Signs read as: **positive pushes toward `review_warranted`, negative toward
`review_not_warranted`.** Lead with `driver_summary`; individual coefficients are
ridge-shrunk across collinear direction features, so a single feature's sign should not
be read on its own, whereas the block sums are stable.

The output supports staff review. It is not a residency, fee, or enforcement
determination, and `review_warranted` means "a person should look at this", nothing more.

---

## 8. What was tried and rejected

Each of these was judged on the same grouped cross-validation as the shipped model.
None of them cleared the bar, and recording that is part of the answer: it is evidence
that the shipped model sits on a plateau rather than at the first thing that worked.

| idea | result |
|---|---|
| per-feed 90- and 365-day half-lives (8 extra features) | log loss +0.001 — no gain |
| splitting T1 updates by `record_action` (correction vs new/status) | +0.000 — accuracy rose to 0.540 but AUC fell; noise |
| non-linear terms on the direction index (squared, absolute deviation) | +0.002 — worse |
| a "T1 title changed state" flag | −0.002 — inside the noise, not worth the extra column |
| observed tag × direction interaction | +0.002 — worse; the tag effect really is additive |
| self-training on the ~11,700 unlabelled cases | worse at **every** confidence threshold (0.55/0.65/0.75) and every pseudo-label weight (0.02/0.05/0.15), from +0.002 to +0.165 log loss. Pseudo-labels sharpen the model's existing beliefs and wreck the calibration that the scoring rewards. |
| forcing the garbled row onto a candidate via the 4-per-candidate capacity constraint | coverage rose to ~99% but address, licence and external signal all got weaker |
| recovering the garbled address row by the gap it leaves in the person's timeline | not possible: consecutive address records chain end-to-start in 3 of 8,322 cases, and gaps run from −136 to +413 days. The records do not tile time, so a missing row has no locatable slot. |

The 4-per-candidate structure does show up in the resolution output as predicted: 9,369 of
12,000 candidates resolve to exactly three address rows, which is the 3-good-plus-1-garbled
pattern of section 1 seen from the other side.

---

## 9. Reproducibility

`predict.py` is deterministic. Run from a clean directory with the data at an arbitrary
path (`OOS_DATA_DIR`), it reproduces `case_predictions.csv` byte for byte, and
`validate.py` passes all eleven checks. Verified 2026-08-17.

---

## 10. Limitations

- **300 labelled candidates.** Every metric above carries roughly ±0.03 of sampling
  noise, and differences between the top few model configurations are inside it. The
  feature set and half-life were chosen on the flat part of their curves rather than at
  the CV optimum, to avoid selecting on noise.
- **One in four evidence rows is unattributable** in the address, licence and external
  feeds. Nothing in the package recovers those rows; the title feed only escapes because
  `vehicle_ref` exists. If the real system has a person identifier on these feeds, the
  ceiling here moves up substantially.
- **The signal itself is noisy.** Even with perfect resolution, a candidate has ~14
  state-bearing records split between two states, so the latent direction can only be
  estimated to a limited precision. Macro AUC 0.72 may be near the achievable ceiling for
  this data; I could not find an aggregation that beat a recency-weighted share.
- **About 350 vehicles carry two candidates' histories** (343 have eight title rows, six
  have nine). They are split by name cluster, but where both clusters are weak the split
  is uncertain.
- **The 3-good-plus-1-garbled noise model is inferred**, not documented. It is verified
  two independent ways (vehicle grouping, DOB grouping) but a third corruption mode that
  neither view exposes would not have been caught.
