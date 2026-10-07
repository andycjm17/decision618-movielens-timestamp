# MovieLens: course replication and a timestamp extension

We reviewed all currently published technical decks and notebooks for Decision 618. Lecture 2 recommends holding out recent observations when the task has time dependence, and using a random split when it does not. Lecture 7 uses a 99/1 random split for MovieLens rating completion. We reproduced that benchmark and added a separate test of predicting ratings in a future period.

The dataset contains 1,000,209 ratings from 6,040 users for 3,706 rated movies. Each observation is a user-movie rating; the target is the explicit 1-5 score. This differs from predicting the next item in a sequence.

Fresh CV selects rank 9 (RMSE 0.918274), with rank 8 close behind (0.918956). The instructor cache selects rank 8. We keep rank 8 for the faithful classroom comparison and save the fresh-selected refit separately. This difference is relevant to reproducibility: the package version and R initialization are different, and the preferred rank is not a universal constant.

## Replication

We reproduced the instructor's exact NumPy split and CV fold assignments: 990,206 training ratings and 10,003 test ratings, with seed 144. We independently recomputed all 20 rank scores from the supplied out-of-fold predictions. The cache selects rank 8. We also ran 400 fresh R softImpute fits, testing ranks 1-20 across 20 folds; their results are saved separately.

The rank-8 classroom refit achieved RMSE 0.919767, MAE 0.710589 and OSR² 0.330010. The saved instructor notebook reports 0.919, 0.709 and 0.332. Our R seed is explicit, while the instructor notebook only seeds NumPy; its R package version is also unrecorded. We used R 4.0.2 and softImpute 1.4-1, so the procedure is reproduced without claiming identical numerical initialization. We also rebuilt the genre-factor heatmap. Its normalized values describe latent loadings, not predicted star ratings.

## What we added

The instructor's ensemble slides already include date and time. Our contribution is a reproducible ensemble, a controlled timestamp ablation, and an evaluation that respects the order of time.

The static Ridge ensemble uses the CF prediction, movie genres and release year, and user demographic categories. The timestamp version adds UTC calendar features, elapsed days, and counts and gaps from strictly earlier ratings in the fitted history. Equal-second entries are excluded from history features. Numerical drift features are capped at the meta-training range. The final test period does not update the fitted history online.

The temporal split uses timestamp-only 80th and 90th percentiles and keeps ties together. It contains 800,164 training, 100,024 validation and 100,021 test ratings. Validation selects CF rank/lambda from a fixed six-model grid and Ridge alpha from {1, 100, 10,000}. Before the final test, CF is refitted on the first 900,188 ratings. The ensemble uses expanding-time out-of-fold CF predictions, rather than in-sample fitted predictions. The held-out test ratings are never used to fit features, choose parameters or refit a model.

## Results

| Model | Random test RMSE | Future test RMSE |
|---|---:|---:|
| Training mean | 1.1237 | 1.0893 |
| CF | 0.9198 | 1.2246 |
| CF + metadata | 0.8990 | 1.0303 |
| CF + metadata + timestamp | **0.8940** | **1.0001** |

Compared with the static ensemble on the same holdout, timestamps reduced RMSE by 0.0050 in the random test and 0.0302 in the future test. Paired user-level bootstrap 95% intervals were [0.0031, 0.0069] and [0.0111, 0.0504] rating points. These intervals are conditional on this split and model-selection procedure. They do not establish a causal effect or a gain in engagement.

The temporal CF result is weaker than the training-mean baseline. We retained it after independently checking the saved predictions against latent-factor dot products and checking all row IDs and cold-start fallbacks. Validation is only 23.84% warm, while the final test is 95.70% warm; this cohort difference limits the transfer of validation-selected parameters. Random and temporal RMSE evaluate different populations and uses, so their difference cannot be assigned entirely to chronology.

Timestamp EDA also matters for the earlier sequential recommendation idea. Of adjacent within-user rating entries, 53.21% share the same second and 88.72% are within a minute. These are rating-entry times, not verified viewing times. A next-item model needs to acknowledge this ambiguity and use its own chronological ranking evaluation.

The project fits Media & Entertainment and connects directly to the course topics of out-of-sample evaluation, cross-validation, feature engineering, regularization and collaborative filtering. Rating accuracy can support candidate ranking, but Top-N quality and business outcomes require additional evaluation. The new notebook and report are research companions; they are not a replacement submission containing unconfirmed team contributions.

**Sources:** Decision 618 Lecture 2, PDF pp.65 and 87-89; Lecture 4, PDF pp.67-70; Lecture 6; Lecture 7, PDF pp.45, 47, 52 and 55-59; ColFil_dist.ipynb cells 5-15 and 17-23 (zero-based); GroupLens MovieLens 1M README. See `sources/course_manifest.json` for exact Canvas file IDs and hashes. Page references use PDF positions, which can differ from slide footers.
