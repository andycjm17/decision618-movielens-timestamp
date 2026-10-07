# Recorded NextItNet comparison evidence

These files support slide 7 of the October 7 deck. The run began on September 12, 2026 UTC. No October 7 retraining is claimed.

`results/test_metrics.csv` and `strict_timestamp_metrics.csv` contain the reported aggregates. `test_per_user.csv` and `sequence_split.csv.gz` allow the presentation preparation script to recompute both cohorts. The selection and protocol JSON files document the choices recorded before test scoring. `summary.json` records the model adaptation and selection. `audit.json` preserves the original audit result.

`code/experiment.py` is the original experiment source for inspection. This evidence folder does not contain the saved TensorFlow checkpoint or a complete training environment. The presentation preparation script needs only NumPy and pandas to re-score saved ranks. Do not treat the evidence folder as a ready-to-run training installation.

The model reuses the official Recommenders NextItNet causal residual encoder and adapts its output to item-only full softmax. SVD optimizes rating prediction. Per-user last-item holdout can include other users' later ratings. Rating timestamps do not establish movie viewing order.

MovieLens data and derived records retain their source terms, documented in `course_rebuild_20261007/sources/README_MovieLens.txt` relative to the repository root. Third-party components retain their original authorship.
