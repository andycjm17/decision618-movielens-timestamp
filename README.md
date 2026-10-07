# Decision 618: MovieLens replication and timestamp analysis

This repository contains the complete October 7, 2026 MovieLens course rebuild: reviewed course sources, raw data, training code, fitted models, prediction results, an executed notebook and a Chinese methods report. Two large prediction caches and the compact review ZIP are stored in the repository's [Release](https://github.com/andycjm17/decision618-movielens-timestamp/releases/tag/v2026.10.07). Their original paths and SHA-256 hashes are recorded in [release_assets.json](release_assets.json).

## Read the results

- [Executed notebook: English with Chinese explanations](course_rebuild_20261007/MovieLens_course_rebuild.ipynb)
- [Chinese PDF: course replication and timestamp analysis](course_rebuild_20261007/output/pdf/MovieLens_Course_Rebuild_CN.pdf)
- [Course train/test split guide in Chinese](course_rebuild_20261007/Course_split_guide_CN.md)
- [Methods and findings in English](course_rebuild_20261007/Methods_and_findings_EN.md)
- [Model metrics](course_rebuild_20261007/results/model_metrics.csv) and [integrity audit](course_rebuild_20261007/results/analysis_audit.json)

We reviewed the nine currently published technical lecture decks and 23 notebooks, reproduced the instructor's exact random split and CV folds, independently rescored the teacher cache, and completed 400 fresh R softImpute CV fits. The new CV selects rank 9; the original cache selects rank 8. The classroom comparison retains rank 8, with the fresh-selected final fit reported separately.

| Model | Random rating-completion RMSE | Future-period rating RMSE |
|---|---:|---:|
| Training mean | 1.1237 | 1.0893 |
| Collaborative filtering | 0.9198 | 1.2246 |
| CF + metadata | 0.8990 | 1.0303 |
| CF + metadata + timestamp | **0.8940** | **1.0001** |

The timestamp increment is measured against the static ensemble on the same holdout. The two evaluation columns contain different populations and answer different questions. Timestamps record rating entry, and 53.21% of adjacent within-user entries share one second. The results therefore do not establish viewing order, next-item ranking performance or a business outcome.

## Presentation

The [October 7 presentation](presentations/Decision618_Deck_20261007/README.md) includes an editable [PowerPoint](presentations/Decision618_Deck_20261007/Presentation_EN_20261007_v4.pptx), [PDF](presentations/Decision618_Deck_20261007/Presentation_EN.pdf), offline HTML, English script, Chinese explanations and fonts. Its nine slides combine the course replication and metadata/timestamp RMSE comparisons with the completed adapted NextItNet ranking experiment. The English script has 557 words and a planned running time of five minutes. This is the only presentation version in the current repository files.

| Ranking model | Full sample HR@10 | Full sample NDCG@10 | Later-test subset HR@10 | Later-test subset NDCG@10 |
|---|---:|---:|---:|---:|
| Popularity | 4.09% | 0.0206 | 3.81% | 0.0191 |
| item-kNN | 7.40% | 0.0389 | 5.95% | 0.0311 |
| SVD | 1.21% | 0.0055 | 1.35% | 0.0066 |
| Adapted NextItNet | **19.94%** | **0.1049** | **14.31%** | **0.0758** |

The recorded NextItNet run began on September 12, 2026 UTC. These results reuse that experiment and do not claim October 7 retraining. Every ranking method uses the same 6,038 eligible users and fitted catalog, filtering previously rated movies. The later-test subset contains 3,494 eligible users whose test rating arrives strictly after the previous rating. It still uses a per-user holdout and can include other users' later ratings in the fitted data. SVD here optimizes star ratings. RMSE and ranking scores measure different targets and should be compared within their own experiments. [Recorded ranking evidence](final_project/nextitnet/README.md) includes saved per-user ranks and exact split membership for re-scoring.

## Restore the complete local project

Clone the repository, then restore the Release assets using an authenticated GitHub CLI:

```bash
git clone https://github.com/andycjm17/decision618-movielens-timestamp.git
cd decision618-movielens-timestamp
python3 scripts/restore_release_assets.py
```

The script verifies each asset's size and SHA-256 before putting it at its original location. It skips an existing file only when its hash matches and refuses to overwrite different content. Use `--check-only` to verify local assets without downloading anything.

## Execute the notebook or refit

The recorded Python environment is 3.9.6; package versions are pinned. Create an environment at the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r course_rebuild_20261007/requirements.txt
.venv/bin/python course_rebuild_20261007/code/execute_notebook.py
```

The default notebook recomputes split and metric checks and renders all eight editable plots from the completed experiment. To refit all models, including the full 400-fit CV:

```bash
.venv/bin/python course_rebuild_20261007/code/run_all.py --fresh-cv
```

Refitting also requires R and softImpute. The recorded run used R 4.0.2 / softImpute 1.4-1. Its isolated macOS R library is included under `course_rebuild_20261007/runtime/r-library`; it is specific to that R/macOS setup. Other platforms need their own R package installation. Package/version or initialization differences can change the numerical results. The dated report records the original run; review its prose when changing the environment or experiments.

## Repository contents

| Path | Contents |
|---|---|
| `course_rebuild_20261007/code/` | Preparation, R fitting, timestamp ensembles, audits, charts and notebook/report builders |
| `course_rebuild_20261007/sources/` | Raw MovieLens tables, 12 course PDFs, 23 teaching notebooks and source manifest |
| `course_rebuild_20261007/source_text/` | Extracted course text used to locate split/CV evidence |
| `course_rebuild_20261007/data/` | Exact random/time splits, mappings and out-of-fold features |
| `course_rebuild_20261007/models/` | Saved CF and Ridge ensemble models |
| `course_rebuild_20261007/results/` | Predictions, metrics, validation choices, training logs and audit records |
| `course_rebuild_20261007/figures/` | Eight analytical charts |
| `course_rebuild_20261007/output/pdf/` | Seven-page Chinese methods companion |
| `presentations/Decision618_Deck_20261007/` | Current v4 English deck with rating and NextItNet comparisons, bilingual notes, scripts, fonts and authoring sources |
| `final_project/nextitnet/` | Recorded ranking metrics, per-user ranks, exact split and original experiment source |
| `release_assets.json` | Download names, original paths, sizes and hashes for large files |
| `UPLOAD_MANIFEST.json` | Inventory of the copied project files and excluded temporary caches |

Course sources and data retain their original authorship and terms. The MovieLens terms and citation are preserved in [README_MovieLens.txt](course_rebuild_20261007/sources/README_MovieLens.txt). This is a coursework repository; no repository-wide open-source license is applied to third-party course material or data.
