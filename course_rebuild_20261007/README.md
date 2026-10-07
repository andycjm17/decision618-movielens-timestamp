# Decision 618 - MovieLens course rebuild

This directory contains the instructor benchmark, a freshly recomputed 400-fit cross-validation experiment, and a timestamp extension. It was built on 2026-10-07 from the currently published Canvas materials for course 4169. The original D3 files under `output/Decision618_D3_20261006` are separate historical artifacts.

Start with `MovieLens_course_rebuild.ipynb` (executed, English with Chinese explanations), `Course_split_guide_CN.md`, or `Methods_and_findings_EN.md`. The Chinese PDF is a research companion, not the final double-spaced D3 submission.

## Inputs and provenance

- `sources/ratings.dat`, `movies.dat`, `users.dat`: instructor's MovieLens 1M files. Ratings and users match the existing official GroupLens source hashes. Canvas modifies 50 movie titles, mostly for ASCII normalization; IDs, genres and release years match.
- `sources/ColFil_dist.ipynb`: exact teaching code.
- `sources/cv_all_1m.pkl`: teacher's complete CV cache; only whitelisted Pandas/NumPy constructors are allowed by the reader. The cache is a source artifact, not a new fit.
- `sources/course_manifest.json`: course file IDs, ordinary Canvas URLs, timestamps and SHA-256 hashes. It contains no signed download URLs or tokens.
- `sources/README_MovieLens.txt`: original GroupLens documentation and use terms.

The full local folder contains all reviewed course sources. The compact ZIP excludes the 158 MB teacher cache, other lecture decks, the large fresh CV prediction matrix, and machine-specific R binaries. It includes raw MovieLens data, audited predictions/metrics, model artifacts, source hashes, code, charts and the executed notebook. Its saved-output review notebook is runnable; a full source-cache audit requires the omitted Canvas cache. Course materials remain local to the team.

## Python environment

The recorded environment is Python 3.9.6 with pinned packages in `requirements.txt`. To create a separate environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

For the present workspace, the working interpreter is:

```bash
../.venv-movielens/bin/python
```

The notebook locates its project folder without a hard-coded username. Its default run recomputes split checks, metric checks and editable plots from the completed experiment artifacts. It does not silently repeat 400 expensive fits. The optional `RETRAIN` cell invokes the full sequential pipeline.

## R environment

Actual fits used R 4.0.2 and softImpute 1.4-1 from the official CRAN Mac R4.0 binary archive, installed only in `runtime/r-library`. That runtime is specific to this Mac and is omitted from the compact ZIP. On another machine, install a compatible R and softImpute yourself; a newer package may change numerical results. The R script prepends the local library if present and otherwise uses the installed library.

```r
install.packages('softImpute')
```

Original instructor R initialization/version were not recorded. New final fits use R seed 144; fresh CV uses `144 + 100 * fold + rank`. The precise version and parameters are stored in each fit CSV. See the official [CRAN package](https://CRAN.R-project.org/package=softImpute) and [author's vignette](https://hastie.su.domains/swData/softImpute/vignette.html).

## Refit every model

Run from this directory after the R package and source cache are available:

```bash
.venv/bin/python code/run_all.py --fresh-cv
```

Without `--fresh-cv`, the pipeline uses the completed new CV scores and refits the selected final model, classroom benchmark, temporal grid and ensemble. The complete experiment performed 400 CV fits and 11 additional CF fits. Times are saved in `results/fresh_cv_fit_log.csv` and each `*_fit.csv` rather than asserted as a hardware guarantee.

To retrieve an omitted source, use the authorized custom Canvas MCP. Resolve the file ID in `sources/course_manifest.json`, ask Canvas for the file's current download URL, and download into `sources` using its recorded filename. Do not save access tokens or signed URLs in this deliverable. The authenticated acquisition script is a local operational helper under the workspace `tmp`, not a published artifact.

## Read or re-execute the notebook

```bash
.venv/bin/python code/execute_notebook.py
```

This creates an ephemeral kernel specification pointing at the calling Python and executes all cells top-to-bottom. It exports `MovieLens_course_rebuild.html` as a static preview. No permanent global Jupyter kernel is installed.

## Audit and interpretation

`results/analysis_audit.json` records independent checks of row alignment, train/test disjointness, exact legacy split, strict temporal boundaries, all reported metrics, feature behavior on ties, portable ensemble predictions, and completion of all 400 fresh fits. `results/independent_r_audit.csv` checks full saved CF predictions against direct UDV dot products and checks cold-start fallback values.

Warm means both user and movie occur in the fitted history. Cold ratings are retained. The future-test ensemble uses first-90% CF refitting and forward out-of-fold calibration; test-period labels and interactions do not update the fitted history. Calendar features are UTC, not inferred user-local time. Rating timestamps do not establish movie-viewing order. The paired bootstrap resamples users and reports event-weighted RMSE differences, conditional on this experimental setup.

The two protocols answer different questions. Random ratings measure missing-rating completion. Future ratings measure prediction in a later period with changing user cohorts. Do not rank protocols by their raw RMSE or use the result as evidence of higher engagement or revenue. Any NextItNet comparison must use a separate common ranking target and split.

For the present workspace, replace `.venv/bin/python` with `../.venv-movielens/bin/python` to use the already validated environment. Regenerating the Chinese PDF also requires ReportLab/Pillow and the macOS Arial Unicode font used by `code/build_report.py`; the saved PDF and analytical pipeline do not require that font.
