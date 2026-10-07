# MovieLens rating prediction and NextItNet, October 7, 2026

The nine-slide English deck combines the October 7 course replication and timestamp experiment with the completed adapted NextItNet ranking experiment. It includes Chinese explanations and bilingual PowerPoint speaker notes. The speaking script has 557 words and slide budgets total five minutes. The cinema palette and typography follow the October 6 presentation.

| File | Use |
|---|---|
| [Presentation_EN_20261007_v4.pptx](Presentation_EN_20261007_v4.pptx) | Editable PowerPoint with six evidence tables and one chart |
| [Presentation_EN.pdf](Presentation_EN.pdf) | PDF for display |
| [Presentation_EN.html](Presentation_EN.html) | Offline browser deck with notes, timer and text editing |
| [Speaker_Script_EN.md](Speaker_Script_EN.md) | English speaking script |
| [Chinese_Explanation.md](Chinese_Explanation.md) | Chinese explanations, source locations and calculations |
| [evidence/](evidence/) | NextItNet comparison values and aggregate result tables |
| [authoring/](authoring/) | Content, source hashes and builders |
| [fonts/](fonts/) | Presentation fonts and Open Font Licenses |
| [SHA256SUMS.txt](SHA256SUMS.txt) | File integrity hashes |

## Comparisons

Slides 3-6 cover the instructor's softImpute benchmark, fresh cross-validation, and metadata/timestamp ensembles. All models within each RMSE column share a rating holdout. The random and global-time columns cover different populations.

Slide 7 compares Popularity, item-kNN, SVD and adapted NextItNet on the same 6,038 eligible users. NextItNet reaches 19.94% HR@10 and 0.1049 NDCG@10. The page also shows all four methods on the 3,494 eligible users whose test rating arrives strictly after the preceding rating. NextItNet reaches 14.31% HR@10 on that subset, compared with 5.95% for item-kNN. This subset removes equal-second test boundaries but retains a per-user holdout. Other users' later ratings can still enter the fitted data.

Slide 8 explains the targets, holdouts and metrics. RMSE measures star-rating errors. HR@10 measures whether the next rated movie appears in the top ten, and NDCG@10 also rewards a higher position. SVD in the ranking experiment optimizes star ratings. Its result should not be generalized to every matrix-factorization method or ranking objective. A shared global-time ranking comparison remains future work.

The NextItNet results come from the recorded September 12 UTC experiment, not a new October 7 training run. The implementation adapts the official Recommenders causal residual encoder to item-only embeddings and full softmax. This is an adapted NextItNet model, rather than an unmodified official end-to-end benchmark. The source notes identify its validation selection and candidate filtering.

The rating evidence is in `course_rebuild_20261007/results/`. The recorded ranking evidence is in `final_project/nextitnet/results/`. Both paths are relative to the project root. `authoring/source_hashes.json` uses those root-relative paths. The presentation's `evidence/` folder contains the aggregate tables and comparison metadata. Relevant course PDF page positions and notebook cells appear in the speaker notes and Chinese explanation. The course already discusses date and time features.

MovieLens timestamps describe rating entry. Offline results do not establish movie viewing order, engagement gains or ROI. The business pilot on slide 9 is a proposal.

## Presenting

Use the PDF for display. Install Bebas Neue and DM Sans from `fonts/` before editing the PowerPoint. The HTML embeds fonts and works offline: Arrow keys or Space advance slides, F enters fullscreen, N opens notes, T toggles the timer and E enables text editing. Ctrl/Cmd+S exports an edited HTML copy. This version uses a separate edit-storage key to protect it from older saved edits. Rehearse the script to adjust pauses.

## Authoring

`prepare_content.py` reads both recorded experiments and recomputes HR@10/NDCG@10 from saved per-user ranks for the full and later-test cohorts. It regenerates the content, English script, Chinese explanations and comparison evidence. It does not train a model.

`build_pptx.mjs` imports the October 6 PPTX, preserves its masters and framing, and duplicates two source slides for the comparisons. It rebuilds the rating-gap chart from the recorded literal values and generates the HTML. `render_html.mjs` exports the PDF.

The builders require the Codex bundled Node runtime, `@oai/artifact-tool`, Playwright, Python with NumPy/pandas, and presentation finalization helpers. Configure `WORKSPACE_ROOT`, `OUTPUT_DIR`, `BUILD_DIR`, `SOURCE_DECK_DIR`, `PRESENTATIONS_SKILL_DIR`, `RUNTIME_PYTHON` and `RUNTIME_NODE_MODULES` for the host. In this repository, set `SOURCE_DECK_DIR` to `presentations/Decision618_D3_20261006` and choose fresh output and build directories. The finalizer refuses to overwrite an existing final PowerPoint. No native Microsoft PowerPoint application test is claimed.

Uploading these files to GitHub does not submit them to Canvas. Course material and MovieLens data retain their authorship and terms.
