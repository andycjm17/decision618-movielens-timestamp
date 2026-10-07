# MovieLens rating prediction, October 7, 2026

The current seven-slide English deck presents the October 7 course replication and timestamp experiment. It has bilingual speaker notes, an English script and Chinese explanations, with a planned running time of five minutes. The dark cinema palette and Bebas Neue / DM Sans typography follow the October 6 presentation.

| File | Use |
|---|---|
| [Presentation_EN_20261007_v3.pptx](Presentation_EN_20261007_v3.pptx) | Editable PowerPoint, including four native evidence tables and a native chart |
| [Presentation_EN.pdf](Presentation_EN.pdf) | Display copy with embedded fonts |
| [Presentation_EN.html](Presentation_EN.html) | Offline browser deck with notes, timer and text editing |
| [Speaker_Script_EN.md](Speaker_Script_EN.md) | English speaking script, planned at five minutes |
| [Chinese_Explanation.md](Chinese_Explanation.md) | Chinese explanations, source locations and calculations |
| [fonts/](fonts/) | PowerPoint fonts and Open Font Licenses |
| [authoring/](authoring/) | Content, source hashes and builders |
| [SHA256SUMS.txt](SHA256SUMS.txt) | Integrity hashes for this version |

## Content

1. Media & Entertainment business problem and the 1-5 rating target
2. MovieLens data quality, sparsity and rating-entry timestamps
3. The instructor's rank-8 benchmark and 400 fresh cross-validation fits
4. Random rating completion and a global future-period holdout
5. Training mean, CF, metadata and timestamp ensemble results
6. Paired timestamp increments, prior-history features and cohort limitations
7. Further offline checks and a proposed business pilot

The main evidence is in `../../course_rebuild_20261007/results/model_metrics.csv` and `timestamp_experiment.json`, with course source IDs/hashes in `../../course_rebuild_20261007/sources/course_manifest.json`. Relevant PDF page positions and notebook cells are recorded in the slide notes and Chinese explanation. The instructor already discusses date and time features. This experiment measures their increment and adds a reproducible global time evaluation.

The current target is an explicit star rating. This deck replaces the October 6 NextItNet ranking presentation as the repository's current presentation. RMSE results do not establish Top-N quality, engagement or ROI. The future CF model is worse than the training mean; the slides retain that result and disclose the validation/test cohort difference. Business pilot measures are proposals.

## Presenting and editing

Use the PDF for display. Install both fonts from `fonts/` before editing the PowerPoint. The HTML embeds fonts and works offline: Arrow keys or Space advance slides, F enters fullscreen, N opens notes, T toggles the timer and E enables text editing. Ctrl/Cmd+S exports an edited HTML copy. This version uses a separate edit-storage key so an older deck's saved edits cannot replace its content.

Speaker notes include source locations and both languages. The speaking script contains 596 words; slide budgets total 300 seconds. Rehearse to adjust delivery and pauses. Uploading these files to GitHub does not submit them to Canvas.

## Authoring

`content.json` contains the source-checked numerical evidence and scripts; `source_hashes.json` identifies the underlying experiment files. `prepare_content.py` regenerates the content and scripts from the recorded analysis. `build_pptx.mjs` imports the October 6 PPTX to preserve its slide masters and framing, replaces the experiment content, and rebuilds the adjacent-rating chart from its audited literal values. It also generates the HTML. `render_html.mjs` exports its PDF and checks presentation controls and text fit.

The builders require the Codex bundled Node runtime, `@oai/artifact-tool`, Playwright, Python and presentation finalization helpers. Run from the repository root and configure `WORKSPACE_ROOT`, `OUTPUT_DIR`, `SOURCE_DECK_DIR`, `PRESENTATIONS_SKILL_DIR`, `RUNTIME_PYTHON` and `RUNTIME_NODE_MODULES` for that host. Set `SOURCE_DECK_DIR` to `presentations/Decision618_D3_20261006` and choose a fresh output directory. The artifact finalizer refuses to overwrite an existing final PPTX. Rendering checks do not claim a native Microsoft PowerPoint application test.
