# Building the paper

The submission to Clinical Pharmacology & Therapeutics (CPT) is built from code and result files only.
`paper/manuscript_cpt.md` holds the text, in which numbers computed from the results are `{{key}}` placeholders.
`paper/manuscript_numbers.py` computes these values from the benchmark evaluation (`benchmarks/evaluation/`) and the run
folders (`benchmarks/runs/`) and writes them to `paper/build/numbers.json`. `paper/build_cpt.py` fills them in, numbers
the citations from `paper/references_cpt.py` in order of first citation, adds Tables 1 and 2 and the figure legends,
runs `paper/check_claims.py` (checks of the worded claims and typed values against the run folders and evaluation
outputs, and of the alternative text of the figures), and writes the Word manuscript and the cover letter to
`paper/submission_cpt/`. It refuses a final build when a number is missing, a worded claim no longer holds, or a CPT
limit is exceeded (main text 4,000 words, abstract 250 words, Study Highlights under 250 words, 50 references, 7
figures and tables, 130 characters per table row); `python paper/build_cpt.py --draft` builds anyway, marks missing
numbers and prints the problems.
The figures, the graphical abstract and the supplementary material are written by their own scripts from the same
result files. All commands run from the repository root.

## Paper items

| Paper item | Command | Reads | Writes |
|---|---|---|---|
| Numbers in the text | `python paper/manuscript_numbers.py` | `benchmarks/evaluation/` (evaluation.json, agent_tests.json, effect_evidence.json, scm_baseline.json, backward_baseline.json, reference_fit_ratios.json), `benchmarks/reference_fits/<dataset>/vpc.json`, run folders | `paper/build/numbers.json`; also `table2.json` (Table 2), `recall.json` (Supplementary Material S3, claim checks) and `run_profiles.json` (claim checks) |
| Claim checks | `python paper/check_claims.py` (also run by `build_cpt.py`) | run folders, `benchmarks/evaluation/runs.csv`, agent_tests.json and evaluation.json, `paper/build/numbers.json`, recall.json, run_profiles.json, `paper/figures/alt_text.txt` | printed list of passed and failed checks (no file) |
| Figure 1 | `python paper/figure1_architecture.py` | `paper/tool_groups.py` (tool groups), `pkagent.config.Budget` (limits per run) | `paper/figures/Figure_1.pdf`, `.png`, `.tiff` |
| Figure 2 | `python benchmarks/figures.py` | `benchmarks/evaluation/runs.csv`, reference_fit_ratios.json | `paper/figures/Figure_2.pdf`, `.png`, `.tiff` |
| Figure 3 | `python benchmarks/figures.py` | `benchmarks/evaluation/runs.csv`, evaluation.json, effect_evidence.json | `paper/figures/Figure_3.pdf`, `.png`, `.tiff` |
| Figure 4 | `python benchmarks/figures.py` | `benchmarks/evaluation/runs.csv`, reference fits, run folders | `paper/figures/Figure_4.pdf`, `.png`, `.tiff` |
| Alt text (figures and graphical abstract) | written by hand; checked by `python paper/check_claims.py` (`alt_text_checks`, also run by `build_cpt.py`) | `paper/build/numbers.json`, `pkagent.config.Budget`, `src/pkagent/prompts.py`, `paper/tool_groups.py`, constants of `benchmarks/figures.py` and `paper/graphical_abstract.py`, `benchmarks/evaluation/runs.csv`, evaluation.json | `paper/figures/alt_text.txt`, copied to `paper/submission_cpt/` by `build_cpt.py` |
| Graphical abstract (image and text) | `python paper/graphical_abstract.py` | `paper/build/numbers.json`, `paper/tool_groups.py` | `paper/figures/Graphical_abstract.pdf`, `.png`, `.tiff`, `.txt`; copies of the `.pdf`, `.tiff` and `.txt` in `paper/submission_cpt/` |
| Manuscript (text, figure legends) | `python paper/build_cpt.py` | `paper/manuscript_cpt.md`, `paper/misleading_text.json`, `paper/build/numbers.json`, `paper/references_cpt.py`, `paper/figures/Figure_<n>.pdf` and `.tiff` | `paper/submission_cpt/PKAgent_CPT_manuscript.docx`; `Figure_1` to `Figure_4` `.pdf` and `.tiff` and `alt_text.txt` copied to `paper/submission_cpt/`; `paper/build/manuscript_cpt_filled.md`, `reference_order.json` |
| Table 1 | `python paper/build_cpt.py` | dataset facts in `table1_rows` of `build_cpt.py` (the expert statements are in Supplementary Material S1.5) | in `PKAgent_CPT_manuscript.docx` |
| Table 2 | `python paper/build_cpt.py` | `paper/build/table2.json` | in `PKAgent_CPT_manuscript.docx` |
| Cover letter | `python paper/build_cpt.py` | `paper/cover_letter_cpt.md` | `paper/submission_cpt/PKAgent_CPT_cover_letter.docx` |
| Supplementary Material S1 | `python paper/supplement_cpt.py` | `src/pkagent/prompts.py` (system prompt, task message), `src/pkagent/spec.py` (schema), `benchmarks/datasets.json` (descriptions and statements) | `paper/submission_cpt/Supplementary_Material_S1.docx` |
| Table S1 | `python paper/supplement_cpt.py` | `src/pkagent/tools.py` (definitions), `paper/tool_groups.py` (groups of Figure 1) | in `Supplementary_Material_S1.docx` |
| Supplementary Material S2 | `python paper/supplement_cpt.py` | `src/pkagent/config.py` (settings), `benchmarks/evaluation/runs.csv`, oral_mm_residual_check.json, `benchmarks/reference_fits/`, `paper/build/numbers.json` | `paper/submission_cpt/Supplementary_Material_S2.docx` |
| Table S2 | `python paper/supplement_cpt.py` | `benchmarks/evaluation/reference_table.csv`, `benchmarks/reference_fits/<dataset>/reference_fit.json` | in `Supplementary_Material_S2.docx` |
| Table S3 | `python paper/supplement_cpt.py` | `benchmarks/evaluation/effect_evidence.json` | in `Supplementary_Material_S2.docx` |
| Supplementary Material S3 | `python paper/supplement_cpt.py` | `benchmarks/evaluation/scm_baseline.json`, backward_baseline.json, `paper/build/recall.json`, final reports in the run folders | `paper/submission_cpt/Supplementary_Material_S3.docx` |
| Table S4 | `python paper/supplement_cpt.py` | `benchmarks/evaluation/runs.csv`, agent_tests.json, standard_vpc.json, run folders | in `Supplementary_Material_S3.docx` |
| Table S5 | `python paper/supplement_cpt.py` | `benchmarks/evaluation/agent_tests.json`, tool logs of the runs | in `Supplementary_Material_S3.docx` |
| Benchmark archive (Data Availability) | `python benchmarks/export_archive.py` | `benchmarks/runs/`, `reference_fits/`, `evaluation/` | `dist/PKAgent_benchmark_archive.zip` |

## Order of the commands

Steps marked *fits* run PKPy2 fits or simulations; step 2 calls the language models. The other steps read existing
files. The run folders, reference fits and evaluation outputs (`benchmarks/runs/`, `reference_fits/`, `evaluation/`)
and the build outputs (`paper/build/`, `paper/submission_cpt/`) are not tracked by git; the first three are in the
release archive of step 14.

1. Data: `Rscript benchmarks/export_data.R`, then `python benchmarks/prepare_data.py` (R packages nlme 3.1-168 and
   nlmixr2data 2.0.10).
2. Runs: the `run_benchmark.py` commands in `benchmarks/README.md` (needs `OPENROUTER_API_KEY`). New runs differ from
   those of the paper, because LLM sampling is not seeded; the runs of the paper are in the release archive (step 14).
3. Reference fits (*fits*): `python benchmarks/reference_fits.py pheno remifentanil oral_mm`, then
   `python benchmarks/reference_table.py`. Supplementary Material S2 also reads
   `benchmarks/reference_fits/oral_mm_proportional`, the oral MM reference fit with proportional error from before
   commit fdd311d; the script now fits log-normal error, so that folder comes from the release archive.
4. `python benchmarks/reference_vpc.py` (*fits*): VPCs of the reference fits.
5. `python benchmarks/residual_check.py`: residual error type of the oral MM simulation.
6. `python benchmarks/standard_vpc.py` (*fits*): standard final VPCs; `evaluate.py` reads its output, so it runs
   before `evaluate.py`.
7. `python benchmarks/evaluate.py`.
8. `python benchmarks/agent_tests.py`.
9. Baselines (*fits*): `python benchmarks/effect_evidence.py`, `python benchmarks/scm_baseline.py pheno`,
   `python benchmarks/backward_baseline.py` (reads effect_evidence.json).
10. `python paper/manuscript_numbers.py`.
11. Figures: `python benchmarks/figures.py`, `python paper/figure1_architecture.py`,
    `python paper/graphical_abstract.py` (reads numbers.json from step 10).
12. `python paper/build_cpt.py` (copies Figures 1 to 4 from step 11 and runs the claim checks).
13. `python paper/supplement_cpt.py`.
14. `python benchmarks/export_archive.py`: the release archive of the runs, reference fits and evaluation (no data
    files; local paths removed).

## Figure 1 and the graphical abstract

Figure 1 and the graphical abstract are vector drawings made by `paper/figure1_architecture.py` and
`paper/graphical_abstract.py` with matplotlib primitives at the printed size (7.0 in wide, Arial). The tool groups and
the tool count come from `paper/tool_groups.py`, which asserts that its groups cover exactly the tools of
`src/pkagent/tools.py`; the numbers of the graphical abstract come from `paper/build/numbers.json`. Raster versions
made with FigureLabs were superseded by these drawings in commit 7a1e4a7; `paper/figures/figurelabs/` keeps them as a
record, and `paper/figures/archive/` keeps an earlier vector Figure 1.

## Implementation notes

- Tested with Python 3.13.5, PKPy2 0.2.1, NumPy 2.2.6, SciPy 1.16.1, pandas 2.3.1 and Matplotlib 3.10.5. The paper
  scripts also need python-docx (tested 1.1.2) and Pillow (tested 11.1.0), which `pip install .` does not install.
- The figure scripts set the font to Arial; without Arial, matplotlib substitutes another font and text widths change.
- The CMYK TIFFs are converted by `tiff_cmyk` in `benchmarks/figures.py` with the U.S. SWOP press profile shipped with
  Windows (`C:/Windows/System32/spool/drivers/color/RSWOP.icm`). Where that file does not exist, the function falls
  back without a warning to Pillow's plain RGB-to-CMYK conversion, which uses no black ink, so the TIFFs differ from
  the submitted ones; the PDF and PNG files do not depend on the profile.
- The figure scripts contain no random elements.
- `build_cpt.py` warns while the AI-use disclosure (`AI_DISCLOSURE`) is a placeholder and prints the repository links
  of the Data Availability Statement, which must exist before submission.
- `paper/build.py`, `paper/manuscript.md`, `paper/supplement.py` and `paper/supplement.md` are an earlier draft for
  another format; no script of the CPT submission reads them.
