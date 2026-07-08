# Handoff — manuscript revision status & file map

**Repo:** https://github.com/TaimourKhan2132/ecg-synthetic-research
**Branch:** `main` (everything below is pushed there)

This note lists (1) where every file is, (2) what has already been changed in the
manuscript, (3) what has been changed / added in the repo, and (4) the paper-side
items still left for you to write.

---

## 1. Key file locations

| What | Path |
|---|---|
| **Edited manuscript (the actual paper)** | `C:\Users\taimo\OneDrive\Desktop\ECG_old_paper_overleaf\main.tex` — this is the local Overleaf project, **not** in the GitHub repo. It already has all the new tables/numbers. |
| **Copy-paste LaTeX tables** | `to_share/latex_tables.tex` (in repo) — `tab:primary`, `tab:effect`, `tab:secondary`, `tab:b1`, `tab:delong` |
| **Detailed findings report** | `to_share/reports/PAPER_REPORT.md` (methods, results, significance, discussion points, limitations, reproducibility) |
| **All figures / CSVs / suites** | `to_share/` — see the index in `to_share/README.md` |

> If you are editing the paper, work in the Overleaf `main.tex`. Use
> `to_share/latex_tables.tex` for any table you want to paste fresh, and pull
> figures from `to_share/figures/`, `to_share/gradcam/` (use the `.pdf` versions).

---

## 2. What has already been done in the manuscript (`main.tex`)

All results are now the **3-fold patient-grouped cross-validation** values on the
**real held-out test**, with significance from **repeated-seed CV (n = 9)**.

- **Table II (aggregate metrics)** — final 3-fold values; added an **ECE row**
  (0.105 / 0.098 / 0.088) and a **Macro-F1 (real+synthetic test)** row
  (0.679 / 0.747 / **0.945**, captioned as a robustness check).
- **Table III (per-class F1)** — final values; caption relabeled **"(3-Fold Mean)"**
  and footnotes the n=9 (+3.1) vs 3-fold (+4.1) TACHY gain.
- **Comparison table** — updated "Ours" rows; related-work line now cites 0.892.
- **Composition tables** — NeuroKit2 shown at **500/class** for Exp C; totals recomputed.
- **DeLong test** — added to §IV-A (see item 0 below); table is `tab:delong`.
- **Prose** (abstract, intro, results, discussion, conclusion) — rewritten to the
  final "modest but statistically significant; the *amount* of simulation matters"
  narrative.
- **Model name** — "Imagen 3" → **"Gemini 3 Pro Image"** throughout (the two Peng /
  Saharia related-work citations are left as-is on purpose).
- **Methods & Limitations** — now describe 3-fold + repeated-seed CV and the B1
  replication.
- **Figures** — `fig/confusion_panel_v2.png` and `fig/gradcam_TACHY_C.png` refreshed
  (same figures live in `to_share/figures/` and `to_share/gradcam/`).

---

## 3. What has been added in the repo (supporting evidence)

| Deliverable | Location |
|---|---|
| **DeLong test** on ROC-AUC (reviewer #1) | `outputs/results/delong/delong_auc_test.csv`, `to_share/csv/delong_auc_test.csv`, code `src/utils/delong_test.py`, LaTeX `tab:delong` |
| **n = 9 significance** (paired) | `to_share/reports/PAPER_REPORT.md` §3, `to_share/latex_tables.tex` `tab:effect`, `to_share/csv/confidence_intervals_3fold.{csv,png}` |
| **EfficientNet-B1** results | `to_share/latex_tables.tex` `tab:b1`, `to_share/secondary_test_suites/B1_*` |
| **Secondary real/synthetic/combined** | `to_share/csv/secondary_summary.{csv,png}`, `to_share/secondary_test_suites/` |
| **Confusion matrices / Grad-CAM** | `to_share/figures/`, `to_share/gradcam/` |
| **Reproducibility (all steps)** | `to_share/reports/PAPER_REPORT.md` §12 |

CI provenance fixed: `confidence_intervals_3fold.csv` now uses the capped Exp C
run, so its Macro-F1 reads **0.892** (matches the paper).

---

## 4. Paper-side items still to do (yours)

Numbers/content for each are in `to_share/`; these are LaTeX/prose edits in `main.tex`.

**0. DeLong sentence — DONE, please just eyeball it.** §IV-A now says the combined
model's AUC gain over baseline is significant for NORM/MI/AFIB (p < 0.001) and
directional for TACHY (p = 0.09). Full numbers: `tab:delong`.

**1. Reviewer #2 — justify EfficientNet-B0.** Add one sentence in the Discussion:
B0 chosen for computational efficiency / deployability (fits the 6 GB GPU already
cited). Future-work on deeper nets is partly covered by the B1 table.

**2. Reviewer #3 — restructure Related Work.** §II still uses ~6 bold inline
lead-ins (`\textbf{Diffusion models.}` etc.). Convert them to `\subsection{}`
headings (or merge into ~3 themed subsections). Pure structure edit, no content
change.

**3. Reviewer #5 — expand Discussion on NeuroKit2.** Add a paragraph reframing NK2
as label-consistent, balanced coverage that stabilizes the synthetic distribution
— cite the calibration gain (ECE 0.105 → 0.088) and the synthetic-robustness result,
not raw real-test accuracy.

**4. Add the EfficientNet-B1 table.** The paper claims the effect "replicates on a
second backbone" but shows no numbers. Paste `tab:b1` from `to_share/latex_tables.tex`
(A 0.865 / B 0.883 / C 0.877; B−A p = 0.024).

**5. (Optional) Add the n=9 significance table** to the PDF near Table IV. Paste
`tab:effect`. The evidence is already in the prose; this just makes it a visible
table for reviewer #1.

**6. (Decision needed) "Diffusion" vs "Generative" framing.** The title and framing
still say "Diffusion Models," but the model is Gemini 3 Pro Image (a general-purpose
generative model, not a confirmed diffusion model). Decide whether to soften
"diffusion" → "generative" throughout, including the title.

---

*Everything in items 0–5 has its numbers ready in `to_share/`. Start with
`to_share/reports/PAPER_REPORT.md` and `to_share/latex_tables.tex`.*
