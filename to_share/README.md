# Manuscript materials — index

This folder contains everything cited in the report. **Start with
`reports/PAPER_REPORT.md`** (the detailed findings). For LaTeX, use the **`.pdf`**
versions of figures (vector, crisp at any size); `.png` versions are for quick
viewing. Ready-to-paste table code is in **`latex_tables.tex`**.

## Folder map
| Folder / file | What it is |
|---|---|
| `reports/PAPER_REPORT.md` | Detailed findings report (methods, results, discussion, limitations, reproducibility) |
| `latex_tables.tex` | Copy-paste LaTeX for Tables 1–4 (needs `\usepackage{booktabs}`) |
| `figures/` | Confusion matrices + domain-comparison graph (PNG + PDF) |
| `gradcam/` | Per-class Grad-CAMs (correct, high-confidence, with zoomed peak region) |
| `csv/` | Result tables **each with a matching PNG graph** |
| `per_run_results/` | Per-experiment metrics, ROC/PR curves, calibration, classification reports |
| `secondary_test_suites/` | Full metric folders for every model × {REAL, SYNTH, COMBINED} |

## Report section → files
| Report section | Table / figure | File(s) |
|---|---|---|
| §3 Primary results | Table 1 | `latex_tables.tex` (tab:primary); `csv/confidence_intervals_3fold.{csv,png}` |
| §3 Significance | Table 2 | `latex_tables.tex` (tab:effect); `csv/confidence_intervals_3fold.csv` |
| §4 Class redistribution | Confusion matrices | `figures/confusion_{A_baseline,B_imagen,C_imagen_nk2,V4_capped}.{png,pdf}` |
| §5 Calibration | reliability diagrams | `per_run_results/*/calibration_reliability_diagram.png` |
| §7 Secondary real+synthetic | Table 3 + graph | `latex_tables.tex` (tab:secondary); `csv/secondary_summary.{csv,png}`; `csv/secondary_realsynth_test.{csv,png}`; `figures/secondary_realsynth.{png,pdf}` |
| §7 per-model detail | full suites | `secondary_test_suites/<model>__<REAL\|SYNTH\|COMBINED>/` |
| §8 Architecture (B1) | Table 4 | `latex_tables.tex` (tab:b1) |
| §Explainability | Grad-CAM | `gradcam/gradcam_{NORM,MI,AFIB,TACHY}.{png,pdf}`, `gradcam/gradcam_overview.{png,pdf}` |

## Naming key
- `A` = real only · `B` = real + diffusion · `C` = real + diffusion + full simulation ·
  `V4` = real + diffusion + capped simulation (best config)
- `B1_*` = same experiments on EfficientNet-B1
- Test sets: `REAL` (real held-out) · `SYNTH` (held-out NeuroKit2, never trained on) ·
  `COMBINED` (balanced 50:50)
