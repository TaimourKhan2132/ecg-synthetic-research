# =============================================================================
# dashboard.py — Streamlit Dashboard for ECG Synthetic Research
# Usage: streamlit run src/viz/dashboard.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from pathlib import Path
from PIL import Image
import json

BASE_DIR    = Path(r"C:\Users\taimo\OneDrive\Documents 1\work\ecg-synthetic-research")
RESULTS_DIR = BASE_DIR / "outputs" / "results"
MODELS_DIR  = BASE_DIR / "outputs" / "models"
CLASSES     = ["NORM", "MI", "AFIB", "TACHY"]

EXP_LABELS = {
    "A": "Baseline (PTB-XL only)",
    "B": "PTB-XL + Imagen",
    "C": "PTB-XL + Imagen + NeuroKit2",
}

st.set_page_config(
    page_title="ECG Synthetic Research",
    layout="wide",
    page_icon="🫀",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CSS
# =============================================================================

st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #2196F3;
        margin: 4px 0;
    }
    .highlight { color: #2196F3; font-weight: bold; }
    .good     { color: #4CAF50; font-weight: bold; }
    .warn     { color: #FF9800; font-weight: bold; }
    .section-header {
        font-size: 1.1em;
        font-weight: bold;
        color: #333;
        border-bottom: 2px solid #2196F3;
        padding-bottom: 4px;
        margin: 12px 0 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def discover_runs():
    runs = {}
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        meta = run_dir / "metrics_summary.csv"
        if not meta.exists():
            continue
        df = pd.read_csv(meta)
        if df.empty:
            continue
        exp_key = run_dir.name.split("_")[1]  # A, B, or C
        runs[run_dir.name] = {
            "path":    run_dir,
            "exp_key": exp_key,
            "meta":    df.iloc[0].to_dict(),
        }
    return runs


@st.cache_data
def load_csv(path):
    p = Path(path)
    return pd.read_csv(p, index_col=0) if p.exists() else None


@st.cache_data
def load_image(path):
    p = Path(path)
    return Image.open(p) if p.exists() else None


def get_run_label(run_name):
    exp_key = run_name.split("_")[1]
    return EXP_LABELS.get(exp_key, run_name)

# =============================================================================
# COMPARISON PLOTS
# =============================================================================

def comparison_bar_chart(runs_dict, selected_runs, metric_cols, title, ylabel):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    x       = np.arange(len(metric_cols))
    width   = 0.8 / len(selected_runs)

    for i, run_name in enumerate(selected_runs):
        meta   = runs_dict[run_name]["meta"]
        vals   = [meta.get(col, 0) for col in metric_cols]
        offset = (i - len(selected_runs) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width * 0.9,
                        label=get_run_label(run_name),
                        color=colors[i % len(colors)], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [c.replace("f1_", "").replace("roc_auc_", "").replace("pr_auc_", "")
         for c in metric_cols],
        fontsize=10
    )
    ax.set_ylim(0.6, 1.05)
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def scalar_comparison_table(runs_dict, selected_runs):
    scalar_metrics = [
        "test_acc", "macro_f1", "weighted_f1",
        "macro_roc_auc", "macro_pr_auc",
        "cohen_kappa", "mcc",
    ]
    rows = []
    for run_name in selected_runs:
        meta = runs_dict[run_name]["meta"]
        row  = {"Experiment": get_run_label(run_name)}
        for m in scalar_metrics:
            row[m] = round(meta.get(m, 0), 4)
        rows.append(row)
    return pd.DataFrame(rows)

# =============================================================================
# MAIN LAYOUT
# =============================================================================

def main():
    st.title("🫀 ECG Synthetic Augmentation Research")
    st.caption(
        "Comparing EfficientNet-B0 trained on PTB-XL real data "
        "vs Imagen augmentation vs combined augmentation"
    )

    runs_dict = discover_runs()

    if not runs_dict:
        st.error("No completed runs found in outputs/results/")
        st.info("Run: `python src/training/train.py --experiment A`")
        return

    # ── Sidebar ──────────────────────────────────────────────────────────
    st.sidebar.header("🧪 Experiment Selection")
    all_runs     = list(runs_dict.keys())
    selected_runs = st.sidebar.multiselect(
        "Select runs to compare",
        all_runs,
        default=all_runs,
        format_func=get_run_label,
    )

    if not selected_runs:
        st.warning("Select at least one run from the sidebar.")
        return

    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ Run Info")
    for run_name in selected_runs:
        meta = runs_dict[run_name]["meta"]
        st.sidebar.markdown(f"**{get_run_label(run_name)}**")
        st.sidebar.markdown(
            f"- Macro F1: `{meta.get('macro_f1', 0):.4f}`\n"
            f"- ROC-AUC: `{meta.get('macro_roc_auc', 0):.4f}`\n"
            f"- Kappa: `{meta.get('cohen_kappa', 0):.4f}`"
        )
        st.sidebar.markdown("---")

    # ══════════════════════════════════════════════════════════════════════
    # TAB LAYOUT
    # ══════════════════════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Summary",
        "📈 Training Curves",
        "🔢 Confusion Matrices",
        "📉 ROC & PR Curves",
        "🎯 Per-Class Analysis",
        "🔬 GRAD-CAM",
    ])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 — SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("## Experiment Comparison")

        # Key metric cards
        cols = st.columns(len(selected_runs))
        for col, run_name in zip(cols, selected_runs):
            meta = runs_dict[run_name]["meta"]
            with col:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<b>{get_run_label(run_name)}</b><br>"
                    f"Macro F1: <span class='good'>"
                    f"{meta.get('macro_f1', 0):.4f}</span><br>"
                    f"ROC-AUC: {meta.get('macro_roc_auc', 0):.4f}<br>"
                    f"Kappa: {meta.get('cohen_kappa', 0):.4f}<br>"
                    f"MCC: {meta.get('mcc', 0):.4f}"
                    f"</div>",
                    unsafe_allow_html=True
                )

        st.markdown("---")

        # Full scalar table
        st.markdown("### All Scalar Metrics")
        comp_df = scalar_comparison_table(runs_dict, selected_runs)
        st.dataframe(
            comp_df.style
            .highlight_max(
                subset=[c for c in comp_df.columns if c != "Experiment"],
                color="#c8f7c5"
            )
            .format({c: "{:.4f}"
                     for c in comp_df.columns if c != "Experiment"}),
            use_container_width=True,
            height=120,
        )

        st.markdown("---")
        st.markdown("### Macro Metrics Comparison")
        fig = comparison_bar_chart(
            runs_dict, selected_runs,
            ["macro_f1", "macro_roc_auc", "macro_pr_auc",
             "cohen_kappa", "mcc"],
            "Macro-Level Metrics Across Experiments",
            "Score"
        )
        st.pyplot(fig, use_container_width=True)

        st.markdown("### Per-Class F1 Comparison")
        fig = comparison_bar_chart(
            runs_dict, selected_runs,
            [f"f1_{cls}" for cls in CLASSES],
            "Per-Class F1 Score Across Experiments",
            "F1 Score"
        )
        st.pyplot(fig, use_container_width=True)

        st.markdown("### Per-Class ROC-AUC Comparison")
        fig = comparison_bar_chart(
            runs_dict, selected_runs,
            [f"roc_auc_{cls}" for cls in CLASSES],
            "Per-Class ROC-AUC Across Experiments",
            "ROC-AUC"
        )
        st.pyplot(fig, use_container_width=True)

        st.markdown("### Per-Class PR-AUC Comparison")
        fig = comparison_bar_chart(
            runs_dict, selected_runs,
            [f"pr_auc_{cls}" for cls in CLASSES],
            "Per-Class PR-AUC Across Experiments",
            "PR-AUC"
        )
        st.pyplot(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 — TRAINING CURVES
    # ══════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("## Training Curves")

        for run_name in selected_runs:
            run_dir = runs_dict[run_name]["path"]
            st.markdown(f"### {get_run_label(run_name)}")

            img = load_image(run_dir / "training_curves.png")
            if img:
                st.image(img, use_container_width=True)
            else:
                st.info("Training curves image not found.")

            hist_path = run_dir / "training_history.csv"
            if hist_path.exists():
                with st.expander("Raw training history"):
                    st.dataframe(
                        pd.read_csv(hist_path).style.format("{:.5f}"),
                        use_container_width=True
                    )
            st.markdown("---")

        # Overlaid Val F1 across all experiments
        if len(selected_runs) > 1:
            st.markdown("### Val F1 Overlay — All Experiments")
            colors = ["#2196F3", "#4CAF50", "#FF9800"]
            fig, ax = plt.subplots(figsize=(10, 5))
            for run_name, color in zip(selected_runs, colors):
                run_dir   = runs_dict[run_name]["path"]
                hist_path = run_dir / "training_history.csv"
                if hist_path.exists():
                    hist = pd.read_csv(hist_path)
                    ax.plot(hist["epoch"], hist["val_f1"],
                            label=get_run_label(run_name),
                            color=color, lw=2)
                    best_ep = hist.loc[hist["val_f1"].idxmax(), "epoch"]
                    best_f1 = hist["val_f1"].max()
                    ax.axhline(y=best_f1, color=color,
                               linestyle=":", alpha=0.4, lw=1)
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel("Val Macro F1", fontsize=11)
            ax.set_title("Validation F1 Comparison Across Experiments",
                         fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3 — CONFUSION MATRICES
    # ══════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("## Confusion Matrices")

        cols = st.columns(len(selected_runs))
        for col, run_name in zip(cols, selected_runs):
            with col:
                st.markdown(f"**{get_run_label(run_name)}**")
                run_dir = runs_dict[run_name]["path"]

                img = load_image(run_dir / "confusion_matrix.png")
                if img:
                    st.image(img, use_container_width=True)

                cm_df = load_csv(run_dir / "confusion_matrix.csv")
                if cm_df is not None:
                    with st.expander("Raw counts"):
                        st.dataframe(cm_df, use_container_width=True)

        # Error analysis
        if len(selected_runs) >= 2:
            st.markdown("---")
            st.markdown("### Confusion Delta — Exp B vs Exp A")
            st.caption(
                "Positive = more correct in B. "
                "Negative = more errors in B."
            )
            run_a = next(
                (r for r in selected_runs if "ptbxl_only" in r), None
            )
            run_b = next(
                (r for r in selected_runs if "imagen" in r
                 and "neurokit" not in r), None
            )
            if run_a and run_b:
                cm_a = load_csv(
                    runs_dict[run_a]["path"] / "confusion_matrix.csv"
                )
                cm_b = load_csv(
                    runs_dict[run_b]["path"] / "confusion_matrix.csv"
                )
                if cm_a is not None and cm_b is not None:
                    delta = cm_b.astype(int) - cm_a.astype(int)
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(
                        delta, annot=True, fmt="d",
                        cmap="RdYlGn", center=0,
                        xticklabels=CLASSES,
                        yticklabels=CLASSES,
                        ax=ax, linewidths=0.5,
                        cbar_kws={"label": "Count change"}
                    )
                    ax.set_title("Confusion Matrix Delta (B − A)",
                                 fontweight="bold")
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("True")
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 4 — ROC & PR CURVES
    # ══════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("## ROC & PR Curves")

        for run_name in selected_runs:
            run_dir = runs_dict[run_name]["path"]
            st.markdown(f"### {get_run_label(run_name)}")
            c1, c2 = st.columns(2)
            with c1:
                img = load_image(run_dir / "roc_curves.png")
                if img:
                    st.image(img, use_container_width=True,
                             caption="ROC Curves")
            with c2:
                img = load_image(run_dir / "pr_curves.png")
                if img:
                    st.image(img, use_container_width=True,
                             caption="Precision-Recall Curves")
            st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 5 — PER-CLASS ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown("## Per-Class Analysis")

        for run_name in selected_runs:
            run_dir = runs_dict[run_name]["path"]
            st.markdown(f"### {get_run_label(run_name)}")
            c1, c2 = st.columns(2)
            with c1:
                img = load_image(run_dir / "per_class_metrics.png")
                if img:
                    st.image(img, use_container_width=True,
                             caption="Per-Class Precision / Recall / F1")
            with c2:
                img = load_image(
                    run_dir / "probability_distributions.png"
                )
                if img:
                    st.image(img, use_container_width=True,
                             caption="Class Confidence Distributions")

            # Full report table
            report_df = load_csv(
                run_dir / "classification_report.csv"
            )
            if report_df is not None:
                with st.expander("Full classification report"):
                    st.dataframe(
                        report_df.style.format("{:.4f}"),
                        use_container_width=True
                    )
            st.markdown("---")

        # TACHY deep dive
        st.markdown("### 🔴 TACHY Class Deep Dive")
        st.caption(
            "TACHY is the most data-scarce class (426 real records). "
            "Augmentation effect is most visible here."
        )
        tachy_rows = []
        for run_name in selected_runs:
            meta = runs_dict[run_name]["meta"]
            tachy_rows.append({
                "Experiment":  get_run_label(run_name),
                "F1":          round(meta.get("f1_TACHY", 0), 4),
                "ROC-AUC":     round(meta.get("roc_auc_TACHY", 0), 4),
                "PR-AUC":      round(meta.get("pr_auc_TACHY", 0), 4),
                "Precision":   round(meta.get("precision_TACHY", 0), 4),
                "Recall":      round(meta.get("recall_TACHY", 0), 4),
            })
        tachy_df = pd.DataFrame(tachy_rows)
        st.dataframe(
            tachy_df.style.highlight_max(
                subset=["F1", "ROC-AUC", "PR-AUC"],
                color="#c8f7c5"
            ),
            use_container_width=True
        )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 6 — GRAD-CAM
    # ══════════════════════════════════════════════════════════════════════
    with tab6:
        st.markdown("## GRAD-CAM Explainability")
        st.caption(
            "GRAD-CAM shows which regions of each ECG image the model "
            "attends to. Hot regions = high attention. "
            "Waveform-focused maps = clinically valid learning."
        )

        for run_name in selected_runs:
            run_dir  = runs_dict[run_name]["path"]
            gc_dir   = run_dir / "gradcam"

            st.markdown(f"### {get_run_label(run_name)}")

            if not gc_dir.exists():
                st.info("GRAD-CAM not yet generated for this run.")
                continue

            overview = gc_dir / "gradcam_overview.png"
            if overview.exists():
                st.image(
                    load_image(overview),
                    caption="GRAD-CAM Overview — All Classes",
                    use_container_width=True
                )

            # Per-class expandable
            st.markdown("**Per-class detail:**")
            cls_cols = st.columns(4)
            for col, cls in zip(cls_cols, CLASSES):
                cls_img = gc_dir / f"gradcam_{cls}.png"
                if cls_img.exists():
                    col.image(
                        load_image(cls_img),
                        caption=cls,
                        use_container_width=True
                    )
            st.markdown("---")

        # Cross-experiment GRAD-CAM comparison
        gc_runs = [
            r for r in selected_runs
            if (runs_dict[r]["path"] / "gradcam").exists()
        ]
        if len(gc_runs) > 1:
            st.markdown("### Cross-Experiment GRAD-CAM Comparison")
            sel_cls = st.selectbox(
                "Select class to compare across experiments",
                CLASSES
            )
            cmp_cols = st.columns(len(gc_runs))
            for col, run_name in zip(cmp_cols, gc_runs):
                gc_img = (runs_dict[run_name]["path"] / "gradcam"
                          / f"gradcam_{sel_cls}.png")
                if gc_img.exists():
                    col.image(
                        load_image(gc_img),
                        caption=get_run_label(run_name),
                        use_container_width=True
                    )


if __name__ == "__main__":
    main()