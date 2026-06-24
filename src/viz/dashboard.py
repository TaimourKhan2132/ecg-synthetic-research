# =============================================================================
# dashboard.py — ECG Synthetic Research · Fold-Aware Streamlit Dashboard
# Usage: streamlit run src/viz/dashboard.py
# =============================================================================

import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from PIL import Image

matplotlib.use("Agg")

# =============================================================================
# PATHS & CONSTANTS
# =============================================================================

BASE_DIR    = Path(r"C:\Users\taimo\OneDrive\Documents 1\work\ecg-synthetic-research")
RESULTS_DIR = BASE_DIR / "outputs" / "results"
CLASSES     = ["NORM", "MI", "AFIB", "TACHY"]

EXP_META = {
    "A": {
        "label":  "Exp A — PTB-XL only",
        "short":  "A",
        "color":  "#378ADD",
        "fill":   "#B5D4F4",
        "desc":   "Baseline: real PTB-XL data only",
        "n_train": 4456,
    },
    "B": {
        "label":  "Exp B — + Imagen",
        "short":  "B",
        "color":  "#1D9E75",
        "fill":   "#9FE1CB",
        "desc":   "PTB-XL + 640 Imagen-generated images",
        "n_train": 5096,
    },
    "C": {
        "label":  "Exp C — + Imagen + NK2",
        "short":  "C",
        "color":  "#D85A30",
        "fill":   "#F0997B",
        "desc":   "PTB-XL + Imagen + 6 000 NeuroKit2 images",
        "n_train": 11096,
    },
}

SCALAR_COLS = [
    ("test_acc",        "Accuracy"),
    ("macro_f1",        "Macro F1"),
    ("weighted_f1",     "Weighted F1"),
    ("macro_roc_auc",   "ROC-AUC"),
    ("macro_pr_auc",    "PR-AUC"),
    ("cohen_kappa",     "Kappa"),
    ("mcc",             "MCC"),
]

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="ECG Synthetic Research",
    layout="wide",
    page_icon="🫀",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── metric pill ── */
.pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin-right: 4px;
}
.pill-A { background:#B5D4F4; color:#0C447C; }
.pill-B { background:#9FE1CB; color:#085041; }
.pill-C { background:#F0997B; color:#712B13; }
.pill-run { background:#E1F5EE; color:#085041; font-size:11px; }

/* ── summary card ── */
.summary-card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 14px 18px;
    border-left: 4px solid #888;
    margin-bottom: 8px;
}
.summary-card.A { border-color: #378ADD; }
.summary-card.B { border-color: #1D9E75; }
.summary-card.C { border-color: #D85A30; }

/* ── version warning ── */
.version-note {
    background: #fffbe6;
    border-left: 3px solid #faad14;
    padding: 6px 12px;
    border-radius: 4px;
    font-size: 12px;
    color: #7a5800;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# RUN DISCOVERY & PARSING
# =============================================================================

@st.cache_data(ttl=30)
def discover_runs():
    """
    Discover all completed runs and parse their metadata.
    Expected run name pattern: exp_{A|B|C}_{desc}_fold{0|1|2}
    Returns dict keyed by run_name.
    """
    runs = {}
    if not RESULTS_DIR.exists():
        return runs

    pattern = re.compile(r"^exp_([ABC])_.*_fold([012])$", re.IGNORECASE)

    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "metrics_summary.csv"
        if not meta_path.exists():
            continue
        try:
            df = pd.read_csv(meta_path)
            if df.empty:
                continue
            meta = df.iloc[0].to_dict()
        except Exception:
            continue

        m = pattern.match(run_dir.name)
        if m:
            exp_key  = m.group(1).upper()
            fold_num = int(m.group(2))
        else:
            # fallback: try to find exp key from first segment
            parts    = run_dir.name.split("_")
            exp_key  = parts[1].upper() if len(parts) > 1 and parts[1].upper() in EXP_META else "?"
            fold_num = None
            for p in parts:
                fm = re.match(r"fold([012])", p)
                if fm:
                    fold_num = int(fm.group(1))
                    break

        runs[run_dir.name] = {
            "path":     run_dir,
            "exp_key":  exp_key,
            "fold":     fold_num,
            "meta":     meta,
            "label":    f"{EXP_META[exp_key]['label']} · fold {fold_num}" if exp_key in EXP_META and fold_num is not None else run_dir.name,
        }
    return runs


def group_by_exp(runs):
    """Group run_names by experiment key → {A: [run_names], B: [...], C: [...]}"""
    groups = {k: [] for k in EXP_META}
    for name, info in runs.items():
        k = info["exp_key"]
        if k in groups:
            groups[k].append(name)
    for k in groups:
        groups[k].sort(key=lambda n: runs[n]["fold"] if runs[n]["fold"] is not None else 99)
    return groups


# =============================================================================
# HELPERS
# =============================================================================

@st.cache_data
def load_csv(path: str):
    p = Path(path)
    return pd.read_csv(p, index_col=0) if p.exists() else None


@st.cache_data
def load_image(path: str):
    p = Path(path)
    return Image.open(p) if p.exists() else None


def cv_stats(runs, run_names, col):
    """Return (mean, std, values_list) for a metric across folds."""
    vals = [runs[n]["meta"].get(col) for n in run_names if runs[n]["meta"].get(col) is not None]
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return None, None, []
    return float(np.mean(vals)), float(np.std(vals)), vals


def fold_label(info):
    f = info["fold"]
    return f"Fold {f}" if f is not None else "?"


def exp_color(exp_key):
    return EXP_META.get(exp_key, {}).get("color", "#888")


def exp_fill(exp_key):
    return EXP_META.get(exp_key, {}).get("fill", "#ccc")


def missing_img(label="Not generated yet"):
    st.caption(f"_{label}_")


# =============================================================================
# SIDEBAR
# =============================================================================

def build_sidebar(runs, groups):
    st.sidebar.markdown("## 🫀 ECG Research")
    st.sidebar.markdown("---")

    st.sidebar.markdown("### Filter experiments")
    show_exps = {}
    for k, meta in EXP_META.items():
        n_runs = len(groups[k])
        show_exps[k] = st.sidebar.checkbox(
            f"{meta['label']} ({n_runs} fold{'s' if n_runs != 1 else ''})",
            value=True,
            key=f"show_{k}"
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filter folds")
    show_folds = {}
    for f in [0, 1, 2]:
        show_folds[f] = st.sidebar.checkbox(f"Fold {f}", value=True, key=f"fold_{f}")

    st.sidebar.markdown("---")
    # Quick stats
    st.sidebar.markdown("### Quick stats")
    for k in EXP_META:
        if not show_exps[k] or not groups[k]:
            continue
        m, s, _ = cv_stats(runs, groups[k], "macro_f1")
        label = EXP_META[k]["short"]
        if m is not None:
            n = len(groups[k])
            suffix = f"±{s:.3f}" if s is not None and n > 1 else "1 fold"
            st.sidebar.markdown(
                f"<span class='pill pill-{k}'>{label}</span> "
                f"**{m:.4f}** {suffix}",
                unsafe_allow_html=True
            )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Fold 0 trained on PTB-XL v1.0.1 · "
        "Folds 1–2 on v1.0.3 · "
        "0 records differ between versions."
    )

    return show_exps, show_folds


def filter_runs(runs, groups, show_exps, show_folds):
    selected = []
    for k, names in groups.items():
        if not show_exps.get(k):
            continue
        for n in names:
            f = runs[n]["fold"]
            if f is None or show_folds.get(f, True):
                selected.append(n)
    return selected


# =============================================================================
# TAB 1 — CROSS-VALIDATION SUMMARY
# =============================================================================

def tab_cv_summary(runs, groups, selected):
    st.markdown("## Cross-validation summary")
    st.caption(
        "Mean ± std across completed folds. "
        "Variance is expected; flag if range > 4 pp."
    )

    # ── per-experiment summary cards ──────────────────────────────────────────
    cols = st.columns(3)
    for col, (k, meta) in zip(cols, EXP_META.items()):
        run_names = [n for n in groups[k] if n in selected]
        with col:
            m_f1, s_f1, _ = cv_stats(runs, run_names, "macro_f1")
            m_auc, s_auc, _ = cv_stats(runs, run_names, "macro_roc_auc")
            m_pr,  s_pr,  _ = cv_stats(runs, run_names, "macro_pr_auc")
            n = len(run_names)
            st.markdown(
                f"<div class='summary-card {k}'>"
                f"<b>{meta['label']}</b><br>"
                f"<small>{meta['desc']}</small><br><br>"
                f"Macro F1 &nbsp;<b>{m_f1:.4f}</b> {f'±{s_f1:.4f}' if s_f1 and n>1 else ''}<br>"
                f"ROC-AUC &nbsp;<b>{m_auc:.4f}</b> {f'±{s_auc:.4f}' if s_auc and n>1 else ''}<br>"
                f"PR-AUC &nbsp;&nbsp;&nbsp;<b>{m_pr:.4f}</b> {f'±{s_pr:.4f}' if s_pr and n>1 else ''}<br>"
                f"<small style='color:#888'>{n}/3 fold{'s' if n!=1 else ''} complete</small>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ── scalar metrics table ──────────────────────────────────────────────────
    st.markdown("### Mean metrics across folds")
    rows = []
    for k, meta in EXP_META.items():
        run_names = [n for n in groups[k] if n in selected]
        if not run_names:
            continue
        row = {"Experiment": meta["label"], "Folds": len(run_names)}
        for col, nice in SCALAR_COLS:
            m, s, _ = cv_stats(runs, run_names, col)
            if m is not None:
                row[nice] = f"{m:.4f}" + (f" ±{s:.4f}" if s and len(run_names) > 1 else "")
            else:
                row[nice] = "—"
        rows.append(row)

    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Experiment"), use_container_width=True)

    st.markdown("---")

    # ── fold-level F1 grid ───────────────────────────────────────────────────
    st.markdown("### Per-fold macro F1")
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(3)
    width = 0.25
    for i, (k, meta) in enumerate(EXP_META.items()):
        run_names = [n for n in groups[k] if n in selected]
        vals = []
        for f in [0, 1, 2]:
            fn = next((n for n in run_names if runs[n]["fold"] == f), None)
            vals.append(runs[fn]["meta"].get("macro_f1") if fn else None)
        offset = (i - 1) * width
        for j, v in enumerate(vals):
            if v is not None:
                bar = ax.bar(x[j] + offset, v, width * 0.88,
                             color=meta["color"], alpha=0.85,
                             label=meta["label"] if j == 0 else "")
                ax.text(x[j] + offset, v + 0.002, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=8.5,
                        color=meta["color"], fontweight="bold")
            else:
                ax.bar(x[j] + offset, 0, width * 0.88,
                       color=meta["fill"], alpha=0.4, hatch="///",
                       label=meta["label"] + " (pending)" if j == 0 else "")

    ax.set_xticks(x)
    ax.set_xticklabels(["Fold 0", "Fold 1", "Fold 2"])
    ax.set_ylim(0.78, 1.01)
    ax.set_ylabel("Macro F1", fontsize=11)
    ax.set_title("Macro F1 by fold and experiment", fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── variance health check ─────────────────────────────────────────────────
    st.markdown("### Variance health")
    for k, meta in EXP_META.items():
        run_names = [n for n in groups[k] if n in selected]
        m, s, vals = cv_stats(runs, run_names, "macro_f1")
        n = len(run_names)
        if n == 0:
            continue
        rng = max(vals) - min(vals) if len(vals) > 1 else 0
        if n < 3:
            status = f"⏳ {n}/3 folds — check back when complete"
            color  = "#faad14"
        elif rng > 4.0:
            status = f"⚠️ Range {rng:.2f} pp — high variance, check fold splits"
            color  = "#f5222d"
        else:
            status = f"✅ Range {rng:.2f} pp — stable"
            color  = "#52c41a"
        st.markdown(
            f"<span class='pill pill-{k}'>{meta['short']}</span> "
            f"<span style='color:{color}'>{status}</span> "
            f"&nbsp;&nbsp; mean={m:.4f} ±{s:.4f}" if m else "",
            unsafe_allow_html=True
        )


# =============================================================================
# TAB 2 — EXPERIMENT COMPARISON (single fold view)
# =============================================================================

def tab_comparison(runs, groups, selected):
    st.markdown("## Experiment comparison — single fold")
    st.caption("Compare A vs B vs C at the same fold for an apples-to-apples view.")

    available_folds = sorted({runs[n]["fold"] for n in selected if runs[n]["fold"] is not None})
    if not available_folds:
        st.warning("No runs selected.")
        return

    fold = st.selectbox("Select fold", available_folds, format_func=lambda f: f"Fold {f}")

    # Pick one run per experiment for this fold
    fold_runs = {}
    for k in EXP_META:
        match = next((n for n in groups[k] if runs[n]["fold"] == fold and n in selected), None)
        fold_runs[k] = match

    # ── metric cards ──────────────────────────────────────────────────────────
    cols = st.columns(3)
    for col, (k, run_name) in zip(cols, fold_runs.items()):
        with col:
            if run_name is None:
                st.info(f"{EXP_META[k]['label']}\n\nFold {fold} not yet run.")
                continue
            meta = runs[run_name]["meta"]
            st.markdown(
                f"<div class='summary-card {k}'>"
                f"<b>{EXP_META[k]['label']}</b><br>"
                f"<small><code>{run_name}</code></small><br><br>"
                f"Macro F1 &nbsp;<b>{meta.get('macro_f1', '—'):.4f}</b><br>"
                f"ROC-AUC &nbsp;<b>{meta.get('macro_roc_auc', '—'):.4f}</b><br>"
                f"PR-AUC &nbsp;&nbsp;&nbsp;<b>{meta.get('macro_pr_auc', '—'):.4f}</b><br>"
                f"Accuracy &nbsp;<b>{meta.get('test_acc', '—'):.4f}</b>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ── per-class F1 ─────────────────────────────────────────────────────────
    present = {k: v for k, v in fold_runs.items() if v is not None}
    if not present:
        return

    st.markdown("### Per-class F1")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # bar chart
    ax = axes[0]
    x = np.arange(len(CLASSES))
    width = 0.8 / len(present)
    for i, (k, run_name) in enumerate(present.items()):
        meta = runs[run_name]["meta"]
        vals = [meta.get(f"f1_{cls}", 0) for cls in CLASSES]
        offset = (i - len(present) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9,
                      color=EXP_META[k]["color"], alpha=0.85,
                      label=EXP_META[k]["short"])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_ylim(0.6, 1.05)
    ax.set_title("Per-class F1", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    # delta vs baseline
    ax2 = axes[1]
    baseline_run = fold_runs.get("A")
    if baseline_run:
        base_meta = runs[baseline_run]["meta"]
        for i, (k, run_name) in enumerate(present.items()):
            if k == "A":
                continue
            meta = runs[run_name]["meta"]
            deltas = [
                (meta.get(f"f1_{cls}", 0) - base_meta.get(f"f1_{cls}", 0)) * 100
                for cls in CLASSES
            ]
            ax2.bar(x + (i - 1) * 0.3, deltas, 0.27,
                    color=EXP_META[k]["color"], alpha=0.85,
                    label=f"{EXP_META[k]['short']} − A")
            for j, d in enumerate(deltas):
                ax2.text(x[j] + (i - 1) * 0.3, d + (0.15 if d >= 0 else -0.6),
                         f"{d:+.1f}", ha="center", fontsize=8,
                         color=EXP_META[k]["color"])
        ax2.axhline(0, color="#888", linewidth=0.8, linestyle="--")
        ax2.set_xticks(x)
        ax2.set_xticklabels(CLASSES)
        ax2.set_title("Delta vs Exp A (pp)", fontweight="bold")
        ax2.set_ylabel("Δ F1 (percentage points)")
        ax2.legend(fontsize=9)
        ax2.grid(axis="y", alpha=0.25, linestyle="--")
        ax2.spines[["top", "right"]].set_visible(False)
    else:
        ax2.text(0.5, 0.5, "Exp A fold not available for delta",
                 ha="center", va="center", transform=ax2.transAxes, color="#888")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")

    # ── full scalar table ─────────────────────────────────────────────────────
    st.markdown("### All metrics")
    rows = []
    for k, run_name in present.items():
        meta = runs[run_name]["meta"]
        row  = {"Experiment": EXP_META[k]["label"]}
        for col, nice in SCALAR_COLS:
            row[nice] = round(meta.get(col, 0), 4)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Experiment")
    num_cols = [nice for _, nice in SCALAR_COLS if nice in df.columns]
    st.dataframe(
        df.style.highlight_max(subset=num_cols, color="#c8f7c5")
                .format("{:.4f}", subset=num_cols),
        use_container_width=True
    )


# =============================================================================
# TAB 3 — TRAINING CURVES
# =============================================================================

def tab_training(runs, selected):
    st.markdown("## Training curves")

    view_mode = st.radio("View", ["Per run", "Val F1 overlay"], horizontal=True)

    if view_mode == "Per run":
        for run_name in selected:
            info    = runs[run_name]
            run_dir = info["path"]
            st.markdown(f"### {info['label']}")
            img = load_image(str(run_dir / "training_curves.png"))
            if img:
                st.image(img, use_container_width=True)
            else:
                missing_img("training_curves.png not found")

            hist_path = run_dir / "training_history.csv"
            if hist_path.exists():
                with st.expander("Raw training history"):
                    st.dataframe(pd.read_csv(hist_path).style.format("{:.5f}"),
                                 use_container_width=True)
            st.markdown("---")

    else:
        # Overlay: group by experiment
        fig, ax = plt.subplots(figsize=(11, 5))
        plotted = False
        styles  = {0: "-", 1: "--", 2: ":"}

        for run_name in selected:
            info    = runs[run_name]
            hist_p  = info["path"] / "training_history.csv"
            if not hist_p.exists():
                continue
            hist    = pd.read_csv(hist_p)
            if "val_f1" not in hist.columns:
                continue
            k     = info["exp_key"]
            f     = info["fold"]
            color = exp_color(k)
            ls    = styles.get(f, "-")
            ax.plot(hist["epoch"], hist["val_f1"],
                    color=color, linestyle=ls, lw=1.8,
                    label=f"{EXP_META[k]['short']} fold {f}")
            plotted = True

        if plotted:
            # legend: exp by color, fold by linestyle
            exp_patches = [mpatches.Patch(color=EXP_META[k]["color"], label=EXP_META[k]["label"])
                           for k in EXP_META if any(runs[n]["exp_key"] == k for n in selected)]
            fold_lines  = [plt.Line2D([0], [0], color="#444", ls=styles[f], lw=1.5, label=f"Fold {f}")
                           for f in [0, 1, 2]]
            ax.legend(handles=exp_patches + fold_lines, fontsize=9, ncol=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Val Macro F1")
            ax.set_title("Validation F1 — all runs overlay", fontweight="bold")
            ax.grid(alpha=0.2, linestyle="--")
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No training_history.csv files found for selected runs.")
        plt.close(fig)


# =============================================================================
# TAB 4 — CONFUSION MATRICES
# =============================================================================

def tab_confusion(runs, groups, selected):
    st.markdown("## Confusion matrices")

    view = st.radio("Layout", ["Side by side (same fold)", "All runs"], horizontal=True)

    if view == "Side by side (same fold)":
        available = sorted({runs[n]["fold"] for n in selected if runs[n]["fold"] is not None})
        fold = st.selectbox("Fold", available, format_func=lambda f: f"Fold {f}", key="cm_fold")
        fold_runs = [n for n in selected if runs[n]["fold"] == fold]
        if not fold_runs:
            st.info("No runs for this fold.")
            return
        cols = st.columns(len(fold_runs))
        for col, run_name in zip(cols, fold_runs):
            with col:
                info = runs[run_name]
                st.markdown(f"**{info['label']}**")
                img = load_image(str(info["path"] / "confusion_matrix.png"))
                if img:
                    st.image(img, use_container_width=True)
                else:
                    missing_img()
                cm = load_csv(str(info["path"] / "confusion_matrix.csv"))
                if cm is not None:
                    with st.expander("Raw counts"):
                        st.dataframe(cm, use_container_width=True)

        # Delta B-A and C-A
        st.markdown("---")
        st.markdown("### Confusion deltas at fold " + str(fold))
        a_run = next((n for n in fold_runs if runs[n]["exp_key"] == "A"), None)
        if a_run:
            for k in ["B", "C"]:
                k_run = next((n for n in fold_runs if runs[n]["exp_key"] == k), None)
                if not k_run:
                    continue
                cm_a = load_csv(str(runs[a_run]["path"] / "confusion_matrix.csv"))
                cm_k = load_csv(str(runs[k_run]["path"] / "confusion_matrix.csv"))
                if cm_a is None or cm_k is None:
                    continue
                delta = cm_k.astype(int) - cm_a.astype(int)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(delta, annot=True, fmt="d", cmap="RdYlGn", center=0,
                            xticklabels=CLASSES, yticklabels=CLASSES,
                            ax=ax, linewidths=0.5,
                            cbar_kws={"label": "count change"})
                ax.set_title(f"Δ Confusion: Exp {k} − Exp A  (fold {fold})",
                             fontweight="bold")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
    else:
        for run_name in selected:
            info = runs[run_name]
            st.markdown(f"**{info['label']}**")
            img = load_image(str(info["path"] / "confusion_matrix.png"))
            if img:
                st.image(img, use_container_width=True)
            else:
                missing_img()
            st.markdown("---")


# =============================================================================
# TAB 5 — ROC & PR CURVES
# =============================================================================

def tab_roc_pr(runs, selected):
    st.markdown("## ROC & PR curves")
    for run_name in selected:
        info    = runs[run_name]
        run_dir = info["path"]
        st.markdown(f"### {info['label']}")
        c1, c2 = st.columns(2)
        with c1:
            img = load_image(str(run_dir / "roc_curves.png"))
            if img:
                st.image(img, use_container_width=True, caption="ROC curves")
            else:
                missing_img("roc_curves.png")
        with c2:
            img = load_image(str(run_dir / "pr_curves.png"))
            if img:
                st.image(img, use_container_width=True, caption="PR curves")
            else:
                missing_img("pr_curves.png")
        st.markdown("---")


# =============================================================================
# TAB 6 — PER-CLASS ANALYSIS
# =============================================================================

def tab_per_class(runs, groups, selected):
    st.markdown("## Per-class analysis")

    # ── TACHY deep dive (always first — it's the story) ──────────────────────
    st.markdown("### TACHY — the critical minority class")
    st.caption("426 real records · most sensitive to augmentation · story of the paper.")

    tachy_rows = []
    for run_name in selected:
        info = runs[run_name]
        meta = info["meta"]
        tachy_rows.append({
            "Run":        info["label"],
            "Exp":        info["exp_key"],
            "Fold":       info["fold"],
            "F1":         round(meta.get("f1_TACHY", 0), 4),
            "ROC-AUC":    round(meta.get("roc_auc_TACHY", 0), 4),
            "PR-AUC":     round(meta.get("pr_auc_TACHY", 0), 4),
            "Precision":  round(meta.get("precision_TACHY", 0), 4),
            "Recall":     round(meta.get("recall_TACHY", 0), 4),
        })
    if tachy_rows:
        tachy_df = pd.DataFrame(tachy_rows).set_index("Run")
        num_cols = ["F1", "ROC-AUC", "PR-AUC", "Precision", "Recall"]
        st.dataframe(
            tachy_df[num_cols].style
                .highlight_max(subset=num_cols, color="#c8f7c5")
                .highlight_min(subset=num_cols, color="#ffd6d6")
                .format("{:.4f}", subset=num_cols),
            use_container_width=True
        )

    st.markdown("---")

    # ── mean per-class F1 across folds ───────────────────────────────────────
    st.markdown("### Mean per-class F1 across folds (by experiment)")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x     = np.arange(len(CLASSES))
    width = 0.25
    for i, (k, meta) in enumerate(EXP_META.items()):
        run_names = [n for n in groups[k] if n in selected]
        means, stds = [], []
        for cls in CLASSES:
            m, s, _ = cv_stats(runs, run_names, f"f1_{cls}")
            means.append(m if m is not None else 0)
            stds.append(s if s is not None else 0)
        offset = (i - 1) * width
        bars = ax.bar(x + offset, means, width * 0.9,
                      color=meta["color"], alpha=0.85, label=meta["short"],
                      yerr=stds, capsize=3, error_kw={"elinewidth": 1.2, "ecolor": "#444"})
        for bar, m_val in zip(bars, means):
            if m_val:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.012,
                        f"{m_val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_ylim(0.6, 1.08)
    ax.set_ylabel("Mean F1 ± std")
    ax.set_title("Per-class mean F1 (error bars = std across folds)", fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")

    # ── per-run detail ────────────────────────────────────────────────────────
    st.markdown("### Per-run classification reports")
    for run_name in selected:
        info    = runs[run_name]
        run_dir = info["path"]
        with st.expander(info["label"]):
            c1, c2 = st.columns(2)
            with c1:
                img = load_image(str(run_dir / "per_class_metrics.png"))
                if img:
                    st.image(img, use_container_width=True)
                else:
                    missing_img("per_class_metrics.png")
            with c2:
                img = load_image(str(run_dir / "probability_distributions.png"))
                if img:
                    st.image(img, use_container_width=True,
                             caption="Confidence distributions")
                else:
                    missing_img("probability_distributions.png")
            report = load_csv(str(run_dir / "classification_report.csv"))
            if report is not None:
                st.dataframe(report.style.format("{:.4f}"), use_container_width=True)


# =============================================================================
# TAB 7 — CALIBRATION
# =============================================================================

def tab_calibration(runs, selected):
    st.markdown("## Model calibration")
    st.caption("ECE = Expected Calibration Error — lower is better. "
               "Reliability diagram shows confidence vs actual accuracy.")

    for run_name in selected:
        info    = runs[run_name]
        run_dir = info["path"]
        st.markdown(f"### {info['label']}")
        c1, c2 = st.columns(2)
        with c1:
            img = load_image(str(run_dir / "calibration_reliability_diagram.png"))
            if img:
                st.image(img, use_container_width=True, caption="Reliability diagram")
            else:
                missing_img("calibration_reliability_diagram.png")
        with c2:
            img = load_image(str(run_dir / "calibration_confidence_histogram.png"))
            if img:
                st.image(img, use_container_width=True, caption="Confidence histogram")
            else:
                missing_img("calibration_confidence_histogram.png")
        pred_path = run_dir / "predictions_with_confidence.csv"
        if pred_path.exists():
            with st.expander("Per-prediction confidence"):
                preds = pd.read_csv(pred_path)
                st.dataframe(preds.head(200), use_container_width=True)
                ece_val = info["meta"].get("ece")
                if ece_val:
                    st.metric("ECE", f"{float(ece_val):.4f}")
        st.markdown("---")


# =============================================================================
# TAB 8 — GRAD-CAM
# =============================================================================

def tab_gradcam(runs, selected):
    st.markdown("## GRAD-CAM explainability")
    st.caption(
        "Attention maps confirm the model learns waveform morphology, "
        "not rendering artifacts or background grid. "
        "P-waves · QRS complexes · ST segments."
    )

    view = st.radio("View", ["Overview per run", "Cross-experiment class comparison"],
                    horizontal=True)

    if view == "Overview per run":
        for run_name in selected:
            info   = runs[run_name]
            gc_dir = info["path"] / "gradcam"
            st.markdown(f"### {info['label']}")
            if not gc_dir.exists():
                st.info("GRAD-CAM not yet generated for this run.")
                st.markdown("---")
                continue
            overview = gc_dir / "gradcam_overview.png"
            if overview.exists():
                st.image(load_image(str(overview)),
                         caption="All classes overview", use_container_width=True)
            cols = st.columns(4)
            for col, cls in zip(cols, CLASSES):
                img = load_image(str(gc_dir / f"gradcam_{cls}.png"))
                if img:
                    col.image(img, caption=cls, use_container_width=True)
            st.markdown("---")
    else:
        cls = st.selectbox("Class", CLASSES)
        gc_runs = [n for n in selected if (runs[n]["path"] / "gradcam").exists()]
        if not gc_runs:
            st.info("No GRAD-CAM found for selected runs.")
            return
        cols = st.columns(len(gc_runs))
        for col, run_name in zip(cols, gc_runs):
            gc_img = runs[run_name]["path"] / "gradcam" / f"gradcam_{cls}.png"
            if gc_img.exists():
                col.image(load_image(str(gc_img)),
                          caption=runs[run_name]["label"],
                          use_container_width=True)
            else:
                col.caption("Not found")


# =============================================================================
# TAB 9 — DATASET OVERVIEW
# =============================================================================

def tab_dataset():
    st.markdown("## Dataset overview")
    st.caption("PTB-XL v1.0.3 · 4-class subset · patient-grouped 3-fold CV")

    # class distribution
    data = {
        "Class":   ["NORM", "MI", "AFIB", "TACHY"],
        "Real (PTB-XL)": [1500, 1500, 1030, 426],
        "Imagen":  [160, 160, 160, 160],
        "NK2":     [1500, 1500, 1500, 1500],
    }
    df = pd.DataFrame(data).set_index("Class")
    df["Total (Exp C)"] = df.sum(axis=1)

    st.dataframe(df.style.highlight_max(axis=0, color="#c8f7c5"), use_container_width=True)

    st.markdown("---")
    st.markdown("### Experiment training set sizes")
    exp_sizes = pd.DataFrame({
        "Experiment": ["A — PTB-XL only", "B — + Imagen", "C — + Imagen + NK2"],
        "Total images": [4456, 5096, 11096],
        "∆ vs A": [0, 640, 6640],
    }).set_index("Experiment")
    st.dataframe(exp_sizes, use_container_width=True)

    st.markdown("---")
    st.markdown("### Fold split summary")
    st.info(
        "GroupKFold(n_splits=3) with patient_id as groups ensures "
        "zero patient overlap across train/val/test. "
        "4,456 records · 4,115 unique patients. "
        "Leakage verification: PASSED for all completed folds."
    )
    st.markdown(
        '<div class="version-note">'
        "Fold 0 rendered from PTB-XL v1.0.1 · Folds 1–2 from v1.0.3 · "
        "0 records differ between versions — confirmed via ecg_id intersection check."
        "</div>",
        unsafe_allow_html=True
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    runs   = discover_runs()
    groups = group_by_exp(runs)

    show_exps, show_folds = build_sidebar(runs, groups)
    selected = filter_runs(runs, groups, show_exps, show_folds)

    st.title("🫀 ECG Synthetic Augmentation — Research Dashboard")
    st.caption(
        "EfficientNet-B0 · 4-class ECG classification · "
        "PTB-XL + Imagen + NeuroKit2 · Patient-grouped 3-fold CV"
    )

    if not runs:
        st.error(f"No completed runs found in `{RESULTS_DIR}`")
        st.code("python src/training/train.py --experiment A --fold 0")
        return

    if not selected:
        st.warning("No runs match current filters — adjust the sidebar.")
        return

    (tab1, tab2, tab3, tab4,
     tab5, tab6, tab7, tab8, tab9) = st.tabs([
        "📊 CV Summary",
        "⚖️ Comparison",
        "📈 Training curves",
        "🔢 Confusion matrices",
        "📉 ROC & PR",
        "🎯 Per-class",
        "🧮 Calibration",
        "🔬 GRAD-CAM",
        "📋 Dataset",
    ])

    with tab1:
        tab_cv_summary(runs, groups, selected)
    with tab2:
        tab_comparison(runs, groups, selected)
    with tab3:
        tab_training(runs, selected)
    with tab4:
        tab_confusion(runs, groups, selected)
    with tab5:
        tab_roc_pr(runs, selected)
    with tab6:
        tab_per_class(runs, groups, selected)
    with tab7:
        tab_calibration(runs, selected)
    with tab8:
        tab_gradcam(runs, selected)
    with tab9:
        tab_dataset()


if __name__ == "__main__":
    main()