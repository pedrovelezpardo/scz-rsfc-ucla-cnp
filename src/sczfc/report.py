"""Tables and publication figures (static, 300 dpi)."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from .interpret import NETWORK_NAMES, NETWORKS  # noqa: E402

# Validated categorical slots (dataviz reference palette) and a blue <-> red diverging map with a gray midpoint
HC_COLOR, SZ_COLOR, THIRD_COLOR = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list(
    "bluegrayred", ["#104281", "#2a78d6", "#86b6ef", "#f0efec", "#f2a3a2", "#e34948", "#9c1f1f"])

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.linewidth": 0.6, "xtick.color": INK2,
    "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 300, "figure.dpi": 150,
})
MM = 1 / 25.4


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(out_dir, f"{name}.tiff"), bbox_inches="tight", facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ tables

def _fmt_mean_sd(x, digits=1):
    x = pd.Series(x).dropna()
    return f"{x.mean():.{digits}f} ± {x.std(ddof=1):.{digits}f}" if len(x) else "—"


def _fmt_n_pct(flag):
    flag = pd.Series(flag).dropna()
    return f"{int(flag.sum())} ({100 * flag.mean():.1f})" if len(flag) else "—"


def _p_fmt(p):
    if p is None or not np.isfinite(p):
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def table1(df: pd.DataFrame) -> pd.DataFrame:
    """Characteristics of an analytic sample. df: one row per participant with y (1 = SZ) and variables."""
    sz, hc = df[df["y"] == 1], df[df["y"] == 0]
    rows = []

    def cont(label, col, digits=1, sz_only=False):
        if col not in df.columns:
            return
        a, b = sz[col].dropna(), hc[col].dropna()
        p = stats.ttest_ind(a, b, equal_var=False).pvalue if (len(a) > 1 and len(b) > 1 and not sz_only) else None
        nn = f" (n = {len(a)})" if len(a) < len(sz) else ""
        rows.append({"Variable": label + nn, "Schizophrenia": _fmt_mean_sd(a, digits),
                     "Healthy controls": "—" if sz_only else _fmt_mean_sd(b, digits),
                     "p": "—" if sz_only else _p_fmt(p), "Test": "" if sz_only else "Welch t"})

    def cat(label, col, sz_only=False, hc_only=False):
        if col not in df.columns:
            return
        if hc_only:
            rows.append({"Variable": label, "Schizophrenia": "—", "Healthy controls": _fmt_n_pct(hc[col]),
                         "p": "—", "Test": ""})
            return
        p = None
        if not sz_only:
            tab = pd.crosstab(df["y"], df[col])
            if tab.shape == (2, 2):
                p = stats.chi2_contingency(tab)[1] if (tab.to_numpy() >= 5).all() else stats.fisher_exact(tab)[1]
        rows.append({"Variable": label, "Schizophrenia": _fmt_n_pct(sz[col]),
                     "Healthy controls": "—" if sz_only else _fmt_n_pct(hc[col]),
                     "p": "—" if sz_only else _p_fmt(p), "Test": "" if sz_only else "χ² / Fisher"})

    rows.append({"Variable": "n", "Schizophrenia": str(len(sz)), "Healthy controls": str(len(hc)), "p": "",
                 "Test": ""})
    cont("Age, years", "age")
    cat("Male sex, n (%)", "male")
    cont("Education, years", "education_years")
    cat("Scanner B (syngo B17), n (%)", "scanner_b")
    cat("T1-weighted ghosting artefact, n (%)", "t1_ghost")
    cont("Mean framewise displacement, mm", "mean_fd", 3)
    cont("Volumes with FD > 0.2 mm, %", "pct_fd_gt_0p2", 1)
    cont("Volumes retained after censoring", "n_keep_primary", 1)
    cont("WAIS-IV Vocabulary, raw score", "wais_vocabulary_raw")
    cont("WAIS-IV Matrix Reasoning, raw score", "wais_matrix_raw")
    cont("CVLT-II trials 1–5, T score", "cvlt_trials1to5_t")
    cat("Schizoaffective disorder (SCID), n (%)", "dx_schizoaffective", sz_only=True)
    cont("BPRS-24 total", "bprs_total", sz_only=True)
    cont("SAPS, sum of global ratings", "saps_global_sum", sz_only=True)
    cont("SANS, sum of global ratings", "sans_global_sum", sz_only=True)
    cont("HAMD-17", "hamd17", sz_only=True)
    cat("Current antipsychotic, n (%)", "antipsychotic_any", sz_only=True)
    cat("  Second-generation, n (%)", "antipsychotic_sga", sz_only=True)
    cat("  First-generation, n (%)", "antipsychotic_fga", sz_only=True)
    cat("Current antidepressant, n (%)", "antidepressant", sz_only=True)
    cat("Current mood stabilizer, n (%)", "mood_stabilizer", sz_only=True)
    cat("Current benzodiazepine, n (%)", "benzodiazepine", sz_only=True)
    cat("No medication record, n (%)", "no_medication_entries", sz_only=True)
    cat("Any lifetime DSM-IV diagnosis (SCID), n (%)", "scid_any_dx", hc_only=True)
    cat("  Substance use disorder, n (%)", "scid_substance_dx", hc_only=True)
    cat("  Depressive disorder, n (%)", "scid_mood_dx", hc_only=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ figures

def flow_diagram(counts: dict, out_dir: str, name="figure1_flow"):
    """Participant flow (counts: dict of box texts)."""
    fig, ax = plt.subplots(figsize=(160 * MM, 140 * MM))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, text, weight="normal", fc="white", size=7.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=INK2,
                                    lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK, weight=weight,
                linespacing=1.3)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.7, color=INK2))

    mx, mw, h = 3, 50, 9
    levels = {"all": 89, "groups": 72, "rest": 55, "deriv": 38, "final": 21}
    keys = list(levels)
    for k, key in enumerate(keys):
        y0 = levels[key]
        box(mx, y0, mw, h, counts[key], weight="bold" if key == "final" else "normal",
            fc="#f7f7f5" if key == "final" else "white")
        if k:
            arrow(mx + mw / 2, levels[keys[k - 1]], mx + mw / 2, y0 + h)
    sx, sw = 60, 38
    side = {"excl_groups": (levels["all"] + levels["groups"] + h) / 2,
            "excl_rest": (levels["groups"] + levels["rest"] + h) / 2,
            "excl_deriv": (levels["rest"] + levels["deriv"] + h) / 2,
            "excl_motion": (levels["deriv"] + levels["final"] + h) / 2}
    for key, yc in side.items():
        if key in counts:
            box(sx, yc - 4, sw, 8, counts[key], fc="#fbfbfa", size=6.8)
            ax.plot([mx + mw / 2, sx - 0.6], [yc, yc], color=INK2, lw=0.7)
            arrow(sx - 2.5, yc, sx - 0.4, yc)
    if "subsamples" in counts:   # drawn without an arrow: not all of these are subsets of the primary sample
        box(mx, 1, 95, 12, counts["subsamples"], fc="#fbfbfa", size=6.6)
    if "footnote" in counts:
        ax.text(mx, -2.5, counts["footnote"], fontsize=6.2, color=INK2, va="top")
    save(fig, out_dir, name)


def performance_figure(y, s_primary, rep_scores, s_conf, null, observed_perm, p_perm, forest: pd.DataFrame,
                       out_dir, name="figure2_performance"):
    fig = plt.figure(figsize=(175 * MM, 165 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=0.38, wspace=0.32)

    # (A) ROC of the repetition-averaged score with the 2.5–97.5% envelope across repetitions
    ax = fig.add_subplot(gs[0, 0])
    grid = np.linspace(0, 1, 201)
    tprs = []
    for s in rep_scores:
        f, t, _ = roc_curve(y, s)
        tprs.append(np.interp(grid, f, t))
    tprs = np.array(tprs)
    ax.fill_between(grid, np.quantile(tprs, 0.025, axis=0), np.quantile(tprs, 0.975, axis=0), color=SZ_COLOR,
                    alpha=0.18, lw=0)
    f, t, _ = roc_curve(y, s_primary)
    ax.plot(f, t, color=SZ_COLOR, lw=2, label="Connectivity model")
    if s_conf is not None:
        f, t, _ = roc_curve(y, s_conf)
        ax.plot(f, t, color=INK2, lw=1.2, ls="--", label="Confounds only")
    ax.plot([0, 1], [0, 1], color=GRID, lw=1)
    ax.set_xlabel("1 − specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title("A", loc="left", weight="bold")

    # (B) permutation null
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(null, bins=30, color="#b7d3f6", edgecolor="white", linewidth=0.5)
    ax.axvline(observed_perm, color=SZ_COLOR, lw=2)
    ax.text(0.02, 0.98, f"Observed AUC = {observed_perm:.3f}\np = {p_perm:.3f} ({len(null):,} permutations)",
            transform=ax.transAxes, color=INK, fontsize=7, va="top", ha="left", zorder=5,
            bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="none", alpha=0.9))
    ax.set_ylim(0, ax.get_ylim()[1] * 1.22)   # headroom for the annotation
    ax.set_xlabel("AUC under permuted labels")
    ax.set_ylabel("Permutations")
    ax.set_title("B", loc="left", weight="bold")

    # (C) forest plot of AUC (95% CI) across analyses
    ax = fig.add_subplot(gs[1, :])
    fr = forest.iloc[::-1].reset_index(drop=True)
    yy = np.arange(len(fr))
    colors = [SZ_COLOR if r == "primary" else (INK2 if r == "baseline" else HC_COLOR) for r in fr["role"]]
    for k, row in fr.iterrows():
        ax.plot([row["auc_ci_low"], row["auc_ci_high"]], [k, k], color=colors[k], lw=1.6, solid_capstyle="round")
        ax.plot(row["auc"], k, "o", ms=5, color=colors[k], mec="white", mew=0.8)
        ax.text(1.005, k, f"{row['auc']:.2f} ({row['auc_ci_low']:.2f}–{row['auc_ci_high']:.2f})", va="center",
                fontsize=6.4, color=INK2, transform=ax.get_yaxis_transform())
    ax.axvline(0.5, color=GRID, lw=1)
    ax.set_yticks(yy)
    ax.set_yticklabels(fr["label"], fontsize=6.8)
    ax.set_xlabel("AUC (95% CI)")
    ax.set_xlim(min(0.3, np.floor(10 * fr["auc_ci_low"].min() - 0.2) / 10), 1.0)
    ax.set_ylim(-0.7, len(fr) - 0.3)
    ax.set_title("C", loc="left", weight="bold")
    save(fig, out_dir, name)


def network_figure(net_matrix, net_sig, stable_edges: pd.DataFrame, atlas: pd.DataFrame, out_dir,
                   name="figure3_networks"):
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from nilearn import plotting

    fig = plt.figure(figsize=(175 * MM, 190 * MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 0.85], hspace=0.42)
    top = gs[0].subgridspec(1, 3, width_ratios=[0.45, 1.5, 0.45])
    ax = fig.add_subplot(top[1])
    M = np.array(net_matrix, float)
    vmax = np.nanmax(np.abs(M)) or 1
    tri = np.tril(np.ones_like(M, bool))
    Mp = np.where(tri, M, np.nan)
    im = ax.imshow(Mp, cmap=DIVERGING, vmin=-vmax, vmax=vmax)
    for i in range(M.shape[0]):
        for j in range(i + 1):
            if net_sig[i, j]:
                ax.text(j, i, "*", ha="center", va="center", fontsize=9, color=INK)
    names = [NETWORK_NAMES[n] for n in NETWORKS]
    ax.set_xticks(range(len(NETWORKS)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(NETWORKS)))
    ax.set_yticklabels(names, fontsize=7)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Mean activation pattern (a.u.)\npositive: higher probability of SZ", fontsize=6.8)
    cb.ax.tick_params(labelsize=6.4)

    ax2 = fig.add_subplot(gs[1])
    coords = atlas[["x_mni", "y_mni", "z_mni"]].to_numpy(float)
    A = np.zeros((100, 100))
    for _, e in stable_edges.iterrows():
        A[int(e["i"]), int(e["j"])] = A[int(e["j"]), int(e["i"])] = e["pattern"]
    used = np.unique(np.r_[stable_edges["i"].astype(int), stable_edges["j"].astype(int)])
    node_colors = [INK2 if k in used else "none" for k in range(100)]
    vmax2 = np.abs(A).max() or 1
    plotting.plot_connectome(A, coords, node_color=node_colors, node_size=[12 if k in used else 0 for k in range(100)],
                             edge_cmap=DIVERGING, edge_vmin=-vmax2, edge_vmax=vmax2, edge_kwargs={"linewidth": 1.5},
                             display_mode="lzr", axes=ax2, colorbar=False, annotate=True)
    pos = ax2.get_position()
    fig.text(0.01, ax.get_position().y1 + 0.01, "A", weight="bold", fontsize=9, va="bottom")
    fig.text(0.01, pos.y1 + 0.01, "B", weight="bold", fontsize=9, va="bottom")
    cax = fig.add_axes([pos.x0 + 0.35 * pos.width, pos.y0 - 0.045, 0.30 * pos.width, 0.014])
    cb2 = fig.colorbar(ScalarMappable(norm=Normalize(-vmax2, vmax2), cmap=DIVERGING), cax=cax,
                       orientation="horizontal")
    cb2.set_label("Activation pattern (a.u.)", fontsize=6.8)
    cb2.ax.tick_params(labelsize=6.4)
    save(fig, out_dir, name)


def qcfc_figure(qc: pd.DataFrame, qcfc_r: dict, distance, out_dir, name="figureS1_qcfc"):
    fig, axes = plt.subplots(1, 3, figsize=(175 * MM, 62 * MM), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    yy = np.arange(len(qc))[::-1]
    colors = [SZ_COLOR if s.startswith("P1") else HC_COLOR for s in qc["strategy"]]
    panels = (("pct_edges_p05", "Edges with QC-FC p < 0.05, %", 5.0),
              ("median_abs_qcfc", "Median |QC-FC|", None),
              ("distance_dependence_rho", "Distance dependence (ρ)", 0.0))
    for k, (ax, (col, title, ref)) in enumerate(zip(axes, panels)):
        ax.barh(yy, qc[col], color=colors, height=0.62)
        if ref is not None:
            ax.axvline(ref, color=INK2, lw=0.8, ls="--" if ref else "-")
        ax.set_xlabel(title, fontsize=7)
        ax.set_title("ABC"[k], loc="left", weight="bold")
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(yy)
    axes[0].set_yticklabels(qc["label"], fontsize=6.4)
    save(fig, out_dir, name)


def group_matrices_figure(mean_hc, mean_sz, tmap, atlas: pd.DataFrame, out_dir, name="figureS3_group_fc"):
    order = np.lexsort((atlas["hemisphere"].to_numpy(), atlas["network"].map(
        {n: k for k, n in enumerate(NETWORKS)}).to_numpy()))
    fig, axes = plt.subplots(1, 3, figsize=(175 * MM, 62 * MM), gridspec_kw={"wspace": 0.18})
    for ax, M, title, cmap, lim in ((axes[0], mean_hc, "Healthy controls (mean Fisher z)", DIVERGING, 0.8),
                                    (axes[1], mean_sz, "Schizophrenia (mean Fisher z)", DIVERGING, 0.8),
                                    (axes[2], tmap, "Group difference (adjusted t)", DIVERGING, 4)):
        im = ax.imshow(M[np.ix_(order, order)], cmap=cmap, vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_title(title, fontsize=7)
        nets = atlas["network"].to_numpy()[order]
        starts = [k for k in range(100) if k == 0 or nets[k] != nets[k - 1]]
        for s0 in starts[1:]:
            ax.axhline(s0 - 0.5, color="white", lw=0.5)
            ax.axvline(s0 - 0.5, color="white", lw=0.5)
        mids = [(a + b) / 2 for a, b in zip(starts, starts[1:] + [100])]
        ax.set_xticks(mids)
        ax.set_yticks(mids)
        ax.set_xticklabels([nets[s] for s in starts], rotation=90, fontsize=5.6)
        ax.set_yticklabels([nets[s] for s in starts] if ax is axes[0] else [], fontsize=5.6)
        ax.tick_params(length=0)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=5.6)
    save(fig, out_dir, name)
