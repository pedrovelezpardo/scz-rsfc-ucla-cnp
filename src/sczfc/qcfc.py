"""Motion quality-control benchmarks (Ciric et al., 2017; Parkes et al., 2018)."""
from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


def partial_corr(X: np.ndarray, v: np.ndarray, covars: np.ndarray | None):
    """Correlation of each column of X with v, controlling for covars (OLS residualization)."""
    n = len(v)
    Z = np.ones((n, 1)) if covars is None else np.column_stack([np.ones(n), covars])
    rx = X - Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    rv = v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    r = (rx.T @ rv) / (np.sqrt((rx ** 2).sum(axis=0)) * np.sqrt((rv ** 2).sum()))
    dof = n - Z.shape[1] - 1
    t = r * np.sqrt(dof / np.clip(1 - r ** 2, 1e-12, None))
    p = 2 * stats.t.sf(np.abs(t), dof)
    return r, p


def qcfc_summary(X: np.ndarray, mean_fd: np.ndarray, covars: np.ndarray | None, distance: np.ndarray) -> dict:
    """QC-FC: partial correlation between mean FD and each edge across participants.

    Returns the percentage of edges significantly related to motion (uncorrected and FDR), the median absolute
    QC-FC correlation and its distance dependence (Spearman correlation with inter-node Euclidean distance)."""
    r, p = partial_corr(X, mean_fd, covars)
    q = multipletests(p, method="fdr_bh")[1]
    rho, p_rho = stats.spearmanr(r, distance)
    return {"pct_edges_p05": float(100 * np.mean(p < 0.05)),
            "pct_edges_fdr05": float(100 * np.mean(q < 0.05)),
            "median_abs_qcfc": float(np.median(np.abs(r))),
            "distance_dependence_rho": float(rho), "distance_dependence_p": float(p_rho)}, r
