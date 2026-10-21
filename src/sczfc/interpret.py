"""Interpretation of the linear classifier: Haufe activation patterns, bootstrap stability, network summaries and
univariate (edge-wise / network-wise) group comparisons."""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from statsmodels.stats.multitest import multipletests

from .ml import LR_GRID, make_model

NETWORKS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
NETWORK_NAMES = {"Vis": "Visual", "SomMot": "Somatomotor", "DorsAttn": "Dorsal attention",
                 "SalVentAttn": "Salience/ventral attention", "Limbic": "Limbic", "Cont": "Frontoparietal control",
                 "Default": "Default mode"}


def select_c(X, y, n_conf, seed=0):
    est, grid = make_model("LR", n_conf)
    gs = GridSearchCV(est, grid, cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring="roc_auc")
    gs.fit(X, y)
    return float(gs.best_params_["clf__C"])


def fit_lr(X, y, n_conf, C):
    est, _ = make_model("LR", n_conf)
    est.set_params(clf__C=C)
    return est.fit(X, y)


def weights_and_pattern(est, X):
    """Weights w (standardized, confound-residualized features) and Haufe et al. (2014) activation pattern
    a = cov(x, s) / var(s), with s = Xw the classifier's latent score."""
    Z = est[:-1].transform(X)
    w = est.named_steps["clf"].coef_.ravel()
    s = Z @ w
    Zc, sc = Z - Z.mean(axis=0), s - s.mean()
    a = Zc.T @ sc / (len(s) - 1) / np.var(s, ddof=1)
    return w, a


def bootstrap_models(X, y, n_conf, C, n_boot=1000, seed=0, n_jobs=-1):
    y = np.asarray(y).astype(int)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)

    def one(b):
        rng = np.random.default_rng(seed + b)
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        est = fit_lr(X[idx], y[idx], n_conf, C)
        w, a = weights_and_pattern(est, X[idx])
        return w.astype(np.float32), a.astype(np.float32)

    res = Parallel(n_jobs=n_jobs)(delayed(one)(b) for b in range(n_boot))
    return np.stack([r[0] for r in res]), np.stack([r[1] for r in res])


def stability_table(w, a, W, A, edges: pd.DataFrame, top_frac=0.01) -> pd.DataFrame:
    """Edge-level stability across bootstrap refits.

    * freq_top: proportion of bootstrap models in which the edge is among the top `top_frac` by |w|;
    * sign_consistency: proportion of bootstrap weights with the same sign as the full-sample weight;
    * bsr: bootstrap ratio of the activation pattern (full-sample pattern / bootstrap SD);
    * a_ci_low / a_ci_high: 95% percentile interval of the pattern."""
    k = max(1, int(round(top_frac * W.shape[1])))
    top = np.argsort(-np.abs(W), axis=1)[:, :k]
    freq = np.bincount(top.ravel(), minlength=W.shape[1]) / W.shape[0]
    t = edges.copy()
    t["weight"], t["pattern"] = w, a
    t["freq_top"] = freq
    t["sign_consistency"] = (np.sign(W) == np.sign(w)[None, :]).mean(axis=0)
    t["pattern_boot_sd"] = A.std(axis=0, ddof=1)
    t["bsr"] = a / t["pattern_boot_sd"]
    t["pattern_ci_low"], t["pattern_ci_high"] = np.quantile(A, 0.025, axis=0), np.quantile(A, 0.975, axis=0)
    t["rank_abs_weight"] = (-np.abs(w)).argsort().argsort() + 1
    return t


def network_index(edges: pd.DataFrame):
    """Canonical (unordered) network pair for each edge and the list of the 28 pairs."""
    pairs = [(a, b) for i, a in enumerate(NETWORKS) for b in NETWORKS[i:]]
    lookup = {}
    for a, b in pairs:
        lookup[(a, b)] = lookup[(b, a)] = f"{a}–{b}"
    labels = [lookup[(x, y)] for x, y in zip(edges["net_i"], edges["net_j"])]
    return np.array(labels), [lookup[p] for p in pairs]


def network_means(values: np.ndarray, pair_of_edge: np.ndarray, pair_names) -> np.ndarray:
    """Mean of `values` (edges, or samples × edges) within each network pair."""
    values = np.atleast_2d(values)
    return np.stack([values[:, pair_of_edge == p].mean(axis=1) for p in pair_names], axis=1)


def group_glm(X, y, covars):
    """Edge-wise OLS: FC ~ 1 + group + covariates. Returns adjusted difference (SZ - HC), t, p and FDR q."""
    n = len(y)
    D = np.column_stack([np.ones(n), np.asarray(y, float)] + ([covars] if covars is not None else []))
    beta, *_ = np.linalg.lstsq(D, X, rcond=None)
    resid = X - D @ beta
    dof = n - D.shape[1]
    sigma2 = (resid ** 2).sum(axis=0) / dof
    se = np.sqrt(sigma2 * np.linalg.inv(D.T @ D)[1, 1])
    t = beta[1] / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    q = multipletests(p, method="fdr_bh")[1]
    return beta[1], t, p, q


def cohens_d(X, y):
    a, b = X[y == 1], X[y == 0]
    pooled = np.sqrt(((len(a) - 1) * a.var(axis=0, ddof=1) + (len(b) - 1) * b.var(axis=0, ddof=1))
                     / (len(a) + len(b) - 2))
    return (a.mean(axis=0) - b.mean(axis=0)) / pooled
