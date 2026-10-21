"""Nested repeated cross-validation with within-fold confound control, metrics and statistical inference."""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from statsmodels.stats.proportion import proportion_confint


class ConfoundResidualizer(BaseEstimator, TransformerMixin):
    """Regress the last `n_confounds` columns (confounds) out of the other columns (features).

    The regression is estimated on the training data only and then applied to the test data (Snoek et al., 2019;
    Chyzhyk et al., 2022). Confound columns are dropped from the output."""

    def __init__(self, n_confounds: int = 0):
        self.n_confounds = n_confounds

    def _split(self, X):
        X = np.asarray(X, float)
        k = self.n_confounds
        return (X, None) if k == 0 else (X[:, :-k], X[:, -k:])

    def fit(self, X, y=None):
        F, C = self._split(X)
        if C is not None:
            D = np.column_stack([np.ones(len(C)), C])
            self.beta_ = np.linalg.lstsq(D, F, rcond=None)[0]
        return self

    def transform(self, X):
        F, C = self._split(X)
        if C is None:
            return F
        return F - np.column_stack([np.ones(len(C)), C]) @ self.beta_


LR_GRID = {"clf__C": np.logspace(-5, 1, 13)}
SVM_GRID = {"clf__C": np.logspace(-6, 0, 13)}
RF_GRID = {"clf__max_features": ["sqrt", 0.05], "clf__min_samples_leaf": [1, 5]}
CONF_GRID = {"clf__C": np.logspace(-3, 2, 6)}


def make_model(name: str, n_conf: int = 0, seed: int = 0):
    """Return (pipeline, hyper-parameter grid). L2 logistic regression is the primary classifier."""
    resid = ("resid", ConfoundResidualizer(n_conf))
    if name == "LR":   # L2-penalised (l1_ratio=0 is the scikit-learn >= 1.8 way of requesting a pure L2 penalty)
        clf = LogisticRegression(C=1.0, l1_ratio=0.0, solver="liblinear", class_weight="balanced", max_iter=10000)
        return Pipeline([resid, ("scale", StandardScaler()), ("clf", clf)]), LR_GRID
    if name == "SVM":
        clf = LinearSVC(C=1e-3, class_weight="balanced", max_iter=100000)
        return Pipeline([resid, ("scale", StandardScaler()), ("clf", clf)]), SVM_GRID
    if name == "RF":
        clf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=1)
        return Pipeline([resid, ("clf", clf)]), RF_GRID
    if name == "LR_confounds":   # confound-only baseline: covariates are the predictors
        clf = LogisticRegression(C=1.0, l1_ratio=0.0, solver="liblinear", class_weight="balanced", max_iter=10000)
        return Pipeline([("scale", StandardScaler()), ("clf", clf)]), CONF_GRID
    raise ValueError(name)


def threshold_for(model_name: str) -> float:
    return 0.0 if model_name == "SVM" else 0.5


def decision_scores(est, X):
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    return est.decision_function(X)


def _fit_predict(model_name, n_conf, X, y, tr, te, seed, inner_folds):
    est, grid = make_model(model_name, n_conf, seed)
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    gs = GridSearchCV(est, grid, cv=inner, scoring="roc_auc", n_jobs=1, refit=True)
    gs.fit(X[tr], y[tr])
    return decision_scores(gs.best_estimator_, X[te]), gs.best_params_


def nested_cv(X, y, model_name="LR", n_conf=0, n_repeats=100, outer_folds=5, inner_folds=5, seed=0, n_jobs=-1):
    """Repeated stratified k-fold (outer) with a stratified k-fold grid search (inner) in every training set.

    Returns a (n_repeats × n_subjects) array of out-of-fold decision scores and the selected hyper-parameters."""
    y = np.asarray(y).astype(int)
    rskf = RepeatedStratifiedKFold(n_splits=outer_folds, n_repeats=n_repeats, random_state=seed)
    splits = list(rskf.split(X, y))
    res = Parallel(n_jobs=n_jobs)(
        delayed(_fit_predict)(model_name, n_conf, X, y, tr, te, seed + k, inner_folds)
        for k, (tr, te) in enumerate(splits))
    scores = np.full((n_repeats, len(y)), np.nan)
    params = []
    for k, ((_, te), (s, bp)) in enumerate(zip(splits, res)):
        scores[k // outer_folds, te] = s
        params.append(bp)
    return scores, params


# ---------------------------------------------------------------------------------------------- metrics

def binary_metrics(y, pred) -> dict:
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    tp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 1)).sum())
    tn, fp = int(((pred == 0) & (y == 0)).sum()), int(((pred == 1) & (y == 0)).sum())
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    out = {"tp": tp, "fn": fn, "tn": tn, "fp": fp, "sensitivity": sens, "specificity": spec,
           "balanced_accuracy": (sens + spec) / 2, "accuracy": (tp + tn) / len(y),
           "ppv": tp / (tp + fp) if tp + fp else np.nan, "npv": tn / (tn + fn) if tn + fn else np.nan}
    for name, k, n in (("sensitivity", tp, tp + fn), ("specificity", tn, tn + fp)):
        lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
        out[f"{name}_ci_low"], out[f"{name}_ci_high"] = float(lo), float(hi)
    return out


def _midrank(x):
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def _delong(y, score_rows):
    """Fast DeLong (Sun & Xu, 2014): AUCs and their covariance for one or more score vectors."""
    y = np.asarray(y).astype(int)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    m, n = len(pos), len(neg)
    S = np.atleast_2d(np.asarray(score_rows, float))
    k = S.shape[0]
    aucs, v01, v10 = np.zeros(k), np.zeros((k, m)), np.zeros((k, n))
    for r in range(k):
        allv = np.r_[S[r, pos], S[r, neg]]
        tz = _midrank(allv)
        tx, ty = _midrank(S[r, pos]), _midrank(S[r, neg])
        aucs[r] = tz[:m].sum() / m / n - (m + 1.0) / 2.0 / n
        v01[r] = (tz[:m] - tx) / n
        v10[r] = 1.0 - (tz[m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def delong_ci(y, s, alpha=0.05):
    aucs, cov = _delong(y, s)
    se = float(np.sqrt(cov[0, 0]))
    z = stats.norm.ppf(1 - alpha / 2)
    return float(aucs[0]), max(0.0, aucs[0] - z * se), min(1.0, aucs[0] + z * se)


def delong_test(y, s1, s2):
    """Two-sided DeLong test for two correlated AUCs computed on the same participants."""
    aucs, cov = _delong(y, np.vstack([s1, s2]))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    zval = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return float(aucs[0] - aucs[1]), float(zval), float(2 * stats.norm.sf(abs(zval)))


def bootstrap_auc_ci(y, s, n_boot=2000, seed=0, alpha=0.05):
    """Percentile CI of the AUC from a participant-level bootstrap stratified by group."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        vals[b] = roc_auc_score(y[idx], s[idx])
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def summarize_scores(scores, y, threshold, n_boot=2000, seed=0) -> dict:
    """Summary of repeated out-of-fold scores.

    * AUC of the repetition-averaged score with a bootstrap 95% CI (participant-level uncertainty) and DeLong CI;
    * distribution of the AUC across repetitions (split-to-split variability);
    * sensitivity / specificity (Wilson 95% CIs) at the pre-specified threshold, on the averaged score."""
    y = np.asarray(y).astype(int)
    rep_auc = np.array([roc_auc_score(y, s) for s in scores])
    rep_bacc = np.array([binary_metrics(y, s >= threshold)["balanced_accuracy"] for s in scores])
    s_bar = np.nanmean(scores, axis=0)
    auc = roc_auc_score(y, s_bar)
    lo, hi = bootstrap_auc_ci(y, s_bar, n_boot=n_boot, seed=seed)
    _, dl_lo, dl_hi = delong_ci(y, s_bar)
    out = {"n": int(len(y)), "n_sz": int(y.sum()), "n_hc": int((1 - y).sum()), "auc": float(auc),
           "auc_ci_low": lo, "auc_ci_high": hi, "auc_delong_low": dl_lo, "auc_delong_high": dl_hi,
           "auc_rep_mean": float(rep_auc.mean()), "auc_rep_p2_5": float(np.quantile(rep_auc, 0.025)),
           "auc_rep_p97_5": float(np.quantile(rep_auc, 0.975)), "bacc_rep_mean": float(rep_bacc.mean()),
           "bacc_rep_p2_5": float(np.quantile(rep_bacc, 0.025)), "bacc_rep_p97_5": float(np.quantile(rep_bacc, 0.975))}
    out.update(binary_metrics(y, s_bar >= threshold))
    return out


def permutation_test(X, y, model_name="LR", n_conf=0, n_perm=1000, seed=0, n_jobs=-1, outer_folds=5,
                     inner_folds=5):
    """Label-permutation test of the complete nested pipeline (Ojala & Garriga, 2010).

    The observed statistic is the out-of-fold AUC of one repetition of nested CV; each permutation re-runs the
    full procedure (confound regression, scaling, inner grid search) with permuted labels.
    p = (1 + #{null >= observed}) / (1 + n_perm)."""
    y = np.asarray(y).astype(int)

    def one(b):
        yp = y if b < 0 else np.random.default_rng(seed + 10_000 + b).permutation(y)
        skf = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
        s = np.zeros(len(y))
        for tr, te in skf.split(X, yp):
            s[te], _ = _fit_predict(model_name, n_conf, X, yp, tr, te, seed, inner_folds)
        return roc_auc_score(yp, s)

    observed = one(-1)
    null = np.array(Parallel(n_jobs=n_jobs)(delayed(one)(b) for b in range(n_perm)))
    p = (1 + np.sum(null >= observed)) / (1 + n_perm)
    return float(observed), null, float(p)


def corrected_resampled_ttest(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected repeated k-fold t-test on paired per-repetition scores."""
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    var = np.var(d, ddof=1) * (1 / k + n_test / n_train)
    t = d.mean() / np.sqrt(var) if var > 0 else 0.0
    return float(d.mean()), float(t), float(2 * stats.t.sf(abs(t), k - 1))
