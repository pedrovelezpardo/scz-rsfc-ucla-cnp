#!/usr/bin/env python
"""Run every analysis reported in the revised manuscript from the extraction package (roi_package_v2.zip).

Usage
-----
    python scripts/02_run_all_analyses.py --package data/roi_package_v2 --out results

Light-weight: works on parcel-level time series (~172 participants × 152 volumes × 100 parcels).
With the default settings it takes roughly 20–60 min on a laptop with 8–16 cores.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import auc as sk_auc, roc_auc_score, roc_curve  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

from sczfc import clinical, cohorts, data, denoise, interpret, ml, qcfc, report  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
CONF4 = ["age", "male", "scanner_b", "mean_fd"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_features(pkg, subj):
    """Fisher-z connectivity vectors for every denoising strategy (cached)."""
    feats, keep, nreg = {}, {}, {}
    ids = subj["participant_id"].tolist()
    for s in denoise.STRATEGIES:
        X = np.full((len(ids), 4950), np.nan)
        nk, nr = np.zeros(len(ids), int), np.zeros(len(ids), int)
        for k, sid in enumerate(ids):
            raw, conf, csf = data.load_raw_ts(pkg, sid), data.read_confounds(pkg, sid), data.load_csf(pkg, sid)
            ts, nk[k], nr[k] = denoise.clean_timeseries(raw, conf, csf, s)
            if ts is not None and np.isfinite(ts).all():
                X[k] = denoise.fc_vector(ts)
        feats[s], keep[s], nreg[s] = X, nk, nr
        log(f"features {s}: {np.isfinite(X).all(axis=1).sum()} complete participants")
    return feats, keep, nreg


def reproduce_original(X, y, atlas, edges, out_dir):
    """Re-run the ORIGINAL analysis code logic (non-nested grid search, Youden threshold on the same out-of-fold
    predictions) on the original 99 participants with the confounds that were actually regressed."""
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("logreg", LogisticRegression(l1_ratio=0.0, solver="liblinear", class_weight="balanced",
                                                   max_iter=5000))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, {"logreg__C": [0.1, 1.0, 3.0, 10.0]}, cv=cv, scoring="roc_auc").fit(X, y)
    pipe.set_params(logreg__C=gs.best_params_["logreg__C"])
    grid, tprs, aucs = np.linspace(0, 1, 200), [], []
    for tr, te in cv.split(X, y):
        pipe.fit(X[tr], y[tr])
        f, t, _ = roc_curve(y[te], pipe.predict_proba(X[te])[:, 1])
        aucs.append(sk_auc(f, t))
        ti = np.interp(grid, f, t)
        ti[0] = 0
        tprs.append(ti)
    mt = np.mean(tprs, axis=0)
    mt[-1] = 1
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    f, t, thr = roc_curve(y, proba)
    j = np.argmax(t - f)
    m = ml.binary_metrics(y, proba >= thr[j])
    pipe.fit(X, y)
    coef = pipe.named_steps["logreg"].coef_.ravel()
    top = np.argsort(-np.abs(coef))[:12]
    shifted = ["Background"] + atlas["short"].tolist()   # labels as indexed by the original code
    rows = [{"rank": r + 1, "beta": float(coef[e]), "roi_i_correct": edges.at[e, "roi_i"],
             "roi_j_correct": edges.at[e, "roi_j"], "roi_i_as_published": shifted[edges.at[e, "i"]],
             "roi_j_as_published": shifted[edges.at[e, "j"]]} for r, e in enumerate(top)]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "tableS7b_original_top12_labels.csv"), index=False)
    res = {"best_C": gs.best_params_["logreg__C"], "gridsearch_best_auc": gs.best_score_,
           "mean_roc_auc": float(sk_auc(grid, mt)), "fold_auc_sd": float(np.std(aucs)),
           "oof_auc": float(roc_auc_score(y, proba)), "youden_threshold": float(thr[j]),
           "sensitivity": m["sensitivity"], "specificity": m["specificity"]}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", default="data/roi_package_v2")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-repeats", type=int, default=100, help="repetitions of nested CV (primary analysis)")
    ap.add_argument("--n-repeats-secondary", type=int, default=20, help="repetitions for the sensitivity analyses")
    ap.add_argument("--n-repeats-nonlinear", type=int, default=20)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--reuse-permutation", action="store_true",
                    help="load results/permutation_null_primary.npy instead of recomputing the null distribution")
    args = ap.parse_args()

    out, figs, tabs = args.out, os.path.join(args.out, "figures"), os.path.join(args.out, "tables")
    for d in (out, figs, tabs):
        os.makedirs(d, exist_ok=True)
    summary = {"settings": vars(args)}

    # ------------------------------------------------------------------ data
    pkg = data.open_package(args.package)
    atlas = data.load_atlas(pkg)
    edges = denoise.edge_table(atlas)
    subj = data.build_subject_table(pkg)
    ok = subj[subj["status"] == "ok"].reset_index(drop=True)
    clin = clinical.clinical_variables(pkg, ok["participant_id"])
    ok = ok.merge(clin, on="participant_id", how="left")
    log(f"extracted participants: {len(ok)} (SZ {ok.y.sum()}, HC {(1 - ok.y).sum()})")
    val_path = os.path.join(pkg, "logs", "validation_original_pipeline.csv")
    if os.path.isfile(val_path):
        val = pd.read_csv(val_path)
        summary["validation_original_timeseries"] = {
            c: float(val[c].median()) for c in val.columns if c.startswith(("corr_", "maxabsdiff_"))}
        summary["validation_original_timeseries"]["n"] = int(len(val))

    feats, nkeep, nreg = build_features(pkg, ok)
    ok["n_keep_primary"] = nkeep["P1_36P_cens05"]
    ok["complete_coverage"] = (ok["n_empty_rois"].fillna(0) == 0) & ok["has_csf"].fillna(False).astype(bool)
    ok["excl_lenient"] = cohorts.lenient_exclusion(ok, pd.Series(nkeep["P1_36P_cens05"])) | ~ok["complete_coverage"]
    ok["excl_stringent"] = cohorts.stringent_exclusion(ok, pd.Series(nkeep["P4_36P_cens02"])) | \
        ~ok["complete_coverage"]
    ok.to_csv(os.path.join(tabs, "participants_analysis_table.csv"), index=False)

    # ------------------------------------------------------------------ cohorts
    idx = {}
    idx["primary"] = np.flatnonzero(~ok["excl_lenient"].to_numpy())
    idx["stringent"] = np.flatnonzero(~ok["excl_stringent"].to_numpy())
    # inclusive motion criteria (sensitivity): mean FD > 0.5 mm or < 2 min (60 volumes) left after censoring
    excl_incl = (ok["mean_fd"] > 0.5) | (ok["n_keep_primary"] < 60) | ~ok["complete_coverage"]
    idx["inclusive"] = np.flatnonzero(~excl_incl.to_numpy())
    prim = ok.iloc[idx["primary"]]
    psm = cohorts.propensity_match(prim, ["age", "male", "scanner_b"], caliper_sd=0.2)
    idx["psm"] = np.array(sorted(ok.index.get_indexer(psm.index)))
    idx["scanner_a"] = idx["primary"][prim["scanner_b"].to_numpy() == 0]
    idx["no_artifact"] = idx["primary"][((prim["t1_ghost"] == 0) & (prim["pe_deviation"] == 0)).to_numpy()]
    idx["no_schizoaffective"] = idx["primary"][(prim["dx_schizoaffective"].fillna(0) == 0).to_numpy()]
    orig = ok[ok["in_original_cohort"]].sort_values("original_order")
    idx["original"] = orig.index.to_numpy()
    balance = pd.concat([cohorts.balance_table(ok.iloc[idx[c]], ["age", "male", "scanner_b", "mean_fd"])
                         .assign(cohort=c) for c in ("primary", "psm", "original")])
    balance.to_csv(os.path.join(tabs, "tableS_balance_smd.csv"), index=False)
    summary["cohorts"] = {c: {"n": int(len(v)), "n_sz": int(ok.y.iloc[v].sum()),
                              "n_hc": int((1 - ok.y.iloc[v]).sum())} for c, v in idx.items()}
    log(f"cohorts: {summary['cohorts']}")

    # ------------------------------------------------------------------ reproduction of the first submission
    Xo = feats["P0_executed"][idx["original"]]
    yo = ok["y"].to_numpy()[idx["original"]]
    if len(yo) and np.isfinite(Xo).all():
        summary["reproduction_original"] = reproduce_original(Xo, yo, atlas, edges, tabs)
        log(f"reproduction of the original analysis: {summary['reproduction_original']}")

    # ------------------------------------------------------------------ QC-FC per denoising strategy
    P = ok.iloc[idx["primary"]]
    qc_rows, qc_r = [], {}
    cov = P[["age", "male", "scanner_b", "y"]].to_numpy(float)
    for s, spec in denoise.STRATEGIES.items():
        X = feats[s][idx["primary"]]
        good = np.isfinite(X).all(axis=1)
        res, r = qcfc.qcfc_summary(X[good], P["mean_fd"].to_numpy()[good], cov[good], edges["distance_mm"].to_numpy())
        qc_r[s] = r
        res.update({"strategy": s, "label": spec["label"], "n": int(good.sum()),
                    "mean_regressors": float(nreg[s][idx["primary"]].mean()),
                    "mean_censored_sz": float(152 - nkeep[s][idx["primary"]][P["y"].to_numpy() == 1].mean()),
                    "mean_censored_hc": float(152 - nkeep[s][idx["primary"]][P["y"].to_numpy() == 0].mean())})
        qc_rows.append(res)
    qc = pd.DataFrame(qc_rows)
    qc.to_csv(os.path.join(tabs, "tableS1_denoising_qcfc.csv"), index=False)
    report.qcfc_figure(qc[qc["strategy"] != "P0_executed"].reset_index(drop=True), qc_r, edges["distance_mm"].to_numpy(), figs)
    log("QC-FC done")

    # ------------------------------------------------------------------ classification analyses
    R, R2, R3 = args.n_repeats, args.n_repeats_secondary, args.n_repeats_nonlinear
    analyses = [
        # key, label, role, strategy, cohort, model, confounds, repeats
        ("primary", "Primary: 36P + censoring, confounds regressed", "primary", "P1_36P_cens05", "primary", "LR", CONF4, R),
        ("no_conf", "Without confound regression", "sensitivity", "P1_36P_cens05", "primary", "LR", [], R2),
        ("p2", "24P + aCompCor + GSR + censoring", "sensitivity", "P2_24P_aCC_GSR_cens05", "primary", "LR", CONF4, R2),
        ("p3", "36P without GSR + censoring", "sensitivity", "P3_32P_noGSR_cens05", "primary", "LR", CONF4, R2),
        ("p5", "24P only", "sensitivity", "P5_24P", "primary", "LR", CONF4, R2),
        ("p0_desc", "6P + WM + GS + aCompCor (no censoring)", "sensitivity", "P0_described", "primary", "LR", CONF4, R2),
        ("p0_exec", "Pipeline of the first submission (letter only)", "letter", "P0_executed", "primary", "LR", CONF4, R2),
        ("stringent", "Stringent motion criteria", "sensitivity", "P4_36P_cens02", "stringent", "LR", CONF4, R2),
        ("inclusive", "Inclusive motion criteria (≥ 2 min of data)", "sensitivity", "P1_36P_cens05", "inclusive", "LR", CONF4, R2),
        ("psm", "Propensity-matched subsample", "sensitivity", "P1_36P_cens05", "psm", "LR", CONF4, R2),
        ("scanner_a", "Scanner A only", "sensitivity", "P1_36P_cens05", "scanner_a", "LR", ["age", "male", "mean_fd"], R2),
        ("no_artifact", "Excluding T1 artefact or non-standard phase encoding", "sensitivity", "P1_36P_cens05", "no_artifact", "LR", CONF4, R2),
        ("no_schizoaffective", "Excluding schizoaffective disorder", "sensitivity", "P1_36P_cens05", "no_schizoaffective", "LR", CONF4, R2),
        ("orig_p1", "Age- and sex-matched subsample", "sensitivity", "P1_36P_cens05", "original", "LR", CONF4, R2),
        ("orig_p0", "First submission, nested CV (letter only)", "letter", "P0_executed", "original", "LR", [], R2),
        ("rf", "Random forest", "sensitivity", "P1_36P_cens05", "primary", "RF", CONF4, R3),
        ("svm", "Linear SVM", "sensitivity", "P1_36P_cens05", "primary", "SVM", CONF4, R3),
        ("conf_only", "Confounds only (age, sex, scanner, FD)", "baseline", None, "primary", "LR_confounds", CONF4, R2),
        ("fd_only", "Mean FD only", "baseline", None, "primary", "LR_confounds", ["mean_fd"], R2),
        ("scanner_only", "Scanner only", "baseline", None, "primary", "LR_confounds", ["scanner_b"], R2),
    ]
    results, scores_store = [], {}
    for key, label, role, strat, coh, model, confs, reps in analyses:
        sel = idx[coh]
        D = ok.iloc[sel]
        y = D["y"].to_numpy()
        if strat is None:
            X, n_conf = D[confs].to_numpy(float), 0
        else:
            F = feats[strat][sel]
            good = np.isfinite(F).all(axis=1)
            F, D, y = F[good], D[good], y[good]
            X = np.column_stack([F, D[confs].to_numpy(float)]) if confs else F
            n_conf = len(confs)
        if min(int(y.sum()), int((1 - y).sum())) < 10:
            log(f"{key}: skipped (fewer than 10 participants in one group)")
            results.append({"key": key, "label": label, "role": role, "skipped": True, "n": int(len(y))})
            continue
        t0 = time.time()
        sc, params = ml.nested_cv(X, y, model, n_conf, n_repeats=reps, seed=args.seed, n_jobs=args.n_jobs)
        res = ml.summarize_scores(sc, y, ml.threshold_for(model), seed=args.seed)
        if key == "scanner_only":   # binary predictor: the AUC equals the balanced accuracy of the 2x2 table
            x = D["scanner_b"].to_numpy()
            tpr, tnr = x[y == 1].mean(), 1 - x[y == 0].mean()
            res["auc"] = float((tpr + tnr) / 2)
            lo, hi = ml.bootstrap_auc_ci(y, x.astype(float), seed=args.seed)
            res["auc_ci_low"], res["auc_ci_high"] = lo, hi
        res.update({"key": key, "label": label, "role": role, "strategy": strat, "cohort": coh, "model": model,
                    "confounds": ",".join(confs), "repeats": reps, "minutes": round((time.time() - t0) / 60, 2)})
        if model in ("LR", "SVM"):
            cs = pd.Series([p["clf__C"] for p in params])
            res["selected_C_median"] = float(cs.median())
        results.append(res)
        scores_store[key] = {"ids": D["participant_id"].tolist(), "y": y, "scores": sc}
        log(f"{key}: AUC {res['auc']:.3f} [{res['auc_ci_low']:.3f}, {res['auc_ci_high']:.3f}] "
            f"BAcc {res['balanced_accuracy']:.3f} ({res['minutes']} min)")
    resdf = pd.DataFrame(results)
    np.savez_compressed(os.path.join(out, "oof_scores_all_analyses.npz"),
                        **{f"{k}__scores": v["scores"] for k, v in scores_store.items()},
                        **{f"{k}__y": v["y"] for k, v in scores_store.items()},
                        **{f"{k}__ids": np.array(v["ids"]) for k, v in scores_store.items()})

    # paired comparisons on the repetition-averaged scores (same participants)
    def avg(k):
        return np.nanmean(scores_store[k]["scores"], axis=0)
    yp = scores_store["primary"]["y"]
    comps = []
    for k in ("rf", "svm", "conf_only", "no_conf"):
        if k in scores_store and scores_store[k]["ids"] == scores_store["primary"]["ids"]:
            d, z, p = ml.delong_test(yp, avg("primary"), avg(k))
            comps.append({"comparison": f"primary vs {k}", "auc_difference": d, "z": z, "p_delong": p})
    pd.DataFrame(comps).to_csv(os.path.join(tabs, "tableS2b_auc_comparisons.csv"), index=False)
    resdf.to_csv(os.path.join(tabs, "tableS2_classification_all.csv"), index=False)
    summary["comparisons"] = comps

    # ------------------------------------------------------------------ permutation test (primary)
    sel = idx["primary"]
    D = ok.iloc[sel]
    F = feats["P1_36P_cens05"][sel]
    Xp = np.column_stack([F, D[CONF4].to_numpy(float)])
    yp = D["y"].to_numpy()
    t0 = time.time()
    null_path = os.path.join(out, "permutation_null_primary.npy")
    if args.reuse_permutation and os.path.isfile(null_path):
        obs, _, _ = ml.permutation_test(Xp, yp, "LR", len(CONF4), n_perm=0, seed=args.seed, n_jobs=args.n_jobs)
        null = np.load(null_path)
        p_perm = float((1 + np.sum(null >= obs)) / (1 + len(null)))
    else:
        obs, null, p_perm = ml.permutation_test(Xp, yp, "LR", len(CONF4), n_perm=args.n_perm, seed=args.seed,
                                                n_jobs=args.n_jobs)
    np.save(os.path.join(out, "permutation_null_primary.npy"), null)
    summary["permutation"] = {"observed_auc_single_repetition": obs, "p": p_perm, "n_perm": int(len(null)),
                              "null_mean": float(null.mean()), "null_sd": float(null.std()),
                              "null_max": float(null.max()), "minutes": round((time.time() - t0) / 60, 1)}
    log(f"permutation: {summary['permutation']}")

    # ------------------------------------------------------------------ interpretation (primary model)
    C_full = interpret.select_c(Xp, yp, len(CONF4), seed=args.seed)
    est = interpret.fit_lr(Xp, yp, len(CONF4), C_full)
    w, a = interpret.weights_and_pattern(est, Xp)
    W, A = interpret.bootstrap_models(Xp, yp, len(CONF4), C_full, n_boot=args.n_boot, seed=args.seed,
                                      n_jobs=args.n_jobs)
    stab = interpret.stability_table(w, a, W, A, edges)
    diff, tval, pval, qval = interpret.group_glm(F, yp, D[CONF4[:-1] + ["mean_fd"]].to_numpy(float))
    stab["mean_fc_sz"], stab["mean_fc_hc"] = F[yp == 1].mean(axis=0), F[yp == 0].mean(axis=0)
    stab["adj_diff_sz_minus_hc"], stab["t_group"], stab["p_group"], stab["q_group"] = diff, tval, pval, qval
    stab["cohens_d"] = interpret.cohens_d(F, yp)
    stab.to_csv(os.path.join(tabs, "tableS5_edge_stability_all.csv"), index=False)
    stable = stab[np.abs(stab["bsr"]) >= 3.29].sort_values("bsr", key=np.abs, ascending=False)
    summary["interpretation"] = {"C_full": C_full, "n_stable_edges_bsr3.29": int(len(stable)),
                                 "n_edges_q05_univariate": int((qval < 0.05).sum()),
                                 "max_freq_top1pct": float(stab["freq_top"].max())}
    t2 = stable.head(15)[["roi_i", "net_i", "roi_j", "net_j", "pattern", "bsr", "freq_top", "sign_consistency",
                          "mean_fc_hc", "mean_fc_sz", "adj_diff_sz_minus_hc", "cohens_d", "q_group"]]
    t2.to_csv(os.path.join(tabs, "table2_stable_edges.csv"), index=False)

    pair_of_edge, pairs = interpret.network_index(edges)
    net_obs = interpret.network_means(a, pair_of_edge, pairs)[0]
    net_boot = interpret.network_means(A, pair_of_edge, pairs)
    p_net = 2 * np.minimum((net_boot <= 0).mean(axis=0), (net_boot >= 0).mean(axis=0))
    p_net = np.clip(p_net, 1 / len(net_boot), 1)
    q_net = multipletests(p_net, method="fdr_bh")[1]
    fc_net = interpret.network_means(F, pair_of_edge, pairs)
    nd, nt, npv, nq = interpret.group_glm(fc_net, yp, D[CONF4].to_numpy(float))
    net_tab = pd.DataFrame({"network_pair": pairs, "mean_pattern": net_obs,
                            "pattern_ci_low": np.quantile(net_boot, 0.025, axis=0),
                            "pattern_ci_high": np.quantile(net_boot, 0.975, axis=0), "p_boot": p_net, "q_boot": q_net,
                            "fc_adj_diff_sz_minus_hc": nd, "fc_t": nt, "fc_p": npv, "fc_q": nq})
    net_tab.to_csv(os.path.join(tabs, "tableS3_network_level.csv"), index=False)
    M = np.zeros((7, 7))
    S = np.zeros((7, 7), bool)
    for (pname, v, q) in zip(pairs, net_obs, q_net):
        a_, b_ = pname.split("–")
        i, j = interpret.NETWORKS.index(a_), interpret.NETWORKS.index(b_)
        i, j = max(i, j), min(i, j)
        M[i, j], S[i, j] = v, q < 0.05
    report.network_figure(M, S, stable.head(30), atlas, figs)

    # group-mean matrices (supplementary figure)
    def to_mat(v):
        Mx = np.zeros((100, 100))
        Mx[denoise.IU] = v
        return Mx + Mx.T
    report.group_matrices_figure(to_mat(F[yp == 0].mean(0)), to_mat(F[yp == 1].mean(0)), to_mat(tval), atlas, figs)

    # ------------------------------------------------------------------ clinical / confound associations
    s_bar = avg("primary")
    Dp = D.assign(score=s_bar)
    rows = []
    for grp, gname in ((1, "SZ"), (0, "HC")):
        G = Dp[Dp["y"] == grp]
        for v in (["mean_fd", "age"] + (["bprs_total", "saps_global_sum", "sans_global_sum", "hamd17",
                                         "wais_vocabulary_raw", "wais_matrix_raw", "antipsychotic_any"]
                                        if grp == 1 else ["wais_vocabulary_raw", "wais_matrix_raw"])):
            g = G[["score", v]].dropna()
            if len(g) > 5 and g[v].nunique() > 1:
                rho, p = stats.spearmanr(g["score"], g[v])
                rows.append({"group": gname, "variable": v, "n": len(g), "spearman_rho": rho, "p": p})
    assoc = pd.DataFrame(rows)
    if len(assoc):
        assoc["q_fdr"] = multipletests(assoc["p"], method="fdr_bh")[1]
    assoc.to_csv(os.path.join(tabs, "tableS4_score_associations.csv"), index=False)
    pd.DataFrame({"participant_id": D["participant_id"], "y": yp, "oof_score_mean": s_bar}).to_csv(
        os.path.join(tabs, "primary_oof_scores.csv"), index=False)

    # ------------------------------------------------------------------ tables 1 and figures 1–2
    report.table1(ok.iloc[idx["primary"]]).to_csv(os.path.join(tabs, "table1_primary_sample.csv"), index=False)
    report.table1(ok.iloc[idx["original"]]).to_csv(os.path.join(tabs, "tableS6_matched_subsample.csv"), index=False)
    report.table1(ok.iloc[idx["psm"]]).to_csv(os.path.join(tabs, "tableS6b_psm_sample.csv"), index=False)

    allp = pd.read_csv(os.path.join(pkg, "participants.tsv"), sep="\t")
    n_all = len(allp)
    n_sz_all, n_hc_all = int((allp.diagnosis == "SCHZ").sum()), int((allp.diagnosis == "CONTROL").sum())
    n_other = n_all - n_sz_all - n_hc_all
    rest = allp[allp.diagnosis.isin(["SCHZ", "CONTROL"]) & (allp.rest == 1)]
    n_rest_sz, n_rest_hc = int((rest.diagnosis == "SCHZ").sum()), int((rest.diagnosis == "CONTROL").sum())
    n_ok_sz, n_ok_hc = int(ok.y.sum()), int((1 - ok.y).sum())
    exl = ok[ok["excl_lenient"]]
    prim_c = summary["cohorts"]["primary"]
    analysed = resdf.set_index("key")

    def n_analysed(key, cohort):
        # participants actually analysed (some lack enough volumes after censoring); cohort size if skipped
        r = analysed.loc[key] if key in analysed.index else None
        if r is not None and pd.notna(r.get("n_sz")):
            return f"SZ {int(r['n_sz'])}, HC {int(r['n_hc'])}"
        c = summary["cohorts"][cohort]
        return f"SZ {c['n_sz']}, HC {c['n_hc']}"

    counts = {
        "all": f"OpenNeuro ds000030 (UCLA CNP)\nn = {n_all}",
        "groups": f"Schizophrenia n = {n_sz_all}\nHealthy controls n = {n_hc_all}",
        "excl_groups": f"Other diagnoses (bipolar disorder,\nADHD), not analysed: n = {n_other}",
        "rest": f"Resting-state fMRI available\nSZ n = {n_rest_sz}; HC n = {n_rest_hc}",
        "excl_rest": f"No resting-state scan\nSZ n = {n_sz_all - n_rest_sz}; HC n = {n_hc_all - n_rest_hc}",
        "deriv": f"fMRIPrep derivatives available\nSZ n = {n_ok_sz}; HC n = {n_ok_hc}",
        "excl_deriv": f"No T1-weighted image (not preprocessed)\nSZ n = {n_rest_sz - n_ok_sz}; HC n = {n_rest_hc - n_ok_hc}",
        "excl_motion": f"Excluded: head motion*\nSZ n = {int(exl.y.sum())}; HC n = {int((1 - exl.y).sum())}",
        "final": f"Primary analytic sample\nSZ n = {prim_c['n_sz']}; HC n = {prim_c['n_hc']}",
        "subsamples": ("Other samples analysed (sensitivity analyses)\n"
                       f"Inclusive motion criteria: {n_analysed('inclusive', 'inclusive')};   "
                       f"stringent motion criteria: {n_analysed('stringent', 'stringent')}\n"
                       f"Propensity-matched: {n_analysed('psm', 'psm')};   "
                       f"age- and sex-matched, no motion criteria: {n_analysed('orig_p1', 'original')}"),
        "footnote": ("*Mean framewise displacement > 0.5 mm, or < 4 min of data after censoring volumes with "
                     "FD > 0.5 mm."),
    }
    report.flow_diagram(counts, figs)
    forest = resdf.loc[resdf["auc"].notna() & (resdf["role"] != "letter"), ["label", "role", "auc", "auc_ci_low", "auc_ci_high"]]
    report.performance_figure(yp, s_bar, scores_store["primary"]["scores"], avg("conf_only"), null, obs, p_perm,
                              forest, figs)

    summary["primary"] = resdf[resdf.key == "primary"].iloc[0].to_dict()
    summary["qcfc"] = qc_rows
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log("all done")


if __name__ == "__main__":
    main()
