"""Confound-regression strategies, volume censoring, cleaning and connectivity features."""
from __future__ import annotations

import numpy as np
import pandas as pd
from nilearn import signal

from .data import framewise_displacement

T_R = 2.0
MOTION = ["X", "Y", "Z", "RotX", "RotY", "RotZ"]
ACOMPCOR5 = [f"aCompCor0{k}" for k in range(5)]

# Denoising strategies. "P0_*" reproduce the first submission; P1 is the primary pipeline of the revision.
STRATEGIES = {
    # what the original code actually regressed (fallback: first 12 numeric confound columns)
    "P0_executed": {"confounds": "fallback12", "censor_fd": None, "high_pass": None,
                    "label": "WM + GS + DVARS×3 + FD + tCompCor×6 (first submission)"},
    # what the first submission described (CSF unavailable in this fMRIPrep release)
    "P0_described": {"confounds": "described", "censor_fd": None, "high_pass": None,
                     "label": "6P + WM + GS + aCompCor×5, no censoring"},
    "P1_36P_cens05": {"confounds": "36P", "censor_fd": 0.5, "high_pass": 0.01,
                      "label": "36P + censoring FD>0.5 mm (primary)"},
    "P2_24P_aCC_GSR_cens05": {"confounds": "24P_acompcor_gsr", "censor_fd": 0.5, "high_pass": 0.01,
                              "label": "24P + aCompCor×5 + GSR + censoring FD>0.5 mm"},
    "P3_32P_noGSR_cens05": {"confounds": "36P_nogsr", "censor_fd": 0.5, "high_pass": 0.01,
                            "label": "36P without GSR (32P) + censoring FD>0.5 mm"},
    "P4_36P_cens02": {"confounds": "36P", "censor_fd": 0.2, "high_pass": 0.01,
                      "label": "36P + censoring FD>0.2 mm"},
    "P5_24P": {"confounds": "24P", "censor_fd": None, "high_pass": 0.01,
               "label": "24P only (no physiological regressors, no censoring)"},
}


def _derivative(a: np.ndarray) -> np.ndarray:
    """Backward difference with a zero first row (fMRIPrep convention)."""
    return np.vstack([np.zeros((1, a.shape[1])), np.diff(a, axis=0)])


def expand(a: np.ndarray) -> np.ndarray:
    """[x, dx/dt, x^2, (dx/dt)^2] — Friston 24P / Satterthwaite 36P style expansion."""
    d = _derivative(a)
    return np.hstack([a, d, a ** 2, d ** 2])


def confound_matrix(conf: pd.DataFrame, csf: np.ndarray | None, kind: str) -> np.ndarray:
    if kind == "fallback12":
        cols = conf.select_dtypes(include=["number"]).columns.tolist()[:12]
        return conf[cols].fillna(0.0).to_numpy(float)
    if kind == "described":
        return conf[MOTION + ["WhiteMatter", "GlobalSignal"] + ACOMPCOR5].fillna(0.0).to_numpy(float)
    motion = conf[MOTION].to_numpy(float)
    if kind == "24P":
        return expand(motion)
    if kind == "24P_acompcor_gsr":
        return np.hstack([expand(motion), conf[ACOMPCOR5 + ["GlobalSignal"]].to_numpy(float)])
    if kind in ("36P", "36P_nogsr"):
        phys = [conf["WhiteMatter"].to_numpy(float)]
        if csf is not None:
            phys.append(np.asarray(csf, float))
        if kind == "36P":
            phys.append(conf["GlobalSignal"].to_numpy(float))
        return expand(np.column_stack([motion] + phys))
    raise ValueError(kind)


def censor_mask(fd: np.ndarray, threshold: float, before: int = 1, after: int = 2, min_segment: int = 5):
    """Volumes kept after censoring (Power et al., 2014): each volume with FD > threshold is removed together with
    `before` preceding and `after` following volumes; uncensored segments shorter than `min_segment` are removed."""
    n = len(fd)
    cens = np.zeros(n, bool)
    for t in np.flatnonzero(fd > threshold):
        cens[max(0, t - before):min(n, t + after + 1)] = True
    keep = ~cens
    t = 0
    while t < n:
        if keep[t]:
            u = t
            while u < n and keep[u]:
                u += 1
            if u - t < min_segment:
                keep[t:u] = False
            t = u
        else:
            t += 1
    return keep


def clean_timeseries(raw: np.ndarray, conf: pd.DataFrame, csf: np.ndarray | None, strategy: str):
    """Return cleaned time series (kept volumes × ROIs), number of kept volumes and number of regressors."""
    spec = STRATEGIES[strategy]
    C = confound_matrix(conf, csf, spec["confounds"])
    kwargs = dict(detrend=True, standardize="zscore_sample", confounds=C, standardize_confounds=True, t_r=T_R)
    if spec["high_pass"]:
        kwargs.update(filter="cosine", high_pass=spec["high_pass"])
    else:
        kwargs.update(filter=False)
    n_keep = raw.shape[0]
    if spec["censor_fd"] is not None:
        keep = censor_mask(framewise_displacement(conf), spec["censor_fd"])
        n_keep = int(keep.sum())
        if n_keep < 20:
            return None, n_keep, C.shape[1]
        kwargs["sample_mask"] = np.flatnonzero(keep)
    ts = signal.clean(raw, **kwargs)
    return ts, n_keep, C.shape[1]


IU = np.triu_indices(100, k=1)


def fc_vector(ts: np.ndarray) -> np.ndarray:
    """Pearson correlation -> Fisher z (clipped as in the original code) -> upper triangle (4,950 edges)."""
    r = np.corrcoef(ts, rowvar=False)
    r = np.clip(np.nan_to_num(r, nan=0.0), -0.999999, 0.999999)
    return np.arctanh(r)[IU]


def edge_table(atlas: pd.DataFrame) -> pd.DataFrame:
    i, j = IU
    e = pd.DataFrame({"i": i, "j": j})
    for side, idx in (("i", i), ("j", j)):
        e[f"roi_{side}"] = atlas["short"].to_numpy()[idx]
        e[f"net_{side}"] = atlas["network"].to_numpy()[idx]
        e[f"hemi_{side}"] = atlas["hemisphere"].to_numpy()[idx]
    xyz = atlas[["x_mni", "y_mni", "z_mni"]].to_numpy(float)
    e["distance_mm"] = np.linalg.norm(xyz[i] - xyz[j], axis=1)
    return e
