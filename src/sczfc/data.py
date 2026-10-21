"""Loading of the extraction package produced by notebooks/01_extract_roi_timeseries_colab.ipynb."""
from __future__ import annotations

import glob
import json
import os
import zipfile

import numpy as np
import pandas as pd

# Scanner serial numbers in participants.tsv. Both are 3T Siemens Trio systems at UCLA; the serial number
# maps one-to-one onto the syngo software version recorded in the BIDS sidecars (B15 vs B17).
SCANNER_B = 35426.0  # coded 1 ("scanner B", syngo MR B17); 35343 is coded 0 ("scanner A", syngo MR B15)
N_ROIS = 100


def open_package(path: str) -> str:
    """Return the package folder. If it does not exist, extract `<path>.zip` or the split archives
    `<path>_timeseries.zip` and `<path>_metadata.zip` into it."""
    if os.path.isdir(path):
        return path
    base = path[:-4] if path.endswith(".zip") else path
    parts = sorted(glob.glob(base + "_*.zip")) or ([base + ".zip"] if os.path.isfile(base + ".zip") else [])
    if not parts:
        raise FileNotFoundError(f"Package not found: {path}")
    os.makedirs(base, exist_ok=True)
    for p in parts:
        with zipfile.ZipFile(p) as z:
            z.extractall(base)
    return base


def load_atlas(pkg: str) -> pd.DataFrame:
    atlas = pd.read_csv(os.path.join(pkg, "atlas_schaefer2018_100p_7n.csv"))
    assert len(atlas) == N_ROIS and (atlas["roi_index"].to_numpy() == np.arange(N_ROIS)).all()
    atlas["short"] = atlas["label"].str.replace("7Networks_", "", regex=False).str.replace("_", " ")
    return atlas


def read_confounds(pkg: str, sid: str) -> pd.DataFrame:
    # 'n/a' (first row of DVARS/FD) is read as NaN by pandas' default NA values
    return pd.read_csv(os.path.join(pkg, "confounds", f"{sid}_task-rest_bold_confounds.tsv"), sep="\t")


def load_raw_ts(pkg: str, sid: str) -> np.ndarray:
    return np.load(os.path.join(pkg, "roi_raw", f"{sid}_schaefer100_raw.npy")).astype(np.float64)


def load_csf(pkg: str, sid: str) -> np.ndarray | None:
    path = os.path.join(pkg, "csf", f"{sid}_csf_raw.npy")
    return np.load(path).astype(np.float64) if os.path.isfile(path) else None


def framewise_displacement(conf: pd.DataFrame) -> np.ndarray:
    """Power's FD as computed by fMRIPrep (first volume set to 0)."""
    fd = np.array(pd.to_numeric(conf["FramewiseDisplacement"], errors="coerce"), dtype=float)  # writable copy
    fd[0] = 0.0
    return np.nan_to_num(fd, nan=0.0)


def motion_metrics(conf: pd.DataFrame) -> dict:
    fd = framewise_displacement(conf)[1:]
    return {
        "mean_fd": float(np.mean(fd)),
        "max_fd": float(np.max(fd)),
        "pct_fd_gt_0p2": float(100 * np.mean(fd > 0.2)),
        "n_fd_gt_0p5": int(np.sum(fd > 0.5)),
    }


def _phase_encoding(pkg: str, sid: str) -> str | None:
    path = os.path.join(pkg, "sidecars", f"{sid}_task-rest_bold.json")
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        meta = json.load(fh)
    return meta.get("PhaseEncodingDirection")


def build_subject_table(pkg: str) -> pd.DataFrame:
    """One row per extracted participant with group, demographics, scanner, QC flags and motion metrics."""
    parts = pd.read_csv(os.path.join(pkg, "participants.tsv"), sep="\t")
    man = pd.read_csv(os.path.join(pkg, "manifest_extraction.csv"))
    df = man.merge(parts.drop(columns=["diagnosis"]), on="participant_id", how="left")
    df["group"] = df["diagnosis"].map({"SCHZ": "SZ", "CONTROL": "HC"})
    df["y"] = (df["group"] == "SZ").astype(int)
    df["male"] = (df["gender"] == "M").astype(int)
    df["scanner_b"] = (df["ScannerSerialNumber"].round() == SCANNER_B).astype(int)
    df["t1_ghost"] = (df["ghost_NoGhost"] == "ghost").astype(int)

    orig_path = os.path.join(pkg, "participants_original_manual_cohort.tsv")
    orig_ids = pd.read_csv(orig_path, sep="\t")["participant_id"].astype(str).str.strip().tolist() \
        if os.path.isfile(orig_path) else []
    orig_ids = [i if i.startswith("sub-") else f"sub-{i}" for i in orig_ids]
    df["in_original_cohort"] = df["participant_id"].isin(orig_ids)
    df["original_order"] = df["participant_id"].map({sid: k for k, sid in enumerate(orig_ids)})

    rows = []
    for sid, status in zip(df["participant_id"], df["status"]):
        rec = {"participant_id": sid, "phase_encoding": _phase_encoding(pkg, sid)}
        if status == "ok":
            rec.update(motion_metrics(read_confounds(pkg, sid)))
            rec["has_csf"] = load_csf(pkg, sid) is not None
        rows.append(rec)
    df = df.merge(pd.DataFrame(rows), on="participant_id", how="left")
    if df["phase_encoding"].notna().any():
        modal = df["phase_encoding"].mode().iloc[0]
        df["pe_deviation"] = (df["phase_encoding"].notna() & (df["phase_encoding"] != modal)).astype(int)
    else:
        df["pe_deviation"] = 0
    return df.sort_values(["y", "participant_id"], ascending=[False, True]).reset_index(drop=True)
