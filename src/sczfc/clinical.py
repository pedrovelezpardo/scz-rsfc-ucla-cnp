"""Phenotype variables (ds000030 /phenotype) used to describe the sample and for exploratory associations."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

# Antipsychotics by generation (generic and brand names as recorded in medication.tsv)
SGA = ["quetiapine", "seroquel", "aripiprazole", "abilify", "olanzapine", "zyprexa", "risperidone", "risperdal",
       "ziprasidone", "geodon", "clozapine", "clozaril", "paliperidone", "invega", "asenapine", "saphris",
       "iloperidone", "fanapt", "lurasidone", "latuda", "amisulpride"]
FGA = ["haloperidol", "haldol", "fluphenazine", "prolixin", "loxapine", "loxitane", "thiothixene", "navane",
       "chlorpromazine", "thorazine", "perphenazine", "trilafon", "trifluoperazine", "stelazine",
       "thioridazine", "mellaril", "pimozide"]
ANTIDEPRESSANT = ["paroxetine", "paxil", "escitalopram", "lexapro", "fluoxetine", "prozac", "citalopram", "celexa",
                  "sertraline", "zoloft", "bupropion", "wellbutrin", "trazodone", "mirtazapine", "remeron",
                  "venlafaxine", "effexor", "duloxetine", "cymbalta", "doxepin"]
MOOD_STABILIZER = ["lithium", "divalproex", "depakote", "valproate", "lamotrigine", "lamictal", "carbamazepine",
                   "tegretol", "oxcarbazepine", "topiramate", "topamax"]
BENZODIAZEPINE = ["clonazepam", "klonopin", "lorazepam", "ativan", "temazepam", "restoril", "alprazolam", "xanax",
                  "diazepam", "valium"]
ANTICHOLINERGIC = ["benztropine", "cogentin", "trihexyphenidyl", "artane"]

SANS_GLOBAL = ["global_bluntaffect", "global_alogia", "global_avolition", "global_anhedonia", "global_attention"]
SAPS_GLOBAL = ["global_hallucinations", "global_delusions", "global_bizarrebehav", "global_posformalthought"]


def _read(pkg: str, name: str) -> pd.DataFrame | None:
    path = os.path.join(pkg, "phenotype", f"{name}.tsv")
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path, sep="\t", low_memory=False).set_index("participant_id")


def _num(df: pd.DataFrame | None, ids, col: str) -> pd.Series:
    if df is None or col not in df.columns:
        return pd.Series(np.nan, index=ids)
    return pd.to_numeric(df.reindex(ids)[col], errors="coerce")


def medication_flags(med: pd.DataFrame | None, ids) -> pd.DataFrame:
    """Current-use flags (med_use == 1) per drug class; participants without any entry are flagged."""
    out = pd.DataFrame(index=ids)
    classes = {"antipsychotic_sga": SGA, "antipsychotic_fga": FGA, "antidepressant": ANTIDEPRESSANT,
               "mood_stabilizer": MOOD_STABILIZER, "benzodiazepine": BENZODIAZEPINE,
               "anticholinergic": ANTICHOLINERGIC}
    for c in classes:
        out[c] = 0
    out["no_medication_entries"] = 1
    if med is None:
        return out
    med = med.reindex(ids)
    for sid in ids:
        any_entry = False
        for k in range(1, 21):
            name = med.at[sid, f"med_name{k}"] if f"med_name{k}" in med.columns else np.nan
            use = med.at[sid, f"med_use{k}"] if f"med_use{k}" in med.columns else np.nan
            if not isinstance(name, str) or not name.strip():
                continue
            any_entry = True
            if pd.to_numeric(use, errors="coerce") != 1:
                continue
            low = name.lower()
            for c, words in classes.items():
                if any(w in low for w in words):
                    out.at[sid, c] = 1
        out.at[sid, "no_medication_entries"] = int(not any_entry)
    out["antipsychotic_any"] = ((out["antipsychotic_sga"] + out["antipsychotic_fga"]) > 0).astype(int)
    return out


def clinical_variables(pkg: str, ids) -> pd.DataFrame:
    ids = list(ids)
    demo, wais, cvlt = _read(pkg, "demographics"), _read(pkg, "wais"), _read(pkg, "cvlt")
    bprs, sans, saps = _read(pkg, "bprs"), _read(pkg, "sans"), _read(pkg, "saps")
    ham, ymrs, scid, med = _read(pkg, "hamilton"), _read(pkg, "ymrs"), _read(pkg, "scid"), _read(pkg, "medication")

    out = pd.DataFrame(index=ids)
    out["education_years"] = _num(demo, ids, "school_yrs")
    out["wais_vocabulary_raw"] = _num(wais, ids, "voc_totalraw")
    out["wais_matrix_raw"] = _num(wais, ids, "mr_totalraw")
    out["cvlt_trials1to5_t"] = _num(cvlt, ids, "cvltz_6")
    if bprs is not None:
        items = [f"bprs{k}" for k in range(1, 25) if f"bprs{k}" in bprs.columns]
        b = bprs.reindex(ids)[items].apply(pd.to_numeric, errors="coerce")
        out["bprs_total"] = b.sum(axis=1, min_count=len(items))   # complete cases only
    out["sans_global_sum"] = sum(_num(sans, ids, c) for c in SANS_GLOBAL)
    out["saps_global_sum"] = sum(_num(saps, ids, c) for c in SAPS_GLOBAL)
    out["hamd17"] = _num(ham, ids, "hamd_17")
    out["ymrs"] = _num(ymrs, ids, "ymrs_score")
    out["scid_dx1"] = scid.reindex(ids)["scid_dx1"] if scid is not None and "scid_dx1" in scid.columns else np.nan
    if scid is not None:
        dx = scid.reindex(ids)[[c for c in scid.columns if c.startswith("scid_dx") and c[7:].isdigit()]]
        codes = dx.apply(lambda col: col.astype(str).str.extract(r"^\s*([0-9V.]+)")[0])
        filled = codes.fillna("")
        real = (codes.notna() & ~filled.apply(lambda col: col.str.startswith("V71.09"))
                & ~filled.apply(lambda col: col.str.startswith("799.9")))
        out["scid_any_dx"] = real.any(axis=1).astype(int).where(dx.notna().any(axis=1))
        sub = codes.fillna("").apply(lambda col: col.str.match(r"^(291|292|303|304|305)"))
        mood = codes.fillna("").apply(lambda col: col.str.match(r"^(296|300\.4|311)"))
        out["scid_substance_dx"] = sub.any(axis=1).astype(int).where(dx.notna().any(axis=1))
        out["scid_mood_dx"] = mood.any(axis=1).astype(int).where(dx.notna().any(axis=1))
    dx1 = out["scid_dx1"].astype(str)
    out["dx_schizoaffective"] = dx1.str.contains("Schizoaffective", case=False).astype(int).where(out["scid_dx1"].notna())
    out = out.join(medication_flags(med, ids))
    out.index.name = "participant_id"
    return out.reset_index()
