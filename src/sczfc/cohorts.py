"""Motion exclusion rules and propensity-score matching."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

MIN_VOLUMES = 120  # 4 min of uncensored data at TR = 2 s (Parkes et al., 2018)


def lenient_exclusion(df: pd.DataFrame, n_keep_05: pd.Series) -> pd.Series:
    """Primary rule: mean FD > 0.5 mm, or < 4 min of data left after censoring volumes with FD > 0.5 mm."""
    return (df["mean_fd"] > 0.5) | (n_keep_05 < MIN_VOLUMES)


def stringent_exclusion(df: pd.DataFrame, n_keep_02: pd.Series) -> pd.Series:
    """Stringent rule (Parkes et al., 2018): mean FD > 0.25 mm, > 20% of FD values > 0.2 mm, any FD > 5 mm,
    or < 4 min of data left after censoring volumes with FD > 0.2 mm."""
    return ((df["mean_fd"] > 0.25) | (df["pct_fd_gt_0p2"] > 20) | (df["max_fd"] > 5)
            | (n_keep_02 < MIN_VOLUMES))


def smd(a: pd.Series, b: pd.Series) -> float:
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def balance_table(df: pd.DataFrame, covars, group_col="y") -> pd.DataFrame:
    s, c = df[df[group_col] == 1], df[df[group_col] == 0]
    return pd.DataFrame({"covariate": covars, "smd": [smd(s[v], c[v]) for v in covars]})


def propensity_match(df: pd.DataFrame, covars, caliper_sd: float = 0.2, seed: int = 0) -> pd.DataFrame:
    """1:1 nearest-neighbour matching without replacement on the logit of the propensity score, with a caliper
    of `caliper_sd` standard deviations of the logit (Austin, 2011). Patients are matched in descending order of
    propensity (hardest first). Returns the matched subset of df."""
    X = df[covars].to_numpy(float)
    ps = LogisticRegression(C=1e6, max_iter=10000).fit(X, df["y"]).predict_proba(X)[:, 1]
    logit = np.log(ps / (1 - ps))
    d = df.assign(_logit=logit)
    cal = caliper_sd * np.std(logit, ddof=1)
    treated = d[d["y"] == 1].sort_values("_logit", ascending=False)
    pool = d[d["y"] == 0].copy()
    pairs = []
    for i, row in treated.iterrows():
        if pool.empty:
            break
        dist = (pool["_logit"] - row["_logit"]).abs()
        j = dist.idxmin()
        if dist[j] <= cal:
            pairs.append((i, j))
            pool = pool.drop(index=j)
    idx = [a for a, _ in pairs] + [b for _, b in pairs]
    return df.loc[idx].copy()
