"""sczfc: resting-state functional connectivity classification of schizophrenia (UCLA CNP, ds000030).

Modules
-------
data       loading of the extraction package and participant-level variables
clinical   phenotype variables (symptoms, cognition, medication) for descriptive tables
denoise    confound strategies, censoring, cleaning and connectivity features
qcfc       motion quality-control benchmarks (QC-FC, distance dependence)
cohorts    motion exclusion rules and propensity-score matching
ml         nested repeated cross-validation, confound control, metrics and inference
interpret  Haufe activation patterns, bootstrap stability, network summaries, univariate tests
report     tables and figures
"""

__version__ = "1.0.0"
