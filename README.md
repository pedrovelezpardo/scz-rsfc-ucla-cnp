# Resting-state functional connectivity classification of schizophrenia — UCLA CNP (OpenNeuro ds000030)

Code accompanying the manuscript *“Classification of schizophrenia from resting-state functional connectivity:
a confound-controlled reanalysis of the UCLA Consortium for Neuropsychiatric Phenomics dataset”* (under review).

The repository contains everything needed to reproduce the reported results from public data:

* **Data**: OpenNeuro [ds000030](https://openneuro.org/datasets/ds000030) (UCLA CNP; Poldrack et al., 2016,
  *Sci Data* 3:160110), CC0 licence. Preprocessed images are the public fMRIPrep 0.4.4 derivatives of release
  R1.0.5 (Gorgolewski, Durnez & Poldrack, 2017, *F1000Research* 6:1262).
* **Atlas**: Schaefer-2018, 100 parcels, 7 networks (Schaefer et al., 2018, *Cereb Cortex* 28:3095–3114).

## Workflow

| Step | Where | What | Time |
|---|---|---|---|
| `notebooks/01_extract_roi_timeseries_colab.ipynb` | Google Colab | Reads the preprocessed BOLD images, extracts **raw** parcel time series and a CSF signal, copies the confounds and phenotype files, and writes `roi_package_v2.zip` (13 MB). In this repository the package is split into `data/roi_package_v2_timeseries.zip` and `data/roi_package_v2_metadata.zip`, which the analysis script extracts automatically. Also checks that the time series of the first submission are reproduced exactly. | 45–90 min |
| `scripts/02_run_all_analyses.py` | Any computer | Denoising strategies, QC-FC benchmarks, nested repeated cross-validation with within-fold confound control, permutation test, bootstrap stability, Haufe activation patterns, univariate tests, tables and figures. | 30–90 min (8–16 cores) |

```bash
pip install -r requirements.txt
python scripts/02_run_all_analyses.py --package data/roi_package_v2 --out results
```

## Main analysis choices

* **Participants**: all participants with schizophrenia (SZ) or healthy controls (HC) with resting-state fMRI and
  fMRIPrep derivatives (SZ 50, HC 122). Primary motion exclusion: mean framewise displacement (FD) > 0.5 mm, or
  < 4 min of data after censoring.
* **Denoising (primary)**: 36 parameters (6 motion parameters, white matter, CSF and global signals, their temporal
  derivatives and quadratic terms; Satterthwaite et al., 2013) + censoring of volumes with FD > 0.5 mm (with one
  preceding and two following volumes; Power et al., 2014) + cosine high-pass filter (0.01 Hz). Six alternative
  strategies are compared with QC-FC benchmarks (Ciric et al., 2017).
* **Features**: Pearson correlation between the 100 parcels, Fisher z, 4,950 edges.
* **Classifier (primary)**: L2-penalised logistic regression; age, sex, scanner and mean FD regressed out of every
  edge **inside each training fold**; nested cross-validation (5 outer folds × 100 repetitions; inner 5-fold grid
  search of C); AUC with participant-level bootstrap 95% CI; sensitivity/specificity at the pre-specified 0.5
  threshold with Wilson CIs; 1,000-permutation test of the complete pipeline.
* **Comparators**: random forest, linear SVM, confound-only models, the original pipeline, and several sensitivity
  analyses (stringent motion criteria, propensity-matched subsample, single scanner, artefact exclusion).
* **Interpretation**: Haufe et al. (2014) activation patterns with 1,000 bootstrap refits (bootstrap ratio),
  network-level summaries, and edge-wise group comparisons adjusted for age, sex, scanner and mean FD.

## Corrections relative to the first submission

While preparing the revision, two coding problems were identified in the original analysis notebook and corrected:

1. **Confound selection.** The original code requested confound columns named as in fMRIPrep ≥ 1.2
   (`trans_x`, `csf`, `white_matter`, `global_signal`, `a_comp_cor_XX`). The fMRIPrep 0.4.4 files use different
   names (`X`, `RotX`, `WhiteMatter`, `GlobalSignal`, `aCompCor00`) and contain no CSF column, so the code fell back to
   the first 12 numeric columns (white matter, global signal, three DVARS variants, FD and six tCompCor
   components). Motion parameters were therefore not regressed. Strategy `P0_executed` reproduces that pipeline; the
   notebook's validation cell documents it.
2. **Parcel labels.** Recent nilearn versions prepend `"Background"` to the Schaefer labels; the original code did
   not remove it, so parcel names in the first submission were shifted by one parcel. Labels are now taken from the
   atlas look-up table and checked (`tableS7b_original_top12_labels.csv` lists the correct and the published names).

## Outputs (`results/`)

`tables/` (CSV): Table 1, Table 2, supplementary tables S1–S7, per-participant analysis table and out-of-fold scores.
`figures/` (PNG, PDF, 300-dpi TIFF): participant flow, classification performance, network-level patterns, QC-FC and
group connectivity matrices. `summary.json`: key numbers quoted in the manuscript.

## Licence

Code: MIT. Derived data in `data/` come from ds000030 (CC0).
