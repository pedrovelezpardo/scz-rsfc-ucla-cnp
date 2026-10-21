# Data

The package `roi_package_v2` (split into `roi_package_v2_timeseries.zip` and `roi_package_v2_metadata.zip` to keep each file below GitHub's web-upload limit) is produced by `notebooks/01_extract_roi_timeseries_colab.ipynb`. It contains only derived,
non-identifiable data from the public OpenNeuro dataset ds000030 (released under CC0):

| Path | Content |
|---|---|
| `roi_raw/sub-*_schaefer100_raw.npy` | Raw (uncleaned) mean BOLD signal of the 100 Schaefer-2018 parcels, volumes × parcels |
| `csf/sub-*_csf_raw.npy` | Mean CSF signal (subject CSF probability ≥ 0.95, eroded once) |
| `confounds/sub-*_task-rest_bold_confounds.tsv` | fMRIPrep 0.4.4 confound time series (unchanged copies) |
| `sidecars/` | BIDS sidecars of the resting-state runs |
| `phenotype/` | Phenotype files used for descriptive statistics (copied from ds000030) |
| `participants.tsv` | ds000030 participants file |
| `participants_original_manual_cohort.tsv` | The 99 participants analysed in the first submission |
| `atlas_schaefer2018_100p_7n.*` | Atlas image, parcel labels, networks and centroids |
| `manifest_extraction.csv`, `logs/` | Extraction log, software versions and validation against the first submission |

Keep both zips in this folder and run `python scripts/02_run_all_analyses.py`; they are extracted to `data/roi_package_v2/` on first use.
