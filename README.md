# Local reproducibility package for revised manuscript

This package is prepared for author review. It is not a public repository release,
does not assert permission for data sharing, and is not a substitute for author
approval of a journal data-availability statement.

## Data and biological scope

`analysis_band/head_confirmed_20261007/head_spectra_126.csv` contains exactly the
confirmed head spectra (`ID-h.spa`). It contains no head/leaf averages. Each column
in the original export was already the average of two measurements; no extra
averaging is applied here. Chemical units are mg/g DW. The supplied corrected
workbook is included to verify the seven-component sums.

`analysis_band/sample_labels.csv` preserves the original 88 extreme-group labels
and row order (44 low, 44 high). The sensitivity analysis restores all 14 samples
removed historically. It does not assign labels to the 38 middle-concentration
samples. The 74-sample labels remain in a separate file for provenance.

## New analysis entry point

Run from the extracted package root with Python and the recorded dependencies:

    python manuscript/sensitivity_88.py
    python manuscript/verify_sensitivity.py

The first script writes protocol, source hashes, labels, splits, inner candidate
scores, outer-fold metrics, sample-level predictions, summary and verification
under `manuscript/sensitivity_88`. Existing outputs are included. It validates
exact head filenames and concentration agreement before training. It evaluates
five baseline families and 17 fixed LDA intervals, plus inner-family and
inner-region selection. It does NOT repeat all 23 original routes.

The verification script replays the saved configurations of five models across
the 74-sample folds. All 1110 classifications matched; the maximum score difference
was below 6e-11. This is a regression check of implementation consistency, not
independent biological validation.

## Dependencies and historical code

The source modules import numpy, pandas, scipy, scikit-learn, openpyxl, xgboost,
lightgbm and pybaselines. Historical version information is provided in
`analysis_band/head_confirmed_20261007/all_algorithms/implementation_versions.json`.
In the working project these packages were available in `analysis_band/deps`.
An extracted package needs these packages installed in its own Python environment;
binary dependencies are not bundled. The package has been executed in the original
project environment, not tested as a clean installation on another machine.

Some shared source files retain historical mixed-tissue loaders for provenance.
Do not execute those historical entry points. The new `sensitivity_88.py` entry
point builds its input from the exact head-only CSV and does not call them.
The original mixed-tissue CSV is intentionally not required for this entry point.

The corrected 74-sample entry point is
`analysis_band/rerun_head_suite_20261007.py classical nir` (or `deep nir` / `tabpfn nir`).
Its protocol files and saved scores are included for audit. The TabPFN checkpoint
and torch binary packages are not bundled; the saved TabPFN protocol records the
checkpoint hash and software versions. Do not treat all-family reruns as verified
on a clean installation solely because the score audit is supplied.

## Interpretation and publication

The principal new process-level result is 68.4% mean outer-fold balanced accuracy
for inner selection among five families on the 88-sample extreme-group cohort.
This removes prediction-based exclusions but is retrospective, from the same
source batch, and is not an external or unselected-population validation.
Scores from the 74-sample cohort remain affected by outcome-based cohort selection.
Fold standard deviations are not confidence intervals. The 74- and 88-sample
partitions differ; score differences do not isolate a causal sample-removal effect.

Before submission, authors must confirm HPLC methods, specimen provenance,
author declarations and data-sharing rights; provide a stable repository/accession
or an accurate restricted-access procedure; and translate the saved protocol grids
into the chosen journal's supplementary presentation. Do not claim the files are
publicly available until such a release exists.
