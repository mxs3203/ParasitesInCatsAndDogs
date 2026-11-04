#!/usr/bin/env bash
# Full analysis, in order. Run from anywhere: bash code/run_all.sh
# NMF_PROCS sets the number of processes for the Bernoulli NMF fits (default 2). The fits are cached in
# results/nmf_cv_jobs and results/nmf_region_jobs; delete both folders after any change to the data.
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python; [ -x "$PY" ] || PY=python3

$PY prepare.py        # cleaned samples: results/dogs.pkl, results/cats.pkl
$PY descriptive.py    # prevalence, age standardisation, year, age and region models
$PY location.py       # settlement type and coastal location
$PY cooccurrence.py   # pairwise co-occurrence
$PY age.py            # age curves and trends within age categories
$PY species.py        # dogs compared with cats
$PY nmf_cv.py         # number of NMF profiles (slow on a fresh run)
$PY nmf_profiles.py   # final profiles and their shares by location
$PY nmf_region.py     # shared against region-specific priors
$PY figures.py        # manuscript/figures
$PY tables.py         # manuscript/tables
