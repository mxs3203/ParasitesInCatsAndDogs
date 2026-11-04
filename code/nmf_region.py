"""Shared against region-specific Dirichlet priors for Bernoulli NMF, five-fold cross-validation in samples with a
known region (results/nmf_region.pkl). With region-specific priors each region has its own profile shares and degree
of mixing, while the profiles are shared. Fits are cached in results/nmf_region_jobs/."""
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import config as cf
import nmf

JOBS = cf.path("nmf_region_jobs")


def data(species):
    t = pd.read_pickle(cf.path(f"{species}s.pkl"))
    taxa = cf.PROFILE_TAXA[species]
    t = t[t[taxa].notna().any(axis=1) & t.region.isin(cf.REGIONS)].reset_index(drop=True)
    x = t[taxa]
    return x.fillna(0).values.astype(float), x.notna().values.astype(float), t.region.map(cf.REGIONS.index).values


def job(args):
    species, prior, fold = args
    K = cf.N_PROFILES[species]
    fn = f"{JOBS}/{species}_k{K}_{prior}_{fold}.pkl"
    if os.path.exists(fn):
        return args, pd.read_pickle(fn)
    t0 = time.time()
    X, M, region = data(species)
    folds = np.random.default_rng(0).integers(0, 5, len(X))
    train, test = folds != fold, folds == fold
    grp = region if prior == "region" else np.zeros(len(X), int)
    f = nmf.fit(X[train], M[train], grp[train], K, row_init=nmf.start(X[train], M[train], K, n_init=20, seed=10),
                seed=100 + 10 * fold)
    P = nmf.predictive_loo(X[test], M[test], grp[test], f["H"], f["alpha"], S=50_000, seed=fold)
    out = dict(heldout=nmf.loglik(P, X[test], M[test]), n_heldout=float(M[test].sum()), alpha=f["alpha"], H=f["H"])
    pd.to_pickle(out, fn)
    print(f"{species} {prior} prior, fold {fold}: {time.time() - t0:.0f} s", flush=True)
    return args, out


def main():
    os.makedirs(JOBS, exist_ok=True)
    jobs = [(sp, prior, f) for sp in cf.SPECIES for prior in ("shared", "region") for f in range(5)]
    with Pool(int(os.environ.get("NMF_PROCS", 2)), maxtasksperchild=1) as pool:
        res = dict(pool.map(job, jobs, chunksize=1))
    out = {}
    for sp in cf.SPECIES:
        o = {prior: np.array([res[(sp, prior, f)]["heldout"] for f in range(5)]) for prior in ("shared", "region")}
        o["diff"] = o["region"] - o["shared"]
        o["n_heldout"] = sum(res[(sp, "shared", f)]["n_heldout"] for f in range(5))
        out[sp] = o
    cf.save(out, "nmf_region.pkl")


if __name__ == "__main__":
    main()
