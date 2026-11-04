"""Number of Bernoulli NMF profiles by five-fold cross-validation (results/nmf_cv.pkl).

Each observed result of a held-out sample is predicted from that sample's other results, with the model fitted to
the other folds; the summed log-likelihood is compared with the independence model (each taxon its own prevalence).
Two chains per fold, one started from a Bernoulli mixture and one at random; the chain with the higher mean log
joint probability is used. Fits are cached in results/nmf_cv_jobs/.
"""
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import config as cf
import nmf

KS = range(2, 6)
N_FOLDS, N_CHAINS = 5, 2
JOBS = cf.path("nmf_cv_jobs")


def data(species):
    t = pd.read_pickle(cf.path(f"{species}s.pkl"))
    x = t[cf.PROFILE_TAXA[species]]
    x = x[x.notna().any(axis=1)]
    return x.fillna(0).values.astype(float), x.notna().values.astype(float)


def folds(n):
    return np.random.default_rng(0).integers(0, N_FOLDS, n)


def job(args):
    species, K, f, chain = args
    fn = f"{JOBS}/{species}_{K}_{f}_{chain}.pkl"
    if os.path.exists(fn):
        return args, pd.read_pickle(fn)
    t0 = time.time()
    X, M = data(species)
    train = folds(len(X)) != f
    init = nmf.start(X[train], M[train], K, n_init=20, seed=10 + f) if chain == 0 else None
    fit = nmf.fit(X[train], M[train], np.zeros(int(train.sum()), int), K, row_init=init, seed=1000 + 10 * f + chain)
    test = ~train
    P = nmf.predictive_loo(X[test], M[test], np.zeros(int(test.sum()), int), fit["H"], fit["alpha"], S=50_000, seed=K + f)
    out = dict(heldout=nmf.loglik(P, X[test], M[test]), loglik_mean=fit["loglik_mean"], conc=float(fit["alpha"].sum()), H=fit["H"])
    pd.to_pickle(out, fn)
    print(f"{species} K={K} fold {f} chain {chain}: {time.time() - t0:.0f} s", flush=True)
    return args, out


def independence(species):
    """Held-out log-likelihood per fold when every result is predicted by the taxon's prevalence in the other folds."""
    X, M = data(species)
    f_ = folds(len(X))
    ll = np.zeros(N_FOLDS)
    for f in range(N_FOLDS):
        train, test = f_ != f, f_ == f
        h = np.clip((M[train] * X[train]).sum(0) / np.clip(M[train].sum(0), nmf.EPS, None), nmf.EPS, 1 - nmf.EPS)
        P = np.clip(np.broadcast_to(h, X[test].shape), nmf.EPS, 1 - nmf.EPS)
        ll[f] = nmf.loglik(P, X[test], M[test])
    return ll, float(M.sum())


def main():
    os.makedirs(JOBS, exist_ok=True)
    jobs = [(sp, K, f, ch) for sp in cf.SPECIES for K in KS for f in range(N_FOLDS) for ch in range(N_CHAINS)]
    with Pool(int(os.environ.get("NMF_PROCS", 2)), maxtasksperchild=1) as pool:
        res = pool.map(job, jobs, chunksize=1)
    out = {}
    for sp in cf.SPECIES:
        base, n_results = independence(sp)
        cv, conc = {}, {}
        for K in KS:
            best = [max((o for (s, k, f_, _), o in res if (s, k, f_) == (sp, K, f)), key=lambda o: o["loglik_mean"])
                    for f in range(N_FOLDS)]
            cv[K] = np.array([o["heldout"] for o in best])
            conc[K] = np.array([o["conc"] for o in best])
        out[sp] = dict(n_results=n_results, independence=base, cv=cv, conc=conc,
                       gain={K: float(v.sum() - base.sum()) for K, v in cv.items()})
    cf.save(out, "nmf_cv.pkl")


if __name__ == "__main__":
    main()
