"""Final Bernoulli NMF fits (four profiles in dogs, two in cats) and profile shares by region and settlement type
(results/nmf_profiles.pkl).

Profile shares are estimated with the two-step model: the profiles are held fixed and membership is regressed on age
category and year with region, or with settlement type and coastal location. This treats each sample as belonging
to one profile, which the fitted Dirichlet concentrations (close to 0) support.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

import config as cf
import nmf


def best_chain(X, M, K):
    chains = []
    for ch in range(2):
        init = nmf.start(X, M, K, n_init=30, seed=4) if ch == 0 else None
        f = nmf.fit(X, M, np.zeros(len(X), int), K, row_init=init, seed=4000 + ch)
        chains.append(dict(H=f["H"], alpha=f["alpha"][0], loglik_mean=f["loglik_mean"]))
    return max(chains, key=lambda o: o["loglik_mean"]), [c["loglik_mean"] for c in chains]


def stability(species, H, K):
    """Cosine similarity of each profile fitted in each cross-validation fold with the matching final profile."""
    sims = []
    for f in range(5):
        fold = max((pd.read_pickle(cf.path("nmf_cv_jobs", f"{species}_{K}_{f}_{ch}.pkl")) for ch in range(2)),
                   key=lambda o: o["loglik_mean"])
        s = nmf.cosine(H, fold["H"])
        r, c = linear_sum_assignment(-s)
        sims.append(s[r, c])
    return np.array(sims)


def covariates(t):
    cols = {"Intercept": np.ones(len(t))}
    for a in ["0-6", "7-12", "49-120", ">120", "unknown"]:
        cols[f"age {a}"] = (t.agef == a).astype(float).values
    cols["year"] = t.yc.astype(float).values
    return cols


def location_shares(t, taxa, H, extra, levels, df):
    """Two-step shares for the levels of one location variable, LR test against age and year only, bootstrap CI."""
    X, M = t[taxa].fillna(0).values.astype(float), t[taxa].notna().values.astype(float)
    base = covariates(t)
    names = list(base) + list(extra)
    Z1 = np.column_stack(list(base.values()) + list(extra.values()))
    Z0 = np.column_stack(list(base.values()))
    Xa, Ma, Za, ca, _ = nmf.aggregate(X, M, Z1)
    m1 = nmf.twostep_fit(Xa, Ma, Za, H, c=ca)
    Xb, Mb, Zb, cb, _ = nmf.aggregate(X, M, Z0)
    m0 = nmf.twostep_fit(Xb, Mb, Zb, H, c=cb)
    lev = {levels[0]: None, **{l: names.index(l) for l in levels[1:]}}
    est = nmf.shares(Za, m1["B"], lev, w=ca)
    rng = np.random.default_rng(0)
    boot = {l: [] for l in levels}
    for _ in range(100):
        idx = rng.integers(0, len(t), len(t))
        Xc, Mc, Zc, cc, _ = nmf.aggregate(X[idx], M[idx], Z1[idx])
        mb = nmf.twostep_fit(Xc, Mc, Zc, H, c=cc, max_iter=300)
        for l, v in nmf.shares(Zc, mb["B"], lev, w=cc).items():
            boot[l].append(v)
    K = H.shape[0]
    lr = 2 * (m1["loglik"] - m0["loglik"])
    return dict(shares=est, shares_ci={l: np.percentile(np.array(v), [2.5, 97.5], axis=0) for l, v in boot.items()},
                p=float(chi2.sf(lr, df * (K - 1))), n=len(t))


def main():
    out = {}
    for sp in cf.SPECIES:
        taxa, K = cf.PROFILE_TAXA[sp], cf.N_PROFILES[sp]
        t = cf.load(sp)
        t = t[t[taxa].notna().any(axis=1)].reset_index(drop=True)
        X, M = t[taxa].fillna(0).values.astype(float), t[taxa].notna().values.astype(float)
        best, chain_ll = best_chain(X, M, K)
        W = nmf.weights(X, M, best["H"], best["alpha"])
        order = np.argsort(-W.mean(0))
        H, alpha, W = best["H"][order], best["alpha"][order], W[:, order]

        main_profile = W.argmax(1)
        n_taxa = t[taxa].sum(axis=1).values
        dominant = pd.DataFrame(dict(profile=range(K), n=np.bincount(main_profile, minlength=K),
                                     single_taxon=[int(((main_profile == k) & (n_taxa == 1)).sum()) for k in range(K)],
                                     mean_taxa=[n_taxa[main_profile == k].mean() for k in range(K)],
                                     min_taxa=[n_taxa[main_profile == k].min() for k in range(K)],
                                     max_taxa=[n_taxa[main_profile == k].max() for k in range(K)],
                                     age_0_6=[int(((main_profile == k) & (t.agef == "0-6")).sum()) for k in range(K)]))

        r = t[t.region.isin(cf.REGIONS)].reset_index(drop=True)
        region = location_shares(r, taxa, H, {g: (r.region == g).astype(float).values for g in cf.REGIONS[1:]},
                                 cf.REGIONS, len(cf.REGIONS) - 1)
        region["n_level"] = r.region.value_counts().reindex(cf.REGIONS).to_dict()
        s = r[r.settlement.notna()].reset_index(drop=True)
        extra = {g: (s.settlement == g).astype(float).values for g in cf.SETTLEMENTS[1:]}
        extra["coastal"] = (s.coast_km < 10).astype(float).values
        settle = location_shares(s, taxa, H, extra, cf.SETTLEMENTS, len(cf.SETTLEMENTS))
        settle["n_level"] = s.settlement.value_counts().reindex(cf.SETTLEMENTS).to_dict()

        out[sp] = dict(taxa=taxa, n=len(t), H=H, alpha=alpha, W=W, chain_loglik=chain_ll, stability=stability(sp, H, K),
                       positives_all=float(n_taxa[n_taxa > 0].mean()), dominant=dominant, region=region, settle=settle)
    cf.save(out, "nmf_profiles.pkl")


if __name__ == "__main__":
    main()
