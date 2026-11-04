"""Settlement type and coastal location: logistic models and prevalence by settlement type (results/location.pkl)."""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

import config as cf

warnings.filterwarnings("ignore")


def lr(m1, m0):
    df = int(round(m1.df_model - m0.df_model))
    return float(chi2.sf(2 * (m1.llf - m0.llf), df))


def odds(m, term):
    lo, hi = np.exp(m.conf_int().loc[term])
    return np.exp(m.params[term]), lo, hi


def models(r, taxa, floor):
    settle = "C(settlement, Treatment('Zagreb'))"
    other = " + ".join(f"Q('s_{s}')" for s in cf.SETTLEMENTS[2:])  # non-Zagreb contrasts, reference cities of 40,000 or more
    rows = []
    for p in taxa:
        d = r.dropna(subset=[p])
        if d[p].sum() < floor:
            continue
        glm = lambda f: smf.glm(f, data=d, family=sm.families.Binomial()).fit()
        base = f"Q('{p}') ~ yc + C(agef, Treatment('13-48'))"
        m0, mr, ms = glm(base), glm(base + " + C(region, Treatment('Zagreb'))"), glm(base + f" + {settle}")
        msc = glm(base + f" + {settle} + coastal")
        mrs = glm(base + " + C(region, Treatment('Zagreb')) + " + other)
        step = glm(base + " + rur + coastal")
        row = dict(parasite=p, positives=int(d[p].sum()), n=len(d), p_settle=lr(ms, m0),
                   p_settle_given_region=lr(mrs, mr), p_region_given_settle=lr(mrs, ms), p_coast_given_settle=lr(msc, ms),
                   p_rural_step=step.pvalues["rur"])
        row["OR_coast_s"], row["lo_coast_s"], row["hi_coast_s"] = odds(msc, "coastal")
        row["OR_rural_step"], row["lo_rural_step"], row["hi_rural_step"] = odds(step, "rur")
        rows.append(row)
    s = pd.DataFrame(rows)
    for c in ("p_settle", "p_settle_given_region", "p_rural_step", "p_coast_given_settle", "p_region_given_settle"):
        s[c.replace("p_", "q_")] = multipletests(s[c], method="fdr_bh")[1]
    return s


def prevalence(r, taxa):
    rows = []
    for p in taxa:
        for s, g in r.groupby("settlement"):
            x = g[p].dropna()
            if len(x):
                lo, hi = proportion_confint(x.sum(), len(x), method="wilson")
                rows.append(dict(parasite=p, settlement=s, k=int(x.sum()), n=len(x), prev=100 * x.mean(), lo=100 * lo, hi=100 * hi))
    return pd.DataFrame(rows)


def main():
    out = {}
    for sp in cf.SPECIES:
        t = cf.load(sp)
        taxa = cf.modelled_taxa(t, sp)
        r = t[t.settlement.notna() & t.region.isin(cf.REGIONS)].copy()
        r["settlement"] = pd.Categorical(r.settlement, cf.SETTLEMENTS)
        r["region"] = pd.Categorical(r.region, cf.REGIONS)
        for s in cf.SETTLEMENTS[2:]:
            r[f"s_{s}"] = (r.settlement == s).astype(float)
        out[sp] = dict(settle_models=models(r, taxa, 40 if sp == "dog" else 20), settle_prev=prevalence(r, taxa))
    cf.save(out, "location.pkl")


if __name__ == "__main__":
    main()
