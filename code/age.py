"""Prevalence by age (natural splines) and yearly trends within age categories (results/age.pkl)."""
import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

import config as cf

warnings.filterwarnings("ignore")
GRID = np.unique(np.r_[np.arange(0, 25, 1), np.arange(26, 61, 2), np.arange(66, 241, 6)])  # months


def glm(formula, data):
    return smf.glm(formula, data=data, family=sm.families.Binomial()).fit()


def lr(m1, m0):
    df = int(round(m1.df_model - m0.df_model))
    return float(chi2.sf(2 * (m1.llf - m0.llf), df))


def curve(d, p, rng):
    """Prevalence by age, adjusted for year and region and averaged over the samples, with a simulation CI."""
    # Years and regions without a single positive are left out: complete separation makes their coefficients unbounded.
    d = d[d.groupby("Year")[p].transform("sum").gt(0) & d.groupby("regionf")[p].transform("sum").gt(0)]
    df = 4 if d[p].sum() >= 150 else 3
    y, X = patsy.dmatrices(f"Q('{p}') ~ cr(la, df={df}, constraints='center') + C(Year) + C(regionf)", d, return_type="matrix")
    m = sm.GLM(np.asarray(y).ravel(), np.asarray(X), family=sm.families.Binomial()).fit()
    draws = rng.multivariate_normal(np.asarray(m.params), np.asarray(m.cov_params()), 300)
    sub = d.sample(min(len(d), 3000), random_state=0)
    est, lo, hi = [], [], []
    for a in GRID:
        Xa = np.asarray(patsy.dmatrix(X.design_info, sub.assign(la=np.log1p(a)), return_type="matrix"))
        est.append(100 * (1 / (1 + np.exp(-(Xa @ np.asarray(m.params))))).mean())
        sim = 100 * (1 / (1 + np.exp(-(Xa @ draws.T)))).mean(0)
        lo.append(np.percentile(sim, 2.5))
        hi.append(np.percentile(sim, 97.5))
    return pd.DataFrame(dict(age=GRID, prev=est, lo=lo, hi=hi))


def age_by_year(d, t, p):
    """Three tests of whether the yearly trend differs by age, and the OR per year within each age category."""
    spline = "cr(la, df=4, constraints='center')"
    row = dict(parasite=p,
               p_spline_x_year=lr(glm(f"Q('{p}') ~ {spline} * yc + C(regionf)", d), glm(f"Q('{p}') ~ {spline} + yc + C(regionf)", d)))
    dd = t.dropna(subset=[p])
    row["p_agegrp_x_year"] = lr(glm(f"Q('{p}') ~ C(agef) * C(Year) + C(regionf)", dd), glm(f"Q('{p}') ~ C(agef) + C(Year) + C(regionf)", dd))
    slopes = glm(f"Q('{p}') ~ C(agef) + C(agef):yc + C(regionf)", dd)
    row["p_slopes"] = lr(slopes, glm(f"Q('{p}') ~ C(agef) + yc + C(regionf)", dd))
    ci = slopes.conf_int()
    for a in cf.AGES:
        k = f"C(agef)[{a}]:yc"
        row[f"ORyr_{a}"], row[f"lo_{a}"], row[f"hi_{a}"] = np.exp(slopes.params[k]), np.exp(ci.loc[k, 0]), np.exp(ci.loc[k, 1])
    return row


def main():
    out = {}
    for sp in cf.SPECIES:
        t = cf.load(sp)
        known = t[t.age_m.notna()]
        rng = np.random.default_rng(0)
        curves, rows = {}, []
        for p in cf.modelled_taxa(t, sp):
            d = known.dropna(subset=[p])
            if d[p].sum() < (40 if sp == "dog" else 20):
                continue
            curves[p] = curve(d, p, rng)
            rows.append(age_by_year(d, t, p))
        ay = pd.DataFrame(rows)
        for c in ("p_spline_x_year", "p_agegrp_x_year", "p_slopes"):
            ay[c.replace("p_", "q_")] = multipletests(ay[c], method="fdr_bh")[1]
        out[sp] = dict(age_curves=curves, age_year=ay)
    cf.save(out, "age.pkl")


if __name__ == "__main__":
    main()
