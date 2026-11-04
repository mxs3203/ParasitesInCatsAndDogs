"""Prevalence, age standardisation and logistic models of year, age and region (results/descriptive.pkl)."""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2, chi2_contingency
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

import config as cf

warnings.filterwarnings("ignore")


def wilson(k, n):
    if n == 0:
        return np.nan, np.nan, np.nan
    lo, hi = proportion_confint(k, n, method="wilson")
    return 100 * k / n, 100 * lo, 100 * hi


def prevalence(t, cols, by=None):
    rows = []
    for g, part in ([(None, t)] if by is None else t.groupby(by)):
        for p in cols:
            x = part[p].dropna()
            e, lo, hi = wilson(x.sum(), len(x))
            rows.append(dict(group=g, parasite=p, k=int(x.sum()), n=len(x), prev=e, lo=lo, hi=hi))
    return pd.DataFrame(rows)


def standardised_by_year(t, known, cols):
    """Crude and directly age-standardised prevalence per year; the standard is the pooled age distribution."""
    weights = known.age_cat.value_counts(normalize=True).sort_index()
    rows = []
    for p in cols:
        for y, part in known.groupby("Year"):
            est, var, ok = 0.0, 0.0, True
            for a, w in weights.items():
                x = part.loc[part.age_cat == a, p].dropna()
                if len(x) == 0:
                    ok = False
                    break
                pa = x.mean()
                est += w * pa
                var += w ** 2 * max(pa * (1 - pa), 0.25 / len(x)) / len(x)
            x = t.loc[t.Year == y, p].dropna()
            crude, lo, hi = wilson(x.sum(), len(x))
            se = var ** 0.5
            rows.append(dict(parasite=p, year=y, crude=crude, crude_lo=lo, crude_hi=hi,
                             std=100 * est if ok else np.nan, std_lo=100 * (est - 1.96 * se) if ok else np.nan,
                             std_hi=100 * (est + 1.96 * se) if ok else np.nan))
    return pd.DataFrame(rows)


def odds(m, term):
    lo, hi = np.exp(m.conf_int().loc[term])
    return np.exp(m.params[term]), lo, hi


def year_age_region(t, cols, min_pos):
    """Logistic regression on year, age category and region (samples with a known region), with LR tests."""
    rows = []
    r = t[t.region.isin(cf.REGIONS)].copy()
    r["region"] = pd.Categorical(r.region, cf.REGIONS)
    age, reg = "C(agef, Treatment('13-48'))", "C(region, Treatment('Zagreb'))"
    for p in cols:
        d = r.dropna(subset=[p])
        if d[p].sum() < min_pos or d.Year.nunique() < 4:
            continue
        glm = lambda f: smf.glm(f, data=d, family=sm.families.Binomial()).fit()
        full = glm(f"Q('{p}') ~ yc + {age} + {reg}")
        no_region = glm(f"Q('{p}') ~ yc + {age}")
        no_age = glm(f"Q('{p}') ~ yc + I(agef == 'unknown') + {reg}")  # unknown age kept as its own level
        row = dict(parasite=p, positives=int(d[p].sum()), n=len(d),
                   p_region=float(chi2.sf(2 * (full.llf - no_region.llf), len(cf.REGIONS) - 1)),
                   p_age=float(chi2.sf(2 * (full.llf - no_age.llf), 4)), p_year=float(full.pvalues["yc"]))
        row["OR_year"], row["OR_year_lo"], row["OR_year_hi"] = odds(full, "yc")
        for g in cf.REGIONS[1:]:
            row[f"OR_{g}"], row[f"lo_{g}"], row[f"hi_{g}"] = odds(full, f"{reg}[T.{g}]")
        for a in ("0-6", "7-12", "49-120", ">120"):
            row[f"OR_age_{a}"], row[f"lo_age_{a}"], row[f"hi_age_{a}"] = odds(full, f"{age}[T.{a}]")
        rows.append(row)
    m = pd.DataFrame(rows)
    for c in ("p_region", "p_year", "p_age"):
        m[c.replace("p_", "q_")] = multipletests(m[c], method="fdr_bh")[1]
    return m


def main():
    out = {}
    for sp in cf.SPECIES:
        t = cf.load(sp)
        taxa = cf.TAXA[sp]
        cols = ["Any parasite"] + taxa
        known = t[t.age_cat.notna()]
        n_taxa = t[taxa].fillna(0).sum(axis=1)
        positive = t[t["Any parasite"] == 1].assign(n_taxa=n_taxa)
        multi = []
        for g, part in list(positive[positive.region.isin(cf.REGIONS)].groupby("region")) + [("All", positive)]:
            e, lo, hi = wilson((part.n_taxa >= 2).sum(), len(part))
            multi.append(dict(region=g, n_pos=len(part), multi=e, lo=lo, hi=hi))
        g, c = t.Giardia, t.Cryptosporidium
        out[sp] = dict(
            prev=prevalence(t, cols).drop(columns="group"),
            prev_year=prevalence(t, cols, "Year"),
            prev_age=prevalence(known, cols, "age_cat"),
            prev_region=prevalence(t[t.region.isin(cf.REGIONS)], cols, "region"),
            age_mix_chi2=chi2_contingency(pd.crosstab(known.Year, known.age_cat))[:3],
            std_year=standardised_by_year(t, known, cols),
            region_models=year_age_region(t, cols, 40 if sp == "dog" else 20),
            multi_region=pd.DataFrame(multi),
            crypto_giardia=dict(
                total=int(c.sum()), with_g=int(((c == 1) & (g == 1)).sum()),
                p_c_given_gpos=float(c[g == 1].mean()), p_c_given_gneg=float(c[g == 0].mean()),
                by_year=t.groupby("Year").apply(lambda y: pd.Series({
                    "gpos": 100 * y.loc[y.Giardia == 1, "Cryptosporidium"].mean(),
                    "gneg": 100 * y.loc[y.Giardia == 0, "Cryptosporidium"].mean()}))))
    cf.save(out, "descriptive.pkl")


if __name__ == "__main__":
    main()
