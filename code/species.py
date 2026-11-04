"""Dogs compared with cats for the genera recorded in both: joint logistic models with species interactions
(results/species.pkl)."""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

import config as cf

warnings.filterwarnings("ignore")
GENERA = ["Giardia", "Toxocara", "Cryptosporidium"]


def glm(formula, data):
    return smf.glm(formula, data=data, family=sm.families.Binomial()).fit()


def lr(m1, m0):
    df = int(round(m1.df_model - m0.df_model))
    return float(chi2.sf(2 * (m1.llf - m0.llf), df))


def combined(m, names):
    """OR and 95% CI of a sum of coefficients."""
    b, v = m.params, m.cov_params()
    est = sum(b[n] for n in names)
    se = np.sqrt(sum(v.loc[i, j] for i in names for j in names))
    return dict(OR=np.exp(est), lo=np.exp(est - 1.96 * se), hi=np.exp(est + 1.96 * se))


def main():
    j = pd.concat([cf.load("dog").assign(species="dog"), cf.load("cat").assign(species="cat")], ignore_index=True)
    j["species"] = pd.Categorical(j.species, ["dog", "cat"])
    j["agef"] = pd.Categorical(j.agef, ["13-48", "0-6", "7-12", "49-120", ">120", "unknown"])
    j["regionf"] = pd.Categorical(j.regionf, cf.REGIONS + ["unknown"])
    rows, effects = [], {}
    for p in GENERA:
        d = j.dropna(subset=[p])
        base = f"Q('{p}') ~ species + agef + yc + regionf"
        add, by_age, by_year = glm(base, d), glm(base + " + species:agef", d), glm(base + " + species:yc", d)
        lo, hi = np.exp(add.conf_int().loc["species[T.cat]"])
        rows.append(dict(parasite=p, n_dog=int((d.species == "dog").sum()), n_cat=int((d.species == "cat").sum()),
                         OR_cat_add=np.exp(add.params["species[T.cat]"]), lo_cat_add=lo, hi_cat_add=hi,
                         p_age=lr(by_age, add), p_year=lr(by_year, add), p_region=lr(glm(base + " + species:regionf", d), add)))
        age = [dict(age=a, **combined(by_age, ["species[T.cat]"] + ([f"species[T.cat]:agef[T.{a}]"] if a != "13-48" else [])))
               for a in cf.AGES]
        trend = [dict(species="dog", **combined(by_year, ["yc"])), dict(species="cat", **combined(by_year, ["yc", "species[T.cat]:yc"]))]
        effects[p] = dict(age=pd.DataFrame(age), trend=pd.DataFrame(trend))
    m = pd.DataFrame(rows)
    q = multipletests(m[["p_age", "p_year", "p_region"]].values.ravel(), method="fdr_bh")[1].reshape(len(m), 3)
    m["q_age"], m["q_year"], m["q_region"] = q[:, 0], q[:, 1], q[:, 2]
    cf.save(dict(models=m, effects=effects), "species.pkl")


if __name__ == "__main__":
    main()
