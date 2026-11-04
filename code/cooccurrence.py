"""Pairwise co-occurrence: crude odds ratios and Mantel-Haenszel odds ratios over age x year x region strata
(results/cooccurrence.pkl)."""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multitest import multipletests

import config as cf

warnings.filterwarnings("ignore")


def pair(x, a, b):
    t = pd.crosstab(x[a], x[b]).reindex(index=[0, 1], columns=[0, 1], fill_value=0).values
    crude = ((t[1, 1] + .5) * (t[0, 0] + .5)) / ((t[1, 0] + .5) * (t[0, 1] + .5))
    code = (2 * x[a] + x[b]).astype(int)
    counts = code.groupby(x.stratum).value_counts().unstack(fill_value=0).reindex(columns=[0, 1, 2, 3], fill_value=0)
    tables = [np.array([[r[0], r[1]], [r[2], r[3]]]) for r in counts.values.astype(float)]
    tables = [s for s in tables if s.sum(0).min() > 0 and s.sum(1).min() > 0]  # strata with an empty row or column carry no information
    row = dict(a=a, b=b, both=int(t[1, 1]), OR_crude=crude, p_crude=fisher_exact(t)[1], n_strata=len(tables))
    try:
        st = StratifiedTable(tables)
        row.update(OR_MH=st.oddsratio_pooled, p_MH=st.test_null_odds(correction=True).pvalue, p_homog=st.test_equal_odds().pvalue)
        row["lo"], row["hi"] = st.oddsratio_pooled_confint()
    except Exception:
        row.update(OR_MH=np.nan, lo=np.nan, hi=np.nan, p_MH=np.nan, p_homog=np.nan)
    return row


def main():
    out = {}
    for sp in cf.SPECIES:
        t = cf.load(sp)
        floor = 50 if sp == "dog" else 20
        taxa = [p for p in cf.TAXA[sp] if t[p].sum() >= floor and t[p].notna().mean() > 0.9]
        stratum = t.agef + "|" + t.Year.astype(str) + "|" + t.regionf
        rows = [pair(t[[a, b]].assign(stratum=stratum).dropna(), a, b) for i, a in enumerate(taxa) for b in taxa[i + 1:]]
        c = pd.DataFrame(rows)
        c["q_crude"] = multipletests(c.p_crude, method="fdr_bh")[1]
        ok = c.p_MH.notna()
        c.loc[ok, "q_MH"] = multipletests(c.loc[ok, "p_MH"], method="fdr_bh")[1]
        out[sp] = dict(cooc=c, taxa=taxa)
    cf.save(out, "cooccurrence.pkl")


if __name__ == "__main__":
    main()
