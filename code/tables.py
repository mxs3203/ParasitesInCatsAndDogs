"""Supplementary Tables S1-S6 (manuscript/tables/SupplementaryTables.xlsx, one sheet each) and the study data
(manuscript/tables/data.xlsx).

The data file leaves out owner and pet names, the sending clinic, registration numbers, receipt dates and raw-file
row references.
"""
import os

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.cell.rich_text import CellRichText, TextBlock
from openpyxl.cell.text import InlineFont
from openpyxl.styles import Alignment, Border, Font, Side
from openpyxl.utils import get_column_letter

import config as cf

FONT, SIZE = "Times New Roman", 12
F, FB = Font(name=FONT, size=SIZE), Font(name=FONT, size=SIZE, bold=True)
IT, UP = InlineFont(rFont=FONT, sz=SIZE, i=True), InlineFont(rFont=FONT, sz=SIZE)
LINE = Side(style="thin", color="000000")
SPECIES = (("dog", "Dogs"), ("cat", "Cats"))
REGION_NAMES = {"Zagreb": "City of Zagreb", "North": "North Croatia", "Pannonian": "Pannonian Croatia",
                "Dalmatia": "Dalmatia", "Istria": "Istria, Kvarner and Lika"}
SETTLEMENT_NAMES = {"Zagreb": "City of Zagreb", "City >= 40k": "City, 40,000 or more", "Town 10-40k": "Town, 10,000-39,999",
                    "Town < 10k": "Town, under 10,000", "Rural municipality": "Rural municipality"}


def taxon(p):
    """Taxon as rich text: names in italics; 'sp.', 'Any parasite' and 'Hookworms' upright."""
    if p == "Any parasite":
        return p
    if p in ("Ancylostoma/Uncinaria", "Ancylostoma"):
        return CellRichText(TextBlock(UP, "Hookworms ("), TextBlock(IT, p), TextBlock(UP, ")"))
    if p.endswith(" sp."):
        return CellRichText(TextBlock(IT, p[:-4]), TextBlock(UP, " sp."))
    return CellRichText(TextBlock(IT, p))


def pair(a, b):
    def runs(p):
        if p in ("Ancylostoma/Uncinaria", "Ancylostoma"):
            return [TextBlock(UP, "hookworms")]
        if p.endswith(" sp."):
            return [TextBlock(IT, p[:-4]), TextBlock(UP, " sp.")]
        return [TextBlock(IT, p)]

    r = runs(a) + [TextBlock(UP, " and ")] + runs(b)
    if r[0].text == "hookworms":
        r[0] = TextBlock(UP, "Hookworms")
    return CellRichText(*r)


def q(v):
    return "< 0.001" if v < 0.001 else f"{v:.3f}"


def ci(o, lo, hi):
    return f"{o:.2f} ({lo:.2f}-{hi:.2f})"


def short_or(v):
    return f"{v:.0f}" if v >= 100 else (f"{v:.1f}" if v >= 10 else f"{v:.2f}")


def share(src, level, k):
    e, (lo, hi) = 100 * src["shares"][level][k], 100 * src["shares_ci"][level][:, k]
    return f"{e:.1f} ({lo:.1f}-{hi:.1f})"


def sheet(wb, name, head, blocks, widths):
    """One table: a bold label row per block, lines above and below the header and under the last row."""
    ws = wb.create_sheet(name)
    ws.append(head)
    for label, rows in blocks:
        if label:
            ws.append([label])
            ws.cell(ws.max_row, 1).font = FB
        for r in rows:
            ws.append(r)
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row):
        for c in row:
            if c.font != FB:
                c.font = FB if c.row == 1 else F
            c.alignment = Alignment(horizontal="left" if c.column == 1 else "center", vertical="center", wrap_text=c.row == 1)
            if isinstance(c.value, float):
                c.number_format = "0.0"
    for c in ws[1]:
        c.border = Border(top=LINE, bottom=LINE)
    for c in ws[ws.max_row]:
        c.border = Border(bottom=LINE)
    for i, w in enumerate(widths):
        ws.column_dimensions[get_column_letter(i + 1)].width = w


def trends(wb, d):
    blocks = []
    for sp, label in SPECIES:
        py = d[sp]["prev_year"].set_index(["parasite", "group"])["prev"]
        rows = []
        for _, r in d[sp]["region_models"].iterrows():
            a, b = py.get((r.parasite, 2016)), py.get((r.parasite, 2022))
            rows.append([taxon(r.parasite), "NR" if pd.isna(a) else round(float(a), 1), "NR" if pd.isna(b) else round(float(b), 1),
                         ci(r.OR_year, r.OR_year_lo, r.OR_year_hi), q(r.q_year)])
        blocks.append((label, rows))
    sheet(wb, "SupTable1", ["Taxon", "2016 (%)", "2022 (%)", "OR per year (95% CI)", "q"], blocks, (34, 11, 11, 24, 10))


def location(wb, d, loc):
    blocks = []
    for sp, label in SPECIES:
        m = d[sp]["region_models"].set_index("parasite")
        s = loc[sp]["settle_models"].set_index("parasite")
        blocks.append((label, [[taxon(p), q(m.loc[p].q_region), q(s.loc[p].q_settle),
                                ci(s.loc[p].OR_rural_step, s.loc[p].lo_rural_step, s.loc[p].hi_rural_step), q(s.loc[p].q_rural_step),
                                ci(s.loc[p].OR_coast_s, s.loc[p].lo_coast_s, s.loc[p].hi_coast_s), q(s.loc[p].q_coast_given_settle)]
                               for p in m.index]))
    sheet(wb, "SupTable2", ["Taxon", "Region, q", "Settlement type, q", "OR per step, city to countryside (95% CI)", "Trend, q",
                            "Coastal vs inland, OR (95% CI)", "Coast, q"], blocks, (34, 11, 12, 22, 10, 20, 10))


def cooccurrence(wb, cooc):
    blocks = []
    for sp, label in SPECIES:
        c = cooc[sp]["cooc"]
        c = c[c.q_MH < 0.05].sort_values("OR_MH", ascending=False)
        blocks.append((label, [[pair(r.a, r.b), int(r.both), short_or(r.OR_crude),
                                f"{short_or(r.OR_MH)} ({short_or(r.lo)}-{short_or(r.hi)})", q(r.q_MH)] for _, r in c.iterrows()]))
    sheet(wb, "SupTable3", ["Pair", "Both positive (n)", "Crude OR", "Adjusted OR (95% CI)", "q"], blocks, (44, 12, 11, 24, 10))


def profiles(wb, nmf):
    # Profile order: dogs 0 low prevalence, 1 helminths and coccidia, 2 Giardia and Cryptosporidium, 3 polyparasitism;
    # cats 0 low prevalence, 1 Giardia and Cryptosporidium.
    blocks = []
    for kind, names, label in (("region", REGION_NAMES, "Region"), ("settle", SETTLEMENT_NAMES, "Settlement type")):
        d, c = nmf["dog"][kind], nmf["cat"][kind]
        rows = [[names[x], int(d["n_level"][x]), share(d, x, 2), share(d, x, 1), share(d, x, 3), int(c["n_level"][x]), share(c, x, 1)]
                for x in names]
        rows.append(["Likelihood-ratio test, P", "", q(d["p"]), "", "", "", q(c["p"])])
        blocks.append((label, rows))
    sheet(wb, "SupTable4", ["Location", "Dogs, n", "Dogs: Giardia and Cryptosporidium profile, % (95% CI)",
                            "Dogs: helminths and coccidia profile, % (95% CI)", "Dogs: polyparasitism profile, % (95% CI)",
                            "Cats, n", "Cats: Giardia and Cryptosporidium profile, % (95% CI)"], blocks, (28, 9, 22, 22, 22, 9, 22))


def age(wb, d, age_res):
    blocks = []
    for sp, label in SPECIES:
        pa = d[sp]["prev_age"].assign(age=lambda t: t.group.map(lambda g: cf.AGES[int(g) - 1])).set_index(["parasite", "age"])["prev"]
        ay = age_res[sp]["age_year"].set_index("parasite")["q_slopes"]
        blocks.append((label, [[taxon(r.parasite)] + [round(float(pa[(r.parasite, a)]), 1) for a in cf.AGES]
                               + [ci(r["OR_age_0-6"], r["lo_age_0-6"], r["hi_age_0-6"]), q(r.q_age),
                                  q(ay[r.parasite]) if r.parasite in ay.index else "NA"]
                               for _, r in d[sp]["region_models"].iterrows()]))
    sheet(wb, "SupTable5", ["Taxon", "0-6 months (%)", "7-12 months (%)", "13-48 months (%)", "49-120 months (%)",
                            "> 120 months (%)", "OR 0-6 vs 13-48 months (95% CI)", "Age, q", "Age × year, q"],
          blocks, (34, 10, 10, 10, 10, 10, 22, 9, 11))


def dogs_and_cats(wb, d, spec):
    models = spec["models"].set_index("parasite")
    prev = {sp: d[sp]["prev"].set_index("parasite") for sp in cf.SPECIES}
    rows = []
    for p in ["Giardia", "Toxocara", "Cryptosporidium"]:
        r, tr = models.loc[p], spec["effects"][p]["trend"].set_index("species")
        pdg, pct = prev["dog"].loc[p], prev["cat"].loc[p]
        rows.append([taxon(p), f"{pdg.prev:.1f} ({pdg.lo:.1f}-{pdg.hi:.1f})", f"{pct.prev:.1f} ({pct.lo:.1f}-{pct.hi:.1f})",
                     ci(r.OR_cat_add, r.lo_cat_add, r.hi_cat_add), ci(*tr.loc["dog", ["OR", "lo", "hi"]]),
                     ci(*tr.loc["cat", ["OR", "lo", "hi"]]), q(r.q_age), q(r.q_year), q(r.q_region)])
    sheet(wb, "SupTable6", ["Genus", "Dogs, prevalence % (95% CI)", "Cats, prevalence % (95% CI)", "Cats vs dogs, OR (95% CI)",
                            "Dogs, OR per year (95% CI)", "Cats, OR per year (95% CI)", "Species × age, q", "Species × year, q",
                            "Species × region, q"], [("", rows)], (20, 16, 16, 18, 18, 18, 11, 11, 11))


VARIABLES = [
    ("Sample", "Study sample number (D = dog, C = cat)"), ("Year", "Year of examination"),
    ("Sex", "Male or female; blank, not recorded"), ("Age (months)", "Age in months; blank, not recorded"),
    ("Age category (months)", "0-6, 7-12, 13-48, 49-120, > 120"), ("Town", "Town of origin"),
    ("Municipality", "Town or municipality (geoBoundaries ADM2)"), ("County", "County"),
    ("Region", "Study region (five regions; blank, location not recorded)"),
    ("Settlement type", "City of Zagreb; city, 40,000 or more; town, 10,000-39,999; town, under 10,000; rural municipality"),
    ("Distance to the sea (km)", "Distance from the town centre to the coastline"),
    ("Coastal", "Yes, within 10 km of the sea"), ("Latitude", "Town centre"), ("Longitude", "Town centre"),
    ("Follow-up sample", "Yes, marked as a follow-up (control) examination"),
    ("Repeated registration number", "Yes, the registration number occurs more than once in the year"),
    ("Parasite taxa (one column each)", "1, detected; 0, not detected; blank, not examined")]


def data_file():
    age_label = {1: "0-6", 2: "7-12", 3: "13-48", 4: "49-120", 5: "> 120"}
    yes_no = lambda s: s.map(lambda v: "yes" if v == 1 or v is True else ("no" if pd.notna(v) else None))
    wb = Workbook()
    wb.remove(wb.active)
    for sp, label in SPECIES:
        t = pd.read_pickle(cf.path(f"{sp}s.pkl"))
        out = pd.DataFrame({
            "Sample": [f"{'D' if sp == 'dog' else 'C'}{i + 1:05d}" for i in range(len(t))],
            "Year": t.Year.astype(int).values,
            "Sex": t.sex.map({0.0: "male", 1.0: "female"}).values,
            "Age (months)": t.age_m.values,
            "Age category (months)": t.age_cat.map(lambda v: age_label.get(int(v)) if pd.notna(v) else None).values,
            "Town": t.city.values, "Municipality": t.adm2.values, "County": t.county.values,
            "Region": t.region.map(lambda r: REGION_NAMES.get(r, r) if pd.notna(r) else None).values,
            "Settlement type": t.settlement.map(lambda s: SETTLEMENT_NAMES.get(s, s) if pd.notna(s) else None).values,
            "Distance to the sea (km)": t.coast_km.round(1).values,
            "Coastal": t.coast_km.map(lambda d: ("yes" if d < 10 else "no") if pd.notna(d) else None).values,
            "Latitude": t.lat.round(5).values, "Longitude": t.lon.round(5).values,
            "Follow-up sample": yes_no(t.followup).values, "Repeated registration number": yes_no(t.repeat_id).values})
        for p in cf.TAXA[sp]:
            out[p] = t[p].values
        ws = wb.create_sheet(label)
        ws.append(list(out.columns))
        for row in out.itertuples(index=False):
            ws.append([None if isinstance(v, float) and np.isnan(v) else
                       (int(v) if isinstance(v, float) and v in (0, 1) else v) for v in row])
        for c in ws[1]:
            c.font = FB
            c.alignment = Alignment(wrap_text=True, vertical="center")
        ws.freeze_panes = "B2"
        n_fixed = len(out.columns) - len(cf.TAXA[sp])
        for i, col in enumerate(out.columns):
            ws.column_dimensions[get_column_letter(i + 1)].width = 9 if i >= n_fixed else max(10, min(28, len(col) + 2))
    ws = wb.create_sheet("Variables")
    ws.append(["Column", "Description"])
    for c in ws[1]:
        c.font = FB
    for v in VARIABLES:
        ws.append(list(v))
    ws.column_dimensions["A"].width, ws.column_dimensions["B"].width = 32, 100
    wb.save(f"{cf.TABLES}/data.xlsx")


def main():
    os.makedirs(cf.TABLES, exist_ok=True)
    d = pd.read_pickle(cf.path("descriptive.pkl"))
    wb = Workbook()
    wb.remove(wb.active)
    trends(wb, d)
    location(wb, d, pd.read_pickle(cf.path("location.pkl")))
    cooccurrence(wb, pd.read_pickle(cf.path("cooccurrence.pkl")))
    profiles(wb, pd.read_pickle(cf.path("nmf_profiles.pkl")))
    age(wb, d, pd.read_pickle(cf.path("age.pkl")))
    dogs_and_cats(wb, d, pd.read_pickle(cf.path("species.pkl")))
    wb.save(f"{cf.TABLES}/SupplementaryTables.xlsx")
    data_file()


if __name__ == "__main__":
    main()
