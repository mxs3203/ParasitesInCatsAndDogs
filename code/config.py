"""Paths, study definitions and helpers shared by the analysis scripts."""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
EXTERNAL = os.path.join(DATA, "external")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "manuscript", "figures")
TABLES = os.path.join(ROOT, "manuscript", "tables")

DOG_RESULTS = os.path.join(DATA, "GI_parasites_dog_2016-22_combined.xlsx")
CAT_RESULTS = os.path.join(DATA, "Mačke 2016.-2022. novo.xlsx")
CAT_REGISTER = os.path.join(DATA, "Mačke VetLab 2016.-2022. novo.xlsx")
DOG_SEX_FIXES = os.path.join(DATA, "dog_sex_fixes.csv")

SPECIES = ("dog", "cat")
YEARS = list(range(2016, 2023))
AGE_GROUPS = {1: "0-6", 2: "7-12", 3: "13-48", 4: "49-120", 5: ">120"}
AGES = list(AGE_GROUPS.values())

# Five study regions built from counties. North and Pannonian follow NUTS-2 2021 (Zagreb County in North);
# Istria includes Kvarner and Lika; Dalmatia runs from Zadar to Dubrovnik.
REGIONS = ["Zagreb", "North", "Pannonian", "Dalmatia", "Istria"]
COUNTY_TO_REGION = {
    "City of Zagreb": "Zagreb",
    "Međimurje": "North", "Varaždin": "North", "Koprivnica-Križevci": "North", "Krapina-Zagorje": "North",
    "Zagreb County": "North",
    "Bjelovar-Bilogora": "Pannonian", "Virovitica-Podravina": "Pannonian", "Požega-Slavonia": "Pannonian",
    "Brod-Posavina": "Pannonian", "Osijek-Baranja": "Pannonian", "Vukovar-Syrmia": "Pannonian",
    "Karlovac": "Pannonian", "Sisak-Moslavina": "Pannonian",
    "Istria": "Istria", "Primorje-Gorski Kotar": "Istria", "Lika-Senj": "Istria",
    "Zadar County": "Dalmatia", "Šibenik-Knin": "Dalmatia", "Split-Dalmatia": "Dalmatia", "Dubrovnik-Neretva": "Dalmatia",
}
SETTLEMENTS = ["Zagreb", "City >= 40k", "Town 10-40k", "Town < 10k", "Rural municipality"]

DOG_TAXA = ["Giardia", "Cryptosporidium", "Ancylostoma/Uncinaria", "Trichuris", "Toxocara", "Toxascaris", "Sarcocystis",
            "Strongyloides", "Capillaria", "Isospora canis", "Isospora ohioensis", "Isospora burrowsi",
            "Hammondia heydorni", "Dipylidium", "Neospora caninum", "Angiostrongylus", "Eimeria sp.", "Eucoleus"]
CAT_TAXA = ["Giardia", "Cryptosporidium", "Toxocara", "Isospora felis", "Isospora rivolta", "Ancylostoma",
            "Aelurostrongylus abstrusus", "Strongyloides", "Toxascaris", "Trichuris", "Capillaria", "Sarcocystis",
            "Hammondia", "Toxoplasma gondii", "Dipylidium", "Taenia", "Joyeuxiella pasqualei", "Hymenolepis", "Eimeria sp."]
TAXA = {"dog": DOG_TAXA, "cat": CAT_TAXA}

# Taxa in the infection profiles: at least 20 positive samples and, in dogs, recorded in every year.
PROFILE_TAXA = {
    "dog": ["Giardia", "Cryptosporidium", "Ancylostoma/Uncinaria", "Trichuris", "Toxocara", "Toxascaris", "Sarcocystis",
            "Strongyloides", "Capillaria", "Isospora canis", "Isospora ohioensis", "Isospora burrowsi", "Eimeria sp."],
    "cat": ["Giardia", "Cryptosporidium", "Toxocara", "Isospora felis", "Isospora rivolta", "Ancylostoma",
            "Aelurostrongylus abstrusus"],
}
N_PROFILES = {"dog": 4, "cat": 2}  # chosen by cross-validation (nmf_cv.py)


def path(*parts):
    return os.path.join(RESULTS, *parts)


def load(species):
    """Cleaned samples of one species with the derived columns used in the models."""
    t = pd.read_pickle(path(f"{species}s.pkl"))
    taxa = TAXA[species]
    t["Any parasite"] = (t[taxa].fillna(0).sum(axis=1) > 0).astype(float)
    t["agef"] = t.age_cat.map(lambda v: AGE_GROUPS[int(v)] if pd.notna(v) else "unknown")
    t["yc"] = t.Year - 2019
    t["regionf"] = t.region.where(t.region.isin(REGIONS), "unknown")
    t["coastal"] = (t.coast_km < 10).astype(float).where(t.coast_km.notna())
    t["rur"] = t.settlement.map({s: i for i, s in enumerate(SETTLEMENTS)})
    t["la"] = np.log1p(t.age_m.clip(0, 240))
    return t


def save(obj, name):
    os.makedirs(RESULTS, exist_ok=True)
    pd.to_pickle(obj, path(name))


def modelled_taxa(t, species):
    """Any parasite plus the taxa with enough positives (40 in dogs, 20 in cats) recorded in over 90% of samples."""
    taxa = TAXA[species]
    floor = 40 if species == "dog" else 20
    return ["Any parasite"] + [p for p in taxa if t[p].sum() >= floor and t[p].notna().mean() > 0.9]
