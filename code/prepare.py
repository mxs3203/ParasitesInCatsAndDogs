"""Read the result workbooks and build one cleaned table per species (results/dogs.pkl, results/cats.pkl).

Dogs come from the combined yearly table, cats from the yearly clinic sheets, linked to the laboratory register for
the date of receipt. Every location is assigned to a county, study region, municipality, settlement type and
distance to the sea.
"""
import json
import re
import unicodedata

import geopandas as gpd
import numpy as np
import openpyxl
import pandas as pd
from shapely.geometry import Point, shape

import config as cf

PLATE = {"ZG": "Zagreb", "ZG0": "Zagreb", "OS": "Osijek", "PU": "Pula", "VG": "Velika Gorica", "VA": "Varaždin",
         "VŽ": "Varaždin", "VZ": "Varaždin", "DU": "Dubrovnik", "SK": "Sisak", "KA": "Karlovac", "ST": "Split",
         "SB": "Slavonski Brod", "RI": "Rijeka", "ZD": "Zadar", "ZA": "Zadar", "BJ": "Bjelovar", "KŽ": "Križevci",
         "KC": "Koprivnica", "VU": "Vukovar", "ČK": "Čakovec", "VK": "Vinkovci", "ŠI": "Šibenik", "VT": "Virovitica",
         "PŽ": "Požega", "KR": "Krapina", "NA": "Našice", "NG": "Nova Gradiška", "DA": "Daruvar", "GS": "Gospić",
         "MA": "Makarska", "IM": "Imotski", "OG": "Ogulin", "SL": "Slatina", "ŽU": "Županja", "DJ": "Đakovo",
         "BM": "Beli Manastir", "KT": "Kutina", "DE": "Delnice", "PS": "Pakrac", "SV.NEDJELJA": "Sveta Nedjelja",
         "SVETA NEDJELJA": "Sveta Nedjelja", "SL.BROD": "Slavonski Brod", "IVANJEC": "Ivanec", "PETIRNJA": "Petrinja",
         "BELJE": "Bilje", "JASTREBA": "Jastrebarsko", "SV.I.ZELINA": "Sveti Ivan Zelina", "SV.I. ZELINA": "Sveti Ivan Zelina",
         "SV. I.ZELINA": "Sveti Ivan Zelina", "SV. I. ZELINA": "Sveti Ivan Zelina", "ZAGREB": "Zagreb",
         "LIŠNJAN": "Ližnjan", "LIZNJAN": "Ližnjan", "MECE,OS": "Mece"}

# Cat towns not in the dog data: spelling variants and approximate coordinates of villages.
CAT_ALIAS = {"Velikagorica": "Velika Gorica", "Zapršić": "Zaprešić", "Šenkovac": "Šenkovec",
             "Va Kućan Gornji": "Kućan Gornji", "Va Papinec": "Varaždin"}
CAT_TOWNS = {"Šenkovec": (46.405, 16.420), "Kućan Gornji": (46.293, 16.383), "Baška": (44.970, 14.750),
             "Buševec": (45.650, 16.110), "Cerovac Vukmanički": (45.430, 15.620), "Donja Stubica": (45.980, 15.970),
             "Drniš": (43.860, 16.160), "Garešnica": (45.570, 16.940), "Hrvatska Kostajnica": (45.230, 16.540),
             "Jalševec Nartski": (45.790, 16.180), "Macinec": (46.370, 16.310), "Slano": (42.780, 17.890),
             "Vukanovec": (46.400, 16.430), "Zlatar Bistrica": (46.040, 16.080), "Čačinci": (45.620, 17.860),
             "Desinić": (46.190, 15.600), "Okučani": (45.260, 17.200)}
CAT_ABROAD = {"Bih", "Trebinje"}
CAT_UNKNOWN = {"Ča", "Đa", "Is", "Pg", "Podgorje", "Klokovec"}  # ambiguous codes

# Column headers of the cat sheets. Species-level columns are pooled to the level analysed; None = not a GI taxon.
CAT_COLUMNS = {
    "giardia": "Giardia", "cry": "Cryptosporidium", "j.strong.tipa": "Ancylostoma", "j.str.tipa": "Ancylostoma",
    "j.trichuris": "Trichuris", "j.toxocara": "Toxocara", "j. toxocara": "Toxocara", "toxocara cati": "Toxocara",
    "toxocara canis": "Toxocara", "j.toxascaris": "Toxascaris", "jaja toxascaris leonina": "Toxascaris",
    "sarcocystis": "Sarcocystis", "sarcoystis": "Sarcocystis", "j.strongyloides": "Strongyloides",
    "isospora felis": "Isospora felis", "i.felis": "Isospora felis", "i. rivolta": "Isospora rivolta",
    "i.rivolta": "Isospora rivolta", "hammodia heydoni": "Hammondia", "j.capillaria": "Capillaria",
    "dipylidium": "Dipylidium", "dypilidium": "Dipylidium", "taenia sp.": "Taenia", "j.taenia sp": "Taenia",
    "toxoplasma gondii": "Toxoplasma gondii", "eimeria sp.": "Eimeria sp.", "eimeria,f'": "Eimeria sp.",
    "aelurostrongylus abstrusus": "Aelurostrongylus abstrusus", "hymenolepis diminuta": "Hymenolepis",
    "hymonolepis deminuta": "Hymenolepis", "hymenolepis deminuta": "Hymenolepis",
    "jayeuxiella pasqualeiu": "Joyeuxiella pasqualei",
    "neospora caninum": None, "sarcoptes": None, "jaja jšugarca": None, "čl.trakavice": None, "demodex": None,
    "isospora canis": None,
}


def city_name(label):
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return None
    lab = re.sub(r"\s+", " ", str(label).strip().upper())
    if lab in ("", "0", "NAN", "NONE", "NEMA PODATAKA"):
        return None
    return PLATE.get(lab, lab.title().replace("I ", "i ").strip())


def followup_and_repeats(t, notes, ref):
    t["followup"] = notes.fillna("").str.upper().str.contains("KONTROLA").astype(int).values
    t["repeat_id"] = (ref.notna() & pd.DataFrame({"y": t.Year, "r": ref}).duplicated(["y", "r"])).astype(int).values
    return t


def read_dogs(log):
    raw = pd.read_excel(cf.DOG_RESULTS, sheet_name="All data")
    raw = raw.rename(columns={"Ancylostoma/Uncinaria (strongyle-type eggs)": "Ancylostoma/Uncinaria"})
    log["dog_records"] = len(raw)
    raw = followup_and_repeats(raw, raw["Other findings / notes"], raw["Ref. no. (Ur. broj)"])  # before dropping rows
    empty = raw[cf.PROFILE_TAXA["dog"]].apply(pd.to_numeric, errors="coerce").isna().all(axis=1)
    log["dog_no_result"] = int(empty.sum())
    raw = raw[~empty].reset_index(drop=True)

    d = pd.DataFrame({
        "species": "dog", "Year": raw["Year"].astype(int), "ID": raw["Ref. no. (Ur. broj)"],
        "sex": pd.to_numeric(raw["Sex (Spol)"], errors="coerce").where(lambda s: s.isin([0, 1])),
        "age_m": pd.to_numeric(raw["Age (months)"], errors="coerce"),
        "age_cat": pd.to_numeric(raw["Age category (D. kat.)"], errors="coerce"),
        "label": raw["Location"], "lat": raw["Latitude"], "lon": raw["Longitude"],
        "notes": raw["Other findings / notes"], "source": raw["Source row"],
        "followup": raw.followup, "repeat_id": raw.repeat_id})
    for p in cf.DOG_TAXA:
        d[p] = pd.to_numeric(raw[p], errors="coerce").clip(upper=1).values

    # Sex: 0 = male, 1 = female. Where a pet name was typed in the sex column, sex comes from the fixes table.
    fixes = pd.read_csv(cf.DOG_SEX_FIXES)
    fix = dict(zip(fixes.source, fixes.sex.map({"M": 0.0, "F": 1.0})))
    hit = d.source.isin(fix)
    assert hit.sum() == len(fix)
    d.loc[hit, "sex"] = d.loc[hit, "source"].map(fix)
    log["dog_sex_fixed"] = int(hit.sum())

    log["dog_followups"], log["dog_repeats"] = int(d.followup.sum()), int(d.repeat_id.sum())

    # Town name per coordinate pair: the most frequent label among records with those coordinates.
    key = list(zip(d.lat.round(4), d.lon.round(4)))
    names = {}
    for k, labels in pd.Series(d.label.values, index=pd.MultiIndex.from_tuples(key)).groupby(level=[0, 1]):
        found = pd.Series([c for c in map(city_name, labels) if c]).value_counts()
        names[k] = found.index[0] if len(found) else None
    d["city"] = [names.get(k) if not pd.isna(k[0]) else None for k in key]
    d.loc[d.lat.isna(), "city"] = None
    coords = {}
    for name in {v for v in names.values() if v}:
        ks = [k for k, v in names.items() if v == name]
        coords[name] = max(ks, key=lambda k: sum(1 for x in key if x == k)) if len(ks) > 1 else ks[0]
    return d, coords


def read_register():
    """Laboratory register: receipt date per year and registration number. Owner columns are not read."""
    rows = []
    for ws in openpyxl.load_workbook(cf.CAT_REGISTER, read_only=True, data_only=True).worksheets:
        for r in ws.iter_rows(values_only=True):
            if r and r[0] is not None and re.fullmatch(r"\d+", str(r[0]).strip()) and re.match(r"\d\d\.\d\d\.\d{4}", str(r[1] or "")):
                rows.append(dict(Year=int(ws.title.strip(".")), jbu=int(str(r[0]).strip()),
                                 date=pd.to_datetime(r[1], format="%d.%m.%Y %H:%M").normalize()))
    reg = pd.DataFrame(rows)
    return reg, reg.drop_duplicates(["Year", "jbu"])


def read_cats(coords, log):
    wb = openpyxl.load_workbook(cf.CAT_RESULTS)
    rows, raw_rows = [], 0
    for year in cf.YEARS:
        sheet = list(wb[f"{year}."].iter_rows(values_only=True))
        head = [(str(h).strip().lower() or None) if h is not None else None for h in sheet[0]]
        compact = [h.replace(" ", "") if h else None for h in head]
        col = {k: compact.index(k) for k in ("ur.broj", "spol", "dob", "lokacija") if k in compact}
        taxa = {}
        for j, h in enumerate(head):
            if CAT_COLUMNS.get(h):
                taxa.setdefault(CAT_COLUMNS[h], []).append(j)
        note_cols = [j for j, h in enumerate(head) if h is None]
        for i, r in enumerate(sheet[1:], start=2):
            raw_rows += any(v is not None for v in r)
            r = list(r) + [None] * 40
            res = {}
            for name, js in taxa.items():
                vals = [r[j] for j in js if isinstance(r[j], (int, float))]
                res[name] = float(max(vals) > 0) if vals else np.nan
            if all(np.isnan(v) for v in res.values()):
                continue
            get = lambda k: r[col[k]] if k in col else None  # 2016 has no location column
            age, sex = get("dob"), get("spol")
            rows.append(dict(species="cat", Year=year, ID=get("ur.broj"), sex=sex if sex in (0, 1) else np.nan,
                             age_m=float(age) if isinstance(age, (int, float)) else np.nan, label=get("lokacija"),
                             notes="; ".join(str(r[j]).strip() for j in note_cols if isinstance(r[j], str) and r[j].strip()) or None,
                             source=f"cat {year}!{i}", **res))
    c = pd.DataFrame(rows)
    log["cat_rows"], log["cat_records"] = raw_rows, len(c)
    c = followup_and_repeats(c, c.notes, c.ID).reset_index(drop=True)
    log["cat_followups"], log["cat_repeats"] = int(c.followup.sum()), int(c.repeat_id.sum())

    reg_all, reg = read_register()
    c["jbu"] = pd.to_numeric(c.ID, errors="coerce")
    c = c.merge(reg, on=["Year", "jbu"], how="left", validate="many_to_one").drop(columns="jbu")
    log["register_entries"], log["register_linked"] = len(reg_all), int(c.date.notna().sum())
    c["age_cat"] = pd.cut(c.age_m, [-0.1, 6, 12, 48, 120, 1e9], labels=[1, 2, 3, 4, 5]).astype(float)

    c["city"] = [CAT_ALIAS.get(x, x) if x else None for x in map(city_name, c.label)]
    coords = {**CAT_TOWNS, **coords}
    c["abroad"] = c.city.isin(CAT_ABROAD)
    c.loc[c.city.isin(CAT_UNKNOWN | CAT_ABROAD), "city"] = None
    c["lat"] = [coords[x][0] if x in coords else np.nan for x in c.city]
    c["lon"] = [coords[x][1] if x in coords else np.nan for x in c.city]
    return c


def add_region(t, counties, tol_km=8):
    def county(lat, lon):
        if pd.isna(lat):
            return None
        p = Point(lon, lat)
        for name, geom in counties:
            if geom.contains(p):
                return name
        name, geom = min(counties, key=lambda c: c[1].distance(p))  # coast and islands outside simplified polygons
        return name if geom.distance(p) * 111 <= tol_km else None

    cache, out = {}, []
    for la, lo in zip(t.lat, t.lon):
        key = (round(la, 4), round(lo, 4)) if not pd.isna(la) else None
        if key not in cache:
            cache[key] = county(la, lo)
        out.append(cache[key])
    t["county"] = out
    t["region"] = [cf.COUNTY_TO_REGION.get(c) if c else None for c in out]
    t.loc[t.lat.notna() & t.county.isna(), "region"] = "Outside Croatia"
    return t


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", s)


def municipalities():
    """Municipalities (geoBoundaries ADM2) with their settlement type, from the 2021 town populations."""
    m = gpd.read_file(f"{cf.EXTERNAL}/hrv_adm2.geojson").to_crs(3765)
    towns = pd.read_csv(f"{cf.EXTERNAL}/hr_towns_population_2021.csv")
    pop = {norm(n): p for n, p in zip(towns.name, towns.population)}
    m["kind"] = np.where(m.shapeName == "Grad Zagreb", "Zagreb", np.where(m.shapeName.str.startswith("Grad "), "town", "municipality"))
    m["base"] = m.shapeName.str.replace(r"^(Grad|Općina|Opcina)\s+", "", regex=True)
    m["population"] = [pop.get(norm(b)) if k != "municipality" else np.nan for b, k in zip(m.base, m.kind)]

    def settlement(kind, p):
        if kind == "Zagreb":
            return "Zagreb"
        if kind == "town" and not pd.isna(p):
            return "City >= 40k" if p >= 40000 else ("Town 10-40k" if p >= 10000 else "Town < 10k")
        return "Rural municipality" if kind in ("town", "municipality") else None  # Lobor: Grad, not a 2021 town

    m["settlement"] = [settlement(k, p) for k, p in zip(m.kind, m.population)]
    return m


def coastline():
    coast = gpd.read_file(f"{cf.EXTERNAL}/ne_10m_coastline.shp").cx[12.5:20.5, 41.5:47.0].to_crs(3765)
    return coast.geometry.unary_union


def environment(points):
    """Municipality, settlement type and distance to the sea (km) for each coordinate pair."""
    m = municipalities()
    pts = gpd.GeoDataFrame(points, geometry=[Point(xy) for xy in zip(points.lon, points.lat)], crs=4326).to_crs(3765)
    cols = ["shapeName", "kind", "base", "population", "settlement", "geometry"]
    j = gpd.sjoin(pts, m[cols], how="left", predicate="within")
    j = j[~j.index.duplicated()]
    out = j.shapeName.isna()
    if out.any():
        near = gpd.sjoin_nearest(pts[out], m[cols], how="left", max_distance=8000)
        near = near[~near.index.duplicated()]
        for c in cols[:-1]:
            j.loc[out, c] = near[c]
    coast = coastline()
    j["coast_km"] = [p.distance(coast) / 1000 for p in pts.geometry]
    j["coast_cat"] = pd.cut(j.coast_km, [-1, 10, 50, 1e4], labels=["< 10 km", "10-50 km", "> 50 km"]).astype(str)
    j.loc[j.coast_km.isna(), "coast_cat"] = None
    return pd.DataFrame(j.drop(columns=["geometry", "index_right"])).rename(columns={"shapeName": "adm2"})


def main():
    log = {}
    dogs, coords = read_dogs(log)
    cats = read_cats(coords, log)
    gj = json.load(open(f"{cf.EXTERNAL}/hrv_adm1.geojson"))
    counties = [(f["properties"]["shapeName"], shape(f["geometry"])) for f in gj["features"]]
    dogs, cats = add_region(dogs, counties), add_region(cats, counties)
    cats.loc[cats.abroad, "region"] = "Outside Croatia"

    points = pd.concat([dogs[["lat", "lon"]], cats[["lat", "lon"]]]).dropna().drop_duplicates().reset_index(drop=True)
    env = environment(points)
    for name, t in (("dogs", dogs), ("cats", cats)):
        t = t.merge(env, on=["lat", "lon"], how="left")
        t.loc[t.region.eq("Outside Croatia"), ["settlement", "coast_cat"]] = None
        cf.save(t, f"{name}.pkl")
    json.dump(log, open(cf.path("prep_log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
