"""Manuscript figures (manuscript/figures, PNG at 300 dpi and PDF).

Main: 2 samples per year, age and sex; 3 trends by year; 4 region and settlement type; 5 co-occurrence;
6 infection profiles in dogs; 7 age, and dogs compared with cats.
Supplementary: S1 sample flow; S2 taxa recorded per year; S3 settlement type and coast; S4 prevalence by region;
S5 crude against adjusted co-occurrence; S6 number of profiles; S7 infection profiles in cats; S8 prevalence by age.
"""
import json
import os
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker
from matplotlib.patches import Patch, Rectangle

import config as cf
import prepare

plt.switch_backend("Agg")
warnings.filterwarnings("ignore")
REG = cf.REGIONS
FONT_DIRS = ["/usr/share/fonts/truetype/liberation", "/System/Library/Fonts/Supplemental", "/Library/Fonts", "C:/Windows/Fonts"]
for d in FONT_DIRS:
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.startswith(("LiberationSerif", "Times New Roman", "times")) and f.endswith(".ttf"):
                font_manager.fontManager.addfont(os.path.join(d, f))
FAMILIES = ["Liberation Serif", "Times New Roman", "Times", "DejaVu Serif"]

K0, BLUE, RED, GREY = "#000000", "#123B7A", "#C62828", "#8c8c8c"
COL = {"dog": BLUE, "cat": RED}
LAB = {"dog": "Dogs", "cat": "Cats"}
AGE = cf.AGES
SETT = cf.SETTLEMENTS
SETT_LAB = ["City of\nZagreb", "City\n≥ 40,000", "Town\n10,000-39,999", "Town\n< 10,000", "Rural\nmunicipality"]
serif = FAMILIES[0]
plt.rcParams.update({
    "font.family": FAMILIES, "font.size": 9, "axes.edgecolor": K0, "axes.labelcolor": K0,
    "xtick.color": K0, "ytick.color": K0, "text.color": K0, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white", "pdf.fonttype": 42,
    "legend.frameon": False,
    "mathtext.fontset": "custom", "mathtext.rm": serif, "mathtext.it": f"{serif}:italic", "mathtext.bf": f"{serif}:bold",
    "hatch.linewidth": 0.5})
W = 6.3
os.makedirs(cf.FIGURES, exist_ok=True)

DOG, CAT = pd.read_pickle(cf.path("dogs.pkl")), pd.read_pickle(cf.path("cats.pkl"))
T = {"dog": DOG, "cat": CAT}
PAR = cf.TAXA
LOG = json.load(open(cf.path("prep_log.json")))
YEARS = cf.YEARS
DS = pd.read_pickle(cf.path("descriptive.pkl"))
COOC = pd.read_pickle(cf.path("cooccurrence.pkl"))
LOC = pd.read_pickle(cf.path("location.pkl"))
AGE_RES = pd.read_pickle(cf.path("age.pkl"))
SPEC = pd.read_pickle(cf.path("species.pkl"))
NMF = pd.read_pickle(cf.path("nmf_profiles.pkl"))
NMF_CV = pd.read_pickle(cf.path("nmf_cv.pkl"))


def save(fig, name):
    fig.savefig(f"{cf.FIGURES}/{name}.png")
    fig.savefig(f"{cf.FIGURES}/{name}.pdf")
    plt.close(fig)


def panel(ax, letter, dx=-30, dy=6):
    """Bold panel letter at the top left, offset in points from the axes corner."""
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy), textcoords="offset points",
                ha="left", va="bottom", fontsize=11, weight="bold")


def grid(ax, axis="y"):
    ax.grid(axis=axis, color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)


def taxon(name):
    """Italic taxon name in mathtext, with 'sp.' upright."""
    sp = name.endswith(" sp.")
    core = name[:-4] if sp else name
    return r"$\it{" + core.replace(" ", r"\ ") + "}$" + (" sp." if sp else "")


def blend(c, share):
    c = np.array(to_rgb(c))
    return tuple(1 - share * (1 - c))


def pct_labels(ax, bars, vals, fs=6.5):
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{v:.0f}" if v >= 1 else f"{v:.1f}",
                ha="center", va="bottom", fontsize=fs)


# Figure S1: data sources, harmonisation, sample flow
fig = plt.figure(figsize=(W, 2.6))
ax = fig.add_subplot(111)
ax.set_axis_off()
ax.set_xlim(0, 1); ax.set_ylim(0, 1)


def box(x, y, title, body, ec, fc="white", ls="-", align=(0.5, 0.5)):
    kids = [TextArea(title, textprops=dict(fontsize=7.8, weight="bold"))]
    if body:
        kids.append(TextArea(body, textprops=dict(fontsize=7.2, multialignment="center", linespacing=1.3)))
    ab = AnnotationBbox(VPacker(children=kids, align="center", pad=0, sep=2.5), (x, y), xycoords="axes fraction",
                        box_alignment=align, frameon=True, pad=0.55, annotation_clip=False,
                        bboxprops=dict(boxstyle="round,rounding_size=0.6", ec=ec, fc=fc, lw=1.0, ls=ls))
    ax.add_artist(ab)
    return ab


def link(x, a, b, head=True, c=K0, ls="-"):
    """Vertical arrow at axes x from the bottom of box a to the top of box b."""
    ax.annotate("", xy=(x, 1), xycoords=("axes fraction", b), xytext=(x, 0), textcoords=("axes fraction", a),
                arrowprops=dict(arrowstyle="-|>" if head else "-", color=c, lw=0.9, ls=ls, shrinkA=0, shrinkB=0,
                                mutation_scale=8))


loc = {sp: int(T[sp].region.isin(REG).sum()) for sp in T}
towns = {sp: int(T[sp].loc[T[sp].region.isin(REG), "city"].nunique()) for sp in T}
XD, XC, XR, XH = 0.20, 0.58, 0.885, 0.39
YS, YP, YE, YA = 0.90, 0.635, 0.40, 0.285
sd = box(XD, YS, "Dog result sheets", f"one per year, 2016-2022\n{LOG['dog_records']:,} records", BLUE)
sc = box(XC, YS, "Cat result sheets", f"one per year, 2016-2022\n{LOG['cat_rows']:,} records", RED)
sr = box(XR, YS, "Laboratory register", f"cats, 2016-2022\n{LOG['register_entries']:,} entries", RED, ls="--")
hm = box(XH, YP, "Harmonisation",
         "column names unified and years combined; blank result = not recorded\n"
         "cat species-level entries pooled to genus; findings outside the recorded taxa not analysed", GREY, fc="#f4f4f4")
rd = box(XD + 0.17, YE, "Removed", f"no parasite result: {LOG['dog_no_result']}", GREY)
rc = box(XC + 0.17, YE, "Removed", f"no parasite result: {LOG['cat_rows'] - LOG['cat_records']}", GREY)
ad = box(XD, YA, f"{len(DOG):,} dog samples",
         f"kept and flagged: {LOG['dog_followups']} follow-ups,\n{LOG['dog_repeats']} repeated registration numbers\n"
         f"location known: {loc['dog']:,} ({towns['dog']} towns)", BLUE, align=(0.5, 1))
ac = box(XC, YA, f"{len(CAT):,} cat samples",
         f"kept and flagged: {LOG['cat_followups']} follow-ups,\n{LOG['cat_repeats']} repeated registration numbers\n"
         f"location known: {loc['cat']:,} ({towns['cat']} towns)\ndate of receipt: {LOG['register_linked']:,}", RED, align=(0.5, 1))
for x, s_, r_, a_ in ((XD, sd, rd, ad), (XC, sc, rc, ac)):
    link(x, s_, hm)
    link(x, hm, a_)
    ax.annotate("", xy=(0, 0.5), xycoords=r_, xytext=(x, 0.5), textcoords=("axes fraction", r_),
                arrowprops=dict(arrowstyle="-", color=K0, lw=0.9, shrinkA=0, shrinkB=0))
ax.annotate("", xy=(1, 0.5), xycoords=ac, xytext=(0.5, 0), textcoords=sr,
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.9, ls="--", shrinkA=0, shrinkB=0, mutation_scale=8,
                            connectionstyle="angle,angleA=-90,angleB=180,rad=0"))
ax.text(XR + 0.012, YE + 0.03, "linked by year and\nregistration number", ha="left", va="center", fontsize=6.8,
        linespacing=1.2)
save(fig, "FigureS1_sample_flow")

# Figure 2: samples per year, age, sex
fig = plt.figure(figsize=(W, 2.3))
sub = GridSpec(1, 3, width_ratios=[1.3, 1.6, 0.9], wspace=0.42, figure=fig)
a1, a2, a3 = (fig.add_subplot(sub[i]) for i in range(3))
x = np.arange(len(YEARS))
for o, sp in ((-0.2, "dog"), (0.2, "cat")):
    v = T[sp].Year.value_counts().reindex(YEARS, fill_value=0).values
    a1.bar(x + o, v, 0.38, color=COL[sp], label=f"{LAB[sp]} (n = {len(T[sp]):,})")
a1.set_xticks(x); a1.set_xticklabels([f"'{y % 100:02d}" for y in YEARS], fontsize=8)
a1.set_xlabel("Year")
a1.set_ylabel("Samples")
a1.set_ylim(0, 4000); a1.set_yticks([0, 1000, 2000, 3000])
panel(a1, "A")
a1.legend(fontsize=7.5, loc="upper left", handlelength=1, borderaxespad=0)
grid(a1)
xa = np.arange(6)
for o, sp in ((-0.2, "dog"), (0.2, "cat")):
    ac = T[sp].age_cat
    v = [100 * (ac == k).mean() for k in (1, 2, 3, 4, 5)] + [100 * ac.isna().mean()]
    b = a2.bar(xa + o, v, 0.38, color=COL[sp])
a2.set_xticks(xa); a2.set_xticklabels(AGE + ["NA"], fontsize=6.8)
a2.set_xlabel("Age (months)")
a2.set_ylabel("% of samples")
panel(a2, "B", dx=-26)
grid(a2)
xs = np.arange(3)
for o, sp in ((-0.2, "dog"), (0.2, "cat")):
    s = T[sp].sex
    v = [100 * (s == 0).mean(), 100 * (s == 1).mean(), 100 * s.isna().mean()]
    a3.bar(xs + o, v, 0.38, color=COL[sp])
a3.set_xticks(xs); a3.set_xticklabels(["Male", "Female", "NA"], fontsize=7)
a3.set_xlabel("Sex")
a3.set_ylabel("% of samples")
panel(a3, "C", dx=-26)
a3.set_ylim(0, 60)
grid(a3)
save(fig, "Figure2_dataset")

# Figure S2: taxa recorded per year
fig = plt.figure(figsize=(W, 3.9))
sub = GridSpec(1, 2, wspace=1.05, figure=fig)
for k, sp in enumerate(("dog", "cat")):
    ax = fig.add_subplot(sub[k])
    t, P = T[sp], PAR[sp]
    for i, p in enumerate(P):
        for j, y in enumerate(YEARS):
            sh = float(t.loc[t.Year == y, p].notna().mean())
            ax.add_patch(Rectangle((j + 0.05, i + 0.06), 0.9, 0.88, fc=blend(COL[sp], sh) if sh > 0 else "white",
                                   ec="none" if sh > 0 else "#c8c8c8", lw=0.5))
        ax.text(len(YEARS) + 0.25, i + 0.5, f"{int(t[p].sum()):,}", ha="left", va="center", fontsize=6.8)
    ax.set_xlim(0, len(YEARS)); ax.set_ylim(len(P), 0)
    ax.set_yticks(np.arange(len(P)) + 0.5)
    ax.set_yticklabels([taxon(p) for p in P], fontsize=7)
    ax.set_xticks(np.arange(len(YEARS)) + 0.5)
    ax.set_xticklabels([f"'{y % 100:02d}" for y in YEARS], fontsize=7)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(len(YEARS) + 0.25, -0.35, "Positive", ha="left", va="bottom", fontsize=6.8, style="italic")
    panel(ax, "A" if sp == "dog" else "B", dx=-12, dy=16)
hd, lb = [], []
for sh, l in ((1, "result in all samples"), (0.5, "result in about half"), (0, "not recorded")):
    hd.append(tuple(Patch(fc=blend(c, sh) if sh > 0 else "white", ec="none" if sh > 0 else "#c8c8c8", lw=0.5)
                    for c in (BLUE, RED)))
    lb.append(l)
fig.legend(hd, lb, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)}, ncol=3, loc="lower center",
           bbox_to_anchor=(0.5, 0.0), fontsize=7, handlelength=2.2, handletextpad=0.5, columnspacing=1.4,
)
fig.subplots_adjust(bottom=0.12)
save(fig, "FigureS2_taxa_by_year")

# Figure S3: location and environment
ADM1 = gpd.read_file(f"{cf.EXTERNAL}/hrv_adm1.geojson")
ADM1["region"] = ADM1.shapeName.map(cf.COUNTY_TO_REGION)
RG = {"Zagreb": "#707070", "North": "#d4d4d4", "Pannonian": "#f2f2f2", "Dalmatia": "#bababa", "Istria": "#939393"}
REG_LAB = {"Zagreb": "City of Zagreb", "North": "North", "Pannonian": "Pannonian", "Dalmatia": "Dalmatia",
           "Istria": "Istria, Kvarner, Lika"}
ADM2 = prepare.municipalities()
land = ADM2.geometry.unary_union
zone = gpd.GeoSeries([prepare.coastline().buffer(10000).intersection(land)], crs=3765).to_crs(4326)
ADM2 = ADM2.to_crs(4326)
SG = dict(zip(SETT, ["#3d3d3d", "#6e6e6e", "#9e9e9e", "#cdcdcd", "#f3f3f3"]))


def frame(ax, x0=13.3):
    ax.set_xlim(x0, 19.6); ax.set_ylim(42.3, 46.6)
    ax.set_aspect(1 / np.cos(np.radians(44.5)))
    ax.set_axis_off()


fig = plt.figure(figsize=(W, 5.6))
gs = GridSpec(2, 1, height_ratios=[2.9, 2.0], hspace=0.38, figure=fig)
sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], width_ratios=[1.25, 1], wspace=0.3)
ax = fig.add_subplot(sub[0])
ADM2.plot(ax=ax, color=[SG[s] for s in ADM2.settlement], edgecolor="white", linewidth=0.15)
zone.plot(ax=ax, facecolor="none", edgecolor=K0, hatch="//////", linewidth=0)
ADM1.dissolve("region").boundary.plot(ax=ax, color=K0, linewidth=0.5)
frame(ax, x0=11.9)
ax.set_anchor("N")  # top of the map level with panel B
panel(ax, "A", dx=0, dy=6)
ax.legend(handles=[Patch(fc=SG[s], ec=K0, lw=0.4, label=l.replace("\n", " ")) for s, l in zip(SETT, SETT_LAB)]
          + [Patch(fc="white", ec=K0, lw=0.4, hatch="//////", label="Within 10 km of the sea")],
          loc="lower left", bbox_to_anchor=(-0.03, -0.03), fontsize=6.8, handlelength=1.3, labelspacing=0.35)
ax = fig.add_subplot(sub[1])
xr = np.arange(len(REG))
for o, sp in ((-0.2, "dog"), (0.2, "cat")):
    t = T[sp][T[sp].region.isin(REG)]
    v = (100 * t.region.value_counts(normalize=True)).reindex(REG, fill_value=0).values
    pct_labels(ax, ax.bar(xr + o, v, 0.38, color=COL[sp], label=f"{LAB[sp]} (n = {len(t):,})"), v)
REG_TICK = {"Zagreb": "City of\nZagreb", "North": "North", "Pannonian": "Pannon-\nian", "Dalmatia": "Dalmatia",
            "Istria": "Istria,\nKvarner,\nLika"}
ax.set_xticks(xr); ax.set_xticklabels([REG_TICK[r] for r in REG], fontsize=6.8)
ax.set_ylabel("% of samples with location")
ax.set_ylim(0, 100)
panel(ax, "B")
ax.legend(fontsize=7, loc="upper right", handlelength=1)
grid(ax)

sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], width_ratios=[1.6, 1], wspace=0.3)
ax = fig.add_subplot(sub[0])
xs = np.arange(len(SETT))
for o, sp in ((-0.2, "dog"), (0.2, "cat")):
    t = T[sp][T[sp].region.isin(REG) & T[sp].settlement.notna()]
    v = (100 * t.settlement.value_counts(normalize=True)).reindex(SETT, fill_value=0).values
    pct_labels(ax, ax.bar(xs + o, v, 0.38, color=COL[sp]), v)
ax.set_xticks(xs); ax.set_xticklabels(SETT_LAB, fontsize=7)
ax.set_ylabel("% of samples with location")
ax.set_ylim(0, 100)
panel(ax, "C")
grid(ax)
ax = fig.add_subplot(sub[1])
COASTAL = ["Coastal\n(< 10 km)", "Inland\n(≥ 10 km)"]
xc = np.arange(2)
for o, sp in ((-0.2, "dog"), (0.2, "cat")):
    t = T[sp][T[sp].region.isin(REG) & T[sp].coast_km.notna()]
    c10 = 100 * (t.coast_km < 10).mean()
    v = [c10, 100 - c10]
    pct_labels(ax, ax.bar(xc + o, v, 0.38, color=COL[sp]), v)
ax.set_xticks(xc); ax.set_xticklabels(COASTAL, fontsize=7)
ax.set_xlim(-0.6, 1.6)
ax.set_xlabel("Distance to the sea")
ax.set_ylim(0, 100)
panel(ax, "D")
grid(ax)
save(fig, "FigureS3_location")

# Figure 3: crude and age-standardised prevalence by year
F3 = [("Any parasite", "Any parasite", "Any parasite"), ("Giardia", "Giardia", "Giardia"),
      ("Cryptosporidium", "Cryptosporidium", "Cryptosporidium"), ("Toxocara", "Toxocara", "Toxocara"),
      ("I. canis / I. felis", "Isospora canis", "Isospora felis"),
      ("I. ohioensis / I. rivolta", "Isospora ohioensis", "Isospora rivolta"),
      ("I. burrowsi", "Isospora burrowsi", None), ("Eimeria sp.", "Eimeria sp.", None), ("Trichuris", "Trichuris", None)]
fig, axs = plt.subplots(3, 3, figsize=(W, 6.0), gridspec_kw=dict(wspace=0.3, hspace=0.35))
for k, (ax, (lab, dp, cp)) in enumerate(zip(axs.ravel(), F3)):
    top = 0
    for sp, p, mk in (("dog", dp, "o"), ("cat", cp, "s")):
        if p is None:
            continue
        t = DS[sp]["std_year"]
        t = t[t.parasite == p].sort_values("year")
        ax.fill_between(t.year, t.crude_lo, t.crude_hi, color=(*to_rgb(COL[sp]), 0.13), lw=0)
        ax.plot(t.year, t.crude, "-", marker=mk, color=COL[sp], ms=3.2, lw=1.2)
        ax.plot(t.year, t["std"], "--", marker=mk, mfc="white", color=COL[sp], ms=3.2, lw=1.0)
        top = max(top, np.nanmax(t.crude_hi), np.nanmax(t["std"]))
    ax.set_ylim(0, top * 1.32)
    ax.set_xticks([2016, 2018, 2020, 2022])
    ax.set_xticklabels(["2016", "2018", "2020", "2022"], fontsize=7.5)
    ax.tick_params(axis="y", labelsize=7.5)
    txt = taxon(lab) if lab.endswith(" sp.") else lab  # mathtext only where "sp." must stay upright
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha="left", va="top", fontsize=8,
            style="normal" if lab == "Any parasite" or lab.endswith(" sp.") else "italic")
    panel(ax, "ABCDEFGHI"[k], dx=-24, dy=3)
    grid(ax)
for ax in axs[:, 0]:
    ax.set_ylabel("Prevalence (%)", fontsize=8.5)
hl = [Line2D([], [], color=BLUE, marker="o", ms=3.5, lw=1.2, label="Dogs, crude"),
      Line2D([], [], color=BLUE, marker="o", mfc="white", ms=3.5, lw=1.0, ls="--", label="Dogs, age-standardised"),
      Line2D([], [], color=RED, marker="s", ms=3.5, lw=1.2, label="Cats, crude"),
      Line2D([], [], color=RED, marker="s", mfc="white", ms=3.5, lw=1.0, ls="--", label="Cats, age-standardised")]
fig.legend(handles=hl, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.955), fontsize=7.5, handlelength=2.2,
           columnspacing=1.2)
save(fig, "Figure3_trends")

# Figure 4: sample locations, prevalence by region and by settlement type
smax = 10900  # circle area scale of the sample maps
SET_T = ["Zagreb", "City ≥40k", "Town 10-40k", "Town <10k", "Rural"]
F4 = [("region", "Hookworms", "Ancylostoma/Uncinaria", "Ancylostoma"), ("region", "Trichuris", "Trichuris", None),
      ("region", "Toxascaris", "Toxascaris", None), ("region", "Sarcocystis", "Sarcocystis", None),
      ("settle", "Hookworms", "Ancylostoma/Uncinaria", "Ancylostoma"), ("settle", "Trichuris", "Trichuris", None),
      ("settle", "Toxocara", "Toxocara", "Toxocara"), ("settle", "Toxascaris", "Toxascaris", None)]
fig = plt.figure(figsize=(W, 8.6))
gs0 = GridSpec(2, 1, height_ratios=[3.0, 4.9], hspace=0.3, figure=fig)
top = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[0], wspace=0.02)
bot = GridSpecFromSubplotSpec(2, 4, subplot_spec=gs0[1], wspace=0.36, hspace=0.75)
# A, B: sample locations
for k, sp in enumerate(("dog", "cat")):
    ax = fig.add_subplot(top[k])
    ADM1.plot(ax=ax, color=[RG[r] for r in ADM1.region], edgecolor="white", linewidth=0.4)
    ADM1.dissolve("region").boundary.plot(ax=ax, color=K0, linewidth=0.6)
    t = T[sp][T[sp].region.isin(REG)]
    cc = t.groupby(["lat", "lon"]).size().reset_index(name="n")
    ax.scatter(cc.lon, cc.lat, s=4 + 400 * np.sqrt(cc.n / smax), facecolor=(*to_rgb(COL[sp]), 0.45),
               edgecolor=COL[sp], lw=0.4, zorder=3)
    frame(ax)
    panel(ax, "AB"[k], dx=0, dy=-4)
    if sp == "dog":
        ax.legend(handles=[Patch(facecolor=RG[r], edgecolor=K0, lw=0.5, label=REG_LAB[r]) for r in REG],
                  loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=7, handlelength=1.3, columnspacing=1.2)
    else:
        hs = [ax.scatter([], [], s=4 + 400 * np.sqrt(nn / smax), facecolor="white", edgecolor=K0, lw=0.6,
                         label=f"{nn:,}" + (" samples" if nn == 10 else "")) for nn in (10, 100, 1000, 10000)]
        ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=7, columnspacing=1.6,
                  handletextpad=0.8, handlelength=2.6, labelspacing=1.6)
# C-J: prevalence by region and by settlement type
axs = np.array([[fig.add_subplot(bot[i, j]) for j in range(4)] for i in range(2)])
for k, (ax, (kind, lab, dp, cp)) in enumerate(zip(axs.ravel(), F4)):
    levels = REG if kind == "region" else SETT
    top_ = 0
    for o, sp, p, mk in ((-0.13, "dog", dp, "o"), (0.13, "cat", cp, "s")):
        if p is None:
            continue
        if kind == "region":
            t = DS[sp]["prev_region"]
            t = t[t.parasite == p].set_index("group").reindex(levels)
        else:
            t = LOC[sp]["settle_prev"]
            t = t[t.parasite == p].set_index("settlement").reindex(levels)
        xx = np.arange(len(levels)) + o
        ax.vlines(xx, t.lo, t.hi, color=COL[sp], lw=1.1)
        ax.plot(xx, t.prev, mk, color=COL[sp], ms=4, mfc=COL[sp] if sp == "dog" else "white", mew=1.1)
        top_ = max(top_, np.nanmax(t.hi))
    ax.set_ylim(0, top_ * 1.3)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels if kind == "region" else SET_T, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Region" if kind == "region" else "Settlement type", fontsize=8)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.text(0.03, 0.97, lab, transform=ax.transAxes, ha="left", va="top", fontsize=8,
            style="normal" if lab == "Hookworms" else "italic")
    panel(ax, "CDEFGHIJ"[k], dx=-24, dy=3)
    grid(ax)
for ax in axs[:, 0]:
    ax.set_ylabel("Prevalence (%)", fontsize=8.5)
hl = [Line2D([], [], color=BLUE, marker="o", ms=4, lw=0, label="Dogs"),
      Line2D([], [], color=RED, marker="s", mfc="white", ms=4, lw=0, label="Cats")]
fig.legend(handles=hl, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.012), fontsize=8)
save(fig, "Figure4_region_settlement")

# Figure 5: co-occurrence
def short(p):
    return (p.replace("Ancylostoma/Uncinaria", "Hookworms").replace("Ancylostoma", "Hookworms").replace("Isospora ", "I. ")
            .replace("Cryptosporidium", "Cryptospor.").replace("Aelurostrongylus abstrusus", "Aelurostr."))


def heatmaps(fig, top):
    for k, sp in enumerate(("dog", "cat")):
        ax = fig.add_subplot(top[k])
        Cc, par = COOC[sp]["cooc"], COOC[sp]["taxa"]
        n = len(par)
        ORm = np.full((n, n), np.nan); Qm = np.ones((n, n))
        for _, r in Cc.iterrows():
            i, j = par.index(r.a), par.index(r.b)
            ORm[i, j] = ORm[j, i] = r.OR_MH
            Qm[i, j] = Qm[j, i] = r.q_MH if pd.notna(r.q_MH) else 1
        cmap = LinearSegmentedColormap.from_list("s", ["#ffffff", COL[sp]])
        L = np.clip(np.log2(np.where(ORm > 0, ORm, 1)), 0, 4)  # OR <= 1 white; 16 or more full colour
        Lm = np.ma.array(L, mask=np.triu(np.ones((n, n), bool)))[1:, :-1]
        im = ax.imshow(Lm, cmap=cmap, vmin=0, vmax=4)
        for i in range(1, n):
            for j in range(i):
                v = ORm[i, j]
                ax.text(j, i - 1, f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center", va="center",
                        fontsize=5.6 if sp == "dog" else 7, color="white" if np.log2(max(v, 1)) > 2.6 else K0,
                        fontweight="bold" if Qm[i, j] < 0.05 else "normal")
        sh = [short(p) for p in par]
        ax.set_xticks(range(n - 1)); ax.set_xticklabels(sh[:-1], rotation=55, ha="right", fontsize=6.8,
                                                        style="italic")
        ax.set_yticks(range(n - 1)); ax.set_yticklabels(sh[1:], fontsize=6.8, style="italic")
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            if lab.get_text() == "Hookworms":
                lab.set_style("normal")
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.tick_params(length=0)
        panel(ax, "AB"[k], dx=-50 if sp == "dog" else -45, dy=2)
        cb = fig.colorbar(im, ax=ax, fraction=0.04 if sp == "dog" else 0.06, pad=0.02, ticks=[0, 1, 2, 3, 4])
        cb.ax.set_yticklabels(["≤1", "2", "4", "8", "16+"], fontsize=6.5)
        if sp == "cat":
            cb.set_label("Adjusted odds ratio", fontsize=7)
        cb.outline.set_visible(False)


# C, D: Cryptosporidium-positive share by year, among Giardia-positive and Giardia-negative samples
fig = plt.figure(figsize=(W, 6.6))
gs0 = GridSpec(2, 1, height_ratios=[3.6, 2.2], hspace=0.42, figure=fig)
heatmaps(fig, GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[0], width_ratios=[1.75, 1], wspace=0.5))
bot = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[1], wspace=0.35)
for k, (key, ymax, lab) in enumerate((("gpos", 100, "Giardia-positive samples"), ("gneg", 1.0, "Giardia-negative samples"))):
    ax = fig.add_subplot(bot[k])
    for sp, mk in (("dog", "o"), ("cat", "s")):
        by = DS[sp]["crypto_giardia"]["by_year"]
        ax.plot(by.index, by[key], "-", marker=mk, color=COL[sp], ms=3.5, lw=1.2, mfc=COL[sp] if sp == "dog" else "white")
    ax.set_ylim(0, ymax)
    ax.set_xticks([2016, 2018, 2020, 2022])
    ax.set_ylabel("Cryptosporidium-positive (%)", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.text(0.97, 0.97, lab, transform=ax.transAxes, ha="right", va="top", fontsize=8)
    panel(ax, "CD"[k], dx=-34, dy=3)
    grid(ax)
    if k == 0:
        ax.legend(handles=[Line2D([], [], color=BLUE, marker="o", ms=3.5, lw=1.2, label="Dogs"),
                           Line2D([], [], color=RED, marker="s", mfc="white", ms=3.5, lw=1.2, label="Cats")],
                  fontsize=7, loc="lower left")
save(fig, "Figure5_cooccurrence")

# Figure S5: crude against adjusted co-occurrence odds ratios
fig, ax = plt.subplots(figsize=(3.4, 3.2))
for sp, mk in (("dog", "o"), ("cat", "s")):
    Cc = COOC[sp]["cooc"].dropna(subset=["OR_MH"])
    Cc = Cc[Cc.OR_MH > 0]
    x, y = np.log2(Cc.OR_crude), np.log2(Cc.OR_MH)
    sig = Cc.q_MH < 0.05
    ax.scatter(x[sig], y[sig], s=14, marker=mk, color=COL[sp], label=f"{LAB[sp]}, q < 0.05 adjusted")
    ax.scatter(x[~sig], y[~sig], s=14, marker=mk, facecolor="white", edgecolor=COL[sp], lw=0.8,
               label=f"{LAB[sp]}, not significant")
lim = [-2, 10.5]
ax.plot(lim, lim, color=K0, lw=0.7, ls=":")
ax.axhline(0, color="#999999", lw=0.6); ax.axvline(0, color="#999999", lw=0.6)
ax.set_xlim(lim); ax.set_ylim(lim)
tk = [-2, 0, 2, 4, 6, 8, 10]
lab_ = [f"{2.0 ** t:g}" if t >= 0 else f"1/{2 ** -t}" for t in tk]
ax.set_xticks(tk); ax.set_xticklabels(lab_, fontsize=7)
ax.set_yticks(tk); ax.set_yticklabels(lab_, fontsize=7)
ax.set_xlabel("Crude odds ratio", fontsize=8.5); ax.set_ylabel("Adjusted odds ratio", fontsize=8.5)
ax.legend(fontsize=6.3, loc="upper left", handletextpad=0.3)
grid(ax, "both")
save(fig, "FigureS5_crude_vs_adjusted")

# Figure 6: Bernoulli NMF infection profiles in dogs


def ptext(ax, p):
    ax.text(0.97, 0.97, "p < 0.001" if p < 0.001 else f"p = {p:.2f}", transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5)


B4 = NMF["dog"]
S4 = {"region": B4["region"], "settle": B4["settle"]}
NAMES4 = ["Low prevalence", "Helminths and coccidia", "Giardia and Cryptosporidium", "Polyparasitism"]
SHOW = [(2, "o", BLUE), (1, "D", "white"), (3, "s", "white")]  # profile, marker, face colour
fig = plt.figure(figsize=(W, 7.6))
top = fig.add_gridspec(1, 1, left=0.36, right=0.97, top=0.97, bottom=0.75)
rows = [fig.add_gridspec(1, 3, left=0.1, right=0.97, top=0.62, bottom=0.45, wspace=0.32),
        fig.add_gridspec(1, 3, left=0.1, right=0.97, top=0.32, bottom=0.15, wspace=0.32)]
ax = fig.add_subplot(top[0])
H4, items = B4["H"], B4["taxa"]
cmap = LinearSegmentedColormap.from_list("s", ["#ffffff", BLUE])
ax.imshow(H4, cmap=cmap, vmin=0, vmax=1, aspect="auto")
for i in range(H4.shape[0]):
    for j in range(H4.shape[1]):
        if H4[i, j] >= 0.05:
            ax.text(j, i, f"{H4[i, j]:.2f}", ha="center", va="center", fontsize=5.6, color="white" if H4[i, j] > 0.5 else K0)
mw = B4["W"].mean(0)
ax.set_yticks(range(4))
ax.set_yticklabels([f"{nm} ({100 * w:.1f}%)" if w < 0.01 else f"{nm} ({100 * w:.0f}%)" for nm, w in zip(NAMES4, mw)], fontsize=6.8)
ax.set_xticks(range(len(items)))
ax.set_xticklabels([short(p) for p in items], rotation=55, ha="right", fontsize=6.8)
for lab in ax.get_xticklabels():
    lab.set_style("normal" if lab.get_text() == "Hookworms" else "italic")
ax.tick_params(length=0)
for s_ in ax.spines.values():
    s_.set_visible(False)
panel(ax, "A", dx=-150, dy=4)
for r_, (kind, levels, ticks, xl) in enumerate((("region", REG, REG, "Region"), ("settle", SETT, SET_T, "Settlement type"))):
    src = S4[kind]
    for n_, (c, mk, fc) in enumerate(SHOW):
        ax = fig.add_subplot(rows[r_][n_])
        est = np.array([src["shares"][x][c] for x in levels]) * 100
        ci_ = np.array([src["shares_ci"][x][:, c] for x in levels]) * 100
        xx = np.arange(len(levels))
        ax.vlines(xx, ci_[:, 0], ci_[:, 1], color=BLUE, lw=1.1)
        ax.plot(xx, est, mk, color=BLUE, ms=4.3, mfc=fc, mew=1.1)
        ax.set_xticks(xx)
        ax.set_xticklabels(ticks, rotation=40, ha="right", fontsize=7)
        ax.set_xlabel(xl, fontsize=8)
        ax.tick_params(axis="y", labelsize=7.5)
        if n_ == 0:
            ax.set_ylabel("Adjusted profile share (%)", fontsize=8.5)
        lo, hi = ax.get_ylim()
        ax.set_ylim(0, hi * 1.3)
        ax.text(0.03, 0.97, NAMES4[c], transform=ax.transAxes, ha="left", va="top", fontsize=7.2,
                style="italic" if c == 2 else "normal")
        if n_ == 2:
            ptext(ax, src["p"])
        panel(ax, "BCDEFG"[3 * r_ + n_], dy=3)
        grid(ax)
save(fig, "Figure6_infection_profiles")

# Figure S7: Bernoulli NMF infection profiles in cats
B2 = NMF["cat"]
S2 = {"region": B2["region"], "settle": B2["settle"]}
NAMES2 = ["Low prevalence", "Giardia and Cryptosporidium"]
fig = plt.figure(figsize=(W, 5.0))
top = fig.add_gridspec(1, 1, left=0.36, right=0.8, top=0.97, bottom=0.66)
row = fig.add_gridspec(1, 2, left=0.1, right=0.97, top=0.5, bottom=0.17, wspace=0.3)
ax = fig.add_subplot(top[0])
H2, items = B2["H"], B2["taxa"]
cmap = LinearSegmentedColormap.from_list("s", ["#ffffff", RED])
ax.imshow(H2, cmap=cmap, vmin=0, vmax=1, aspect="auto")
for i in range(H2.shape[0]):
    for j in range(H2.shape[1]):
        if H2[i, j] >= 0.05:
            ax.text(j, i, f"{H2[i, j]:.2f}", ha="center", va="center", fontsize=5.6, color="white" if H2[i, j] > 0.5 else K0)
mw = B2["W"].mean(0)
ax.set_yticks(range(2))
ax.set_yticklabels([f"{nm} ({100 * w:.0f}%)" for nm, w in zip(NAMES2, mw)], fontsize=6.8)
ax.set_xticks(range(len(items)))
ax.set_xticklabels([short(p) for p in items], rotation=55, ha="right", fontsize=6.8)
for lab in ax.get_xticklabels():
    lab.set_style("normal" if lab.get_text() == "Hookworms" else "italic")
ax.tick_params(length=0)
for s_ in ax.spines.values():
    s_.set_visible(False)
panel(ax, "A", dx=-150, dy=4)
for n_, (kind, levels, ticks, xl) in enumerate((("region", REG, REG, "Region"), ("settle", SETT, SET_T, "Settlement type"))):
    src = S2[kind]
    ax = fig.add_subplot(row[n_])
    est = np.array([src["shares"][x][1] for x in levels]) * 100
    ci_ = np.array([src["shares_ci"][x][:, 1] for x in levels]) * 100
    xx = np.arange(len(levels))
    ax.vlines(xx, ci_[:, 0], ci_[:, 1], color=RED, lw=1.1)
    ax.plot(xx, est, "o", color=RED, ms=4.3, mew=1.1)
    ax.set_xticks(xx)
    ax.set_xticklabels(ticks, rotation=40, ha="right", fontsize=7)
    ax.set_xlabel(xl, fontsize=8)
    ax.tick_params(axis="y", labelsize=7.5)
    if n_ == 0:
        ax.set_ylabel("Adjusted profile share (%)", fontsize=8.5)
    lo, hi = ax.get_ylim()
    ax.set_ylim(0, hi * 1.3)
    ax.text(0.03, 0.97, NAMES2[1], transform=ax.transAxes, ha="left", va="top", fontsize=7.2, style="italic")
    ptext(ax, src["p"])
    panel(ax, "BC"[n_], dy=3)
    grid(ax)
save(fig, "FigureS7_cat_profiles")

# Figure S6: choice of the number of Bernoulli NMF profiles
fig, axs = plt.subplots(1, 2, figsize=(W, 2.6), gridspec_kw=dict(wspace=0.35))
for k, sp in enumerate(("dog", "cat")):
    ax = axs[k]
    g = NMF_CV[sp]["gain"]
    Ks = sorted(g)
    ax.plot(Ks, [g[x] for x in Ks], "-o", color=COL[sp], ms=4, lw=1.2)
    ax.axvline(cf.N_PROFILES[sp], color="#999999", lw=0.7, ls=":")
    ax.set_xticks(Ks)
    ax.set_xlim(1.7, 5.3)
    ax.set_xlabel("Number of profiles", fontsize=8.5)
    if k == 0:
        ax.set_ylabel("Held-out log-likelihood\ngain over independence", fontsize=8.5)
    lo, hi = min(g.values()), max(g.values())
    ax.set_ylim(lo - 0.3 * (hi - lo), hi + 0.2 * (hi - lo))
    ax.tick_params(labelsize=7.5)
    grid(ax)
    panel(ax, "AB"[k], dy=3)
save(fig, "FigureS6_profile_selection")

# Figure 7: age, and dogs compared with cats
AGE_TICKS = [0, 2, 6, 12, 24, 60, 144]
AGE_MAX = 144  # about the 96th percentile of age in dogs and the 95th in cats
F7_TOP = [("Giardia", "Giardia", "Giardia"), ("Toxocara", "Toxocara", "Toxocara"), ("Cryptosporidium", "Cryptosporidium", "Cryptosporidium")]
F7_MID = [("Any parasite", ("dog", "cat"), "Any parasite"), ("Hookworms", ("dog",), "Ancylostoma/Uncinaria"),
          ("I. canis", ("dog",), "Isospora canis"), ("I. felis", ("cat",), "Isospora felis")]


def age_axis(ax, label=True):
    ax.set_xscale("function", functions=(lambda x: np.log1p(np.maximum(x, 0)), np.expm1))
    ax.set_xlim(0, AGE_MAX)
    ax.set_xticks(AGE_TICKS)
    ax.set_xticklabels([str(t) for t in AGE_TICKS], fontsize=6.8)
    if label:
        ax.set_xlabel("Age (months)", fontsize=8)


def curve(ax, sp, p):
    c = AGE_RES[sp]["age_curves"][p]
    c = c[c.age <= AGE_MAX]
    ax.fill_between(c.age, c.lo, c.hi, color=(*to_rgb(COL[sp]), 0.15), lw=0)
    ax.plot(c.age, c.prev, color=COL[sp], lw=1.3, ls="-" if sp == "dog" else "--")
    return float(c.prev.max()), float(c.hi.max())


fig = plt.figure(figsize=(W, 8.0))
R1 = fig.add_gridspec(1, 3, left=0.09, right=0.98, top=0.97, bottom=0.74, wspace=0.32)
R2 = fig.add_gridspec(1, 4, left=0.09, right=0.98, top=0.64, bottom=0.44, wspace=0.42)
R3 = fig.add_gridspec(1, 2, left=0.09, right=0.98, top=0.33, bottom=0.12, wspace=0.28)
for k, (lab, dp, cp) in enumerate(F7_TOP):
    ax = fig.add_subplot(R1[k])
    top = max(curve(ax, "dog", dp)[1], curve(ax, "cat", cp)[1])
    ax.set_ylim(0, top * 1.25)
    age_axis(ax)
    ax.tick_params(axis="y", labelsize=7.5)
    if k == 0:
        ax.set_ylabel("Prevalence (%)", fontsize=8.5)
    ax.text(0.97, 0.97, lab, transform=ax.transAxes, ha="right", va="top", fontsize=8, style="italic")
    panel(ax, "ABC"[k], dx=-24, dy=3)
    grid(ax)
for k, (lab, sps, p) in enumerate(F7_MID):
    ax = fig.add_subplot(R2[k])
    pm, hm = (max(v) for v in zip(*[curve(ax, sp, p) for sp in sps]))
    ax.set_ylim(0, min(hm, 2 * pm) * 1.3)  # wide CI at older ages are cut at the top of the panel
    age_axis(ax)
    ax.tick_params(axis="y", labelsize=7.2)
    ax.tick_params(axis="x", labelsize=6.2)
    for t_ in ax.get_xticklabels():
        t_.set_rotation(90)
    if k == 0:
        ax.set_ylabel("Prevalence (%)", fontsize=8.5)
    ax.text(0.97, 0.97, lab, transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            style="normal" if lab in ("Hookworms", "Any parasite") else "italic")
    panel(ax, "DEFG"[k], dx=-22, dy=3)
    grid(ax)
# H: Giardia, OR per year within each age category (separate slopes), dogs and cats
ax = fig.add_subplot(R3[0])
xs = np.arange(len(AGE))
for o, sp, mk in ((-0.12, "dog", "o"), (0.12, "cat", "s")):
    r = AGE_RES[sp]["age_year"].set_index("parasite").loc["Giardia"]
    est = np.array([r[f"ORyr_{a}"] for a in AGE]); lo = np.array([r[f"lo_{a}"] for a in AGE]); hi = np.array([r[f"hi_{a}"] for a in AGE])
    ax.vlines(xs + o, lo, hi, color=COL[sp], lw=1.1)
    ax.plot(xs + o, est, mk, color=COL[sp], ms=4, mfc=COL[sp] if sp == "dog" else "white", mew=1.1)
ax.axhline(1, color="#888888", lw=0.7, ls="--")
ax.set_yscale("log")
ax.set_yticks([0.4, 0.6, 0.8, 1, 1.25]); ax.set_yticklabels(["0.4", "0.6", "0.8", "1", "1.25"])
ax.minorticks_off()
ax.set_ylim(0.33, 1.5)
ax.set_xticks(xs); ax.set_xticklabels(AGE, fontsize=7.2)
ax.set_xlabel("Age (months)", fontsize=8)
ax.set_ylabel(taxon("Giardia") + ", OR per year", fontsize=8.5)
ax.tick_params(axis="y", labelsize=7.5)
qd = AGE_RES["dog"]["age_year"].set_index("parasite").loc["Giardia", "q_slopes"]
qc = AGE_RES["cat"]["age_year"].set_index("parasite").loc["Giardia", "q_slopes"]
ax.text(0.03, 0.04, f"Age × year: dogs q = {qd:.2f}, cats q = {qc:.3f}", transform=ax.transAxes, ha="left", va="bottom", fontsize=7)
panel(ax, "H", dx=-30, dy=3)
grid(ax)
# I: cats against dogs, OR by age category (species x age model)
ax = fig.add_subplot(R3[1])
for o, p, mk, fc in ((-0.18, "Giardia", "o", K0), (0.0, "Toxocara", "s", "white"), (0.18, "Cryptosporidium", "D", "white")):
    e = SPEC["effects"][p]["age"].set_index("age").reindex(AGE)
    ax.vlines(xs + o, e.lo, e.hi, color=K0, lw=1.0)
    ax.plot(xs + o, e.OR, mk, color=K0, ms=4, mfc=fc, mew=1.0, label=p)
ax.axhline(1, color="#888888", lw=0.7, ls="--")
ax.set_yscale("log")
ax.set_yticks([0.25, 0.5, 1, 2, 4]); ax.set_yticklabels(["0.25", "0.5", "1", "2", "4"])
ax.minorticks_off()
ax.set_ylim(0.18, 16)
ax.set_xticks(xs); ax.set_xticklabels(AGE, fontsize=7.2)
ax.set_xlabel("Age (months)", fontsize=8)
ax.set_ylabel("OR, cats vs dogs", fontsize=8.5)
ax.tick_params(axis="y", labelsize=7.5)
lg = ax.legend(fontsize=7, loc="upper center", ncol=3, handletextpad=0.3, columnspacing=0.9)
for t_ in lg.get_texts():
    t_.set_fontstyle("italic")
panel(ax, "I", dx=-30, dy=3)
grid(ax)
hl = [Line2D([], [], color=BLUE, lw=1.3, marker="o", ms=4, label="Dogs"),
      Line2D([], [], color=RED, lw=1.3, ls="--", marker="s", mfc="white", ms=4, label="Cats")]
fig.legend(handles=hl, ncol=2, loc="upper center", bbox_to_anchor=(0.53, 0.035), fontsize=8, handlelength=2.6)
save(fig, "Figure7_age_dogs_cats")

# Figures S4 and S8: prevalence by region and by age, all modelled taxa
ALLT = [("Any parasite", "Any parasite", "Any parasite"), ("Giardia", "Giardia", "Giardia"),
        ("Cryptosporidium", "Cryptosporidium", "Cryptosporidium"), ("Toxocara", "Toxocara", "Toxocara"),
        ("Hookworms", "Ancylostoma/Uncinaria", "Ancylostoma"), ("I. canis / I. felis", "Isospora canis", "Isospora felis"),
        ("I. ohioensis / I. rivolta", "Isospora ohioensis", "Isospora rivolta"), ("I. burrowsi", "Isospora burrowsi", None),
        ("Eimeria sp.", "Eimeria sp.", None), ("Trichuris", "Trichuris", None), ("Toxascaris", "Toxascaris", None),
        ("Sarcocystis", "Sarcocystis", None), ("Strongyloides", "Strongyloides", None), ("Capillaria", "Capillaria", None),
        ("Aelurostrongylus abstrusus", None, "Aelurostrongylus abstrusus")]


def tlabel(ax, lab, fs=7.5):
    txt = taxon(lab) if lab.endswith(" sp.") else lab
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha="left", va="top", fontsize=fs,
            style="normal" if lab in ("Any parasite", "Hookworms") or lab.endswith(" sp.") else "italic")


def species_legend(fig, y):
    hl = [Line2D([], [], color=BLUE, marker="o", ms=4, lw=1.2, label="Dogs"),
          Line2D([], [], color=RED, marker="s", mfc="white", ms=4, lw=1.2, ls="--", label="Cats")]
    fig.legend(handles=hl, ncol=2, loc="upper center", bbox_to_anchor=(0.5, y), fontsize=8, handlelength=2.6)


# S4: prevalence by region with Wilson 95% CI
fig, axs = plt.subplots(5, 3, figsize=(W, 8.6), gridspec_kw=dict(wspace=0.3, hspace=0.22))
for k, (ax, (lab, dp, cp)) in enumerate(zip(axs.ravel(), ALLT)):
    top = 0
    for o, sp, p, mk in ((-0.13, "dog", dp, "o"), (0.13, "cat", cp, "s")):
        if p is None:
            continue
        t = DS[sp]["prev_region"]
        t = t[t.parasite == p].set_index("group").reindex(REG)
        xx = np.arange(len(REG)) + o
        ax.vlines(xx, t.lo, t.hi, color=COL[sp], lw=1.0)
        ax.plot(xx, t.prev, mk, color=COL[sp], ms=3.6, mfc=COL[sp] if sp == "dog" else "white", mew=1.0)
        top = max(top, np.nanmax(t.hi))
    ax.set_ylim(0, top * 1.35)
    ax.set_xticks(range(len(REG)))
    ax.set_xticklabels(list(REG) if k >= 12 else [], rotation=45, ha="right", fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    if k % 3 == 0:
        ax.set_ylabel("Prevalence (%)", fontsize=8)
    tlabel(ax, lab)
    panel(ax, "ABCDEFGHIJKLMNO"[k], dx=-22, dy=2)
    grid(ax)
species_legend(fig, 0.045)
save(fig, "FigureS4_prevalence_by_region")

# S8: prevalence by age (natural splines, as Figure 7), up to 144 months
fig, axs = plt.subplots(5, 3, figsize=(W, 8.6), gridspec_kw=dict(wspace=0.3, hspace=0.22))
for k, (ax, (lab, dp, cp)) in enumerate(zip(axs.ravel(), ALLT)):
    pm = hm = 0
    for sp, p in (("dog", dp), ("cat", cp)):
        if p is None or p not in AGE_RES[sp]["age_curves"]:
            continue
        a_, b_ = curve(ax, sp, p)
        pm, hm = max(pm, a_), max(hm, b_)
    ax.set_ylim(0, min(hm, 2 * pm) * 1.3)  # wide CI at the extremes of age are cut at the top of the panel
    age_axis(ax, label=k >= 12)
    if k < 12:
        ax.set_xticklabels([])
    ax.tick_params(axis="y", labelsize=7)
    if k % 3 == 0:
        ax.set_ylabel("Prevalence (%)", fontsize=8)
    tlabel(ax, lab)
    panel(ax, "ABCDEFGHIJKLMNO"[k], dx=-22, dy=2)
    grid(ax)
species_legend(fig, 0.06)
save(fig, "FigureS8_age_curves")
