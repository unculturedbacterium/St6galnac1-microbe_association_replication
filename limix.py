MIXED MODEL WITH GRM; categorical genotype + explicit DOMINANCE decomposition
glimix_core direct interface

Phenotype : microbe_abundance
Fixed     : sex, cohort, genotype
Random    : GRM (σ²_g), Cage (σ²_cage), Noise (σ²_e)


import numpy as np
import pandas as pd
import struct
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2 as chi2_dist
from scipy.optimize import minimize_scalar

from numpy_sugar.linalg import economic_qs
from glimix_core.lmm import LMM


#GRM reader
def read_gcta_grm(grm_prefix):
    print(f"Reading GRM files with prefix: {grm_prefix}")
    grm_id = pd.read_csv(grm_prefix + ".grm.id", sep="\t", header=None,
                         names=["FID", "IID"], dtype=str)
    n      = len(grm_id)
    n_vals = n * (n + 1) // 2
    with open(grm_prefix + ".grm.bin", "rb") as fh:
        grm_vals = np.array(struct.unpack(f"{n_vals}f", fh.read(n_vals * 4)),
                            dtype=np.float64)
    G = np.zeros((n, n))
    G[np.tril_indices(n)] = grm_vals
    G = G + G.T - np.diag(np.diag(G))
    print(f"  GRM: {n}x{n}, range [{G.min():.4f}, {G.max():.4f}]")
    return G, grm_id["IID"].tolist()


def norm_K(K):
    d = np.diag(K).mean()
    return K / d if d > 0 else K


# two-random-effect LMM (GRM + cage)
def fit_lmm_two_re(y, M, K_grm, K_cage, label="", restricted=True):
    print(f"  [{label}] fitting (cols={M.shape[1]}, REML={restricted})")
    K_grm_n  = norm_K(K_grm)  + 1e-4 * np.eye(len(y))
    K_cage_n = norm_K(K_cage) + 1e-4 * np.eye(len(y))

    def lml_at_w(w):
        w = float(np.clip(w, 1e-6, 1 - 1e-6))
        QS  = economic_qs(w * K_grm_n + (1 - w) * K_cage_n)
        lmm = LMM(y, M, QS, restricted=restricted)
        lmm.fit(verbose=False)
        return lmm.lml(), lmm

    best_lml, best_lmm, best_w = -np.inf, None, 0.5
    for w in np.linspace(0.05, 0.95, 19):
        lml, lmm = lml_at_w(w)
        if lml > best_lml:
            best_lml, best_lmm, best_w = lml, lmm, w

    lo, hi = max(0.001, best_w - 0.1), min(0.999, best_w + 0.1)
    opt = minimize_scalar(lambda w: -lml_at_w(w)[0], bounds=(lo, hi),
                          method="bounded", options={"xatol": 1e-4})
    lml_r, lmm_r = lml_at_w(opt.x)
    if lml_r > best_lml:
        best_lml, best_lmm, best_w = lml_r, lmm_r, float(opt.x)

    s, delta, w = best_lmm.scale, best_lmm.delta, best_w
    sg = s * (1 - delta) * w
    sc = s * (1 - delta) * (1 - w)
    se = s * delta
    total = sg + sc + se
    return {"lml": best_lml, "sigma2_g": sg, "sigma2_cage": sc, "sigma2_e": se,
            "total_var": total, "h2": sg / total, "beta": np.array(best_lmm.beta),
            "w_opt": w, "lmm": best_lmm}


# LOAD DATA
print("\n" + "=" * 70 + "\nLOADING DATA\n" + "=" * 70)
data = pd.read_csv("microbe_abundance.csv", dtype={"rfid": str})
data = data.dropna(subset=["rfid", "sex", "cohort", "cage",
                           "microbe_abundance", "genotype"])

# Order genotype biologically: homo_REF (ref) -> HET -> homo_ALT.
def pick(patterns, levels):
    for p in patterns:
        for lv in levels:
            if p.lower() in lv.lower():
                return lv
    return None

levels   = list(data["genotype"].unique())
homo_ref = pick(["homo_ref", "homref", "ref/ref", "0/0"], levels) or levels[0]
het      = pick(["het", "ref/alt", "0/1", "1/0"], levels)
homo_alt = pick(["homo_alt", "homalt", "alt/alt", "1/1"], levels)
geno_order = [g for g in [homo_ref, het, homo_alt] if g] + \
             [g for g in levels if g not in (homo_ref, het, homo_alt)]

data["genotype"] = pd.Categorical(data["genotype"], categories=geno_order,
                                   ordered=True)
print(f"Genotype order (reference first): {geno_order}")
print("\n=== MICROBE ABUNDANCE BY GENOTYPE ===")
print(data.groupby("genotype", observed=True)["microbe_abundance"]
          .agg(n="count", mean="mean", sd="std", median="median"))

# GRM
print("\n" + "=" * 70 + "\nLOADING GRM\n" + "=" * 70)
try:
    G_full, grm_ids = read_gcta_grm("my_subset_no_chr10")
    use_grm = True
except FileNotFoundError as e:
    print(f"WARNING: {e} — using identity matrix (no GRM file present)")
    use_grm = False

# ALIGN 
print("\n" + "=" * 70 + "\nALIGNING SAMPLES\n" + "=" * 70)
if use_grm:
    common = list(set(data["rfid"]) & set(grm_ids))
    if not common:
        raise ValueError("No matching IDs between phenotype and GRM!")
    data_m   = data.set_index("rfid").loc[common].reset_index()
    order    = [grm_ids.index(s) for s in common]
    G_sub    = G_full[np.ix_(order, order)]
else:
    data_m = data.copy()
    G_sub  = np.eye(len(data_m))
data_m["genotype"] = pd.Categorical(data_m["genotype"], categories=geno_order,
                                    ordered=True)
n = len(data_m)
print(f"Final dataset: {n} individuals")

# CAGE COVARIANCE 
cage_enc = LabelEncoder().fit_transform(data_m["cage"].values)
Z_cage   = np.zeros((n, cage_enc.max() + 1))
Z_cage[np.arange(n), cage_enc] = 1.0
K_cage   = Z_cage @ Z_cage.T
n_cages  = Z_cage.shape[1]
print(f"Cages: {n_cages}")

# DESIGN MATRICES (genotype as additive + dominance contrasts)
sex_d    = pd.get_dummies(data_m["sex"],    drop_first=True, prefix="sex").values.astype(float)
cohort_d = pd.get_dummies(data_m["cohort"], drop_first=True, prefix="cohort").values.astype(float)
intercept = np.ones((n, 1))

code_a = {homo_ref: -1.0, het: 0.0, homo_alt: 1.0}   # additive
code_d = {homo_ref:  0.0, het: 1.0, homo_alt: 0.0}   # dominance (het indicator)
a = data_m["genotype"].map(code_a).to_numpy(dtype=float).reshape(-1, 1)
d = data_m["genotype"].map(code_d).to_numpy(dtype=float).reshape(-1, 1)

nuisance = np.hstack([intercept, sex_d, cohort_d])    # intercept + sex + cohort
M_null = nuisance                                      # no genotype
M_add  = np.hstack([nuisance, a])                      # additive only
M_full = np.hstack([nuisance, a, d])                   # additive + dominance
y = data_m["microbe_abundance"].values.astype(float)
n_nuis = nuisance.shape[1]
print(f"M_null {M_null.shape} | M_add {M_add.shape} | M_full {M_full.shape}")

# 6. FIT
print("\n" + "=" * 70 + "\nFITTING (ML for LRTs, REML for variance components)\n" + "=" * 70)
ml_null = fit_lmm_two_re(y, M_null, G_sub, K_cage, "null_ML", restricted=False)
ml_add  = fit_lmm_two_re(y, M_add,  G_sub, K_cage, "add_ML",  restricted=False)
ml_full = fit_lmm_two_re(y, M_full, G_sub, K_cage, "full_ML", restricted=False)
reml_full = fit_lmm_two_re(y, M_full, G_sub, K_cage, "full_REML", restricted=True)

# LIKELIHOOD-RATIO TESTS
print("\n" + "=" * 70 + "\nLIKELIHOOD-RATIO TESTS\n" + "=" * 70)
def lrt(lml_alt, lml_null, df):
    stat = 2.0 * (lml_alt - lml_null)
    return stat, chi2_dist.sf(max(stat, 0.0), df)

chi_geno, p_geno = lrt(ml_full["lml"], ml_null["lml"], df=2)  # overall genotype
chi_add,  p_add  = lrt(ml_add["lml"],  ml_null["lml"], df=1)  # additive trend
chi_dom,  p_dom  = lrt(ml_full["lml"], ml_add["lml"],  df=1)  # DOMINANCE

print(f"  Overall genotype  : chi2={chi_geno:7.3f}  df=2  p={p_geno:.4e}")
print(f"  Additive trend    : chi2={chi_add:7.3f}  df=1  p={p_add:.4e}")
print(f"  DOMINANCE         : chi2={chi_dom:7.3f}  df=1  p={p_dom:.4e}")

# EFFECT SIZES & DEGREE OF DOMINANCE (from REML full model) 
beta = reml_full["beta"]
b_a  = beta[n_nuis]        # additive coefficient
b_d  = beta[n_nuis + 1]    # dominance deviation (HET - homozygote midpoint)
k    = b_d / abs(b_a) if b_a != 0 else np.nan

if   abs(k) < 0.25: dom_class = "≈ additive (no/weak dominance)"
elif abs(k) < 0.75: dom_class = "partial dominance"
elif abs(k) <= 1.25: dom_class = "≈ complete dominance"
else:                dom_class = "overdominance"
toward = (homo_alt if (b_d * b_a) > 0 else homo_ref) if b_a != 0 else "neither"

print("\n" + "=" * 70 + "\nEFFECT SIZES\n" + "=" * 70)
print(f"  additive  b_a = {b_a:10.3f}  (per-allele half-effect)")
print(f"  dominance b_d = {b_d:10.3f}  (HET minus homozygote midpoint)")
print(f"  degree of dominance k = b_d/|b_a| = {k:.3f}  -> {dom_class}")
print(f"  heterozygote leans toward: {toward}")

# VARIANCE COMPONENTS (REML full)
vc, tv = reml_full, reml_full["total_var"]
print("\n" + "=" * 70 + "\nVARIANCE COMPONENTS (REML)\n" + "=" * 70)
print(f"  σ²_g (GRM)   = {vc['sigma2_g']:.4f} ({100*vc['sigma2_g']/tv:.1f}%)")
print(f"  σ²_cage      = {vc['sigma2_cage']:.4f} ({100*vc['sigma2_cage']/tv:.1f}%)")
print(f"  σ²_e (noise) = {vc['sigma2_e']:.4f} ({100*vc['sigma2_e']/tv:.1f}%)")
print(f"  h² (SNP)     = {vc['h2']:.4f}")

# ADJUSTED VALUES: remove sex+cohort+random effects, keep genotype 
K_grm_n  = norm_K(G_sub)  + 1e-4 * np.eye(n)
K_cage_n = norm_K(K_cage) + 1e-4 * np.eye(n)
K_comb   = vc["w_opt"] * K_grm_n + (1 - vc["w_opt"]) * K_cage_n
s, delta = reml_full["lmm"].scale, reml_full["lmm"].delta
V        = s * (1 - delta) * K_comb + s * delta * np.eye(n)
resid    = y - M_full @ beta
u_hat    = s * (1 - delta) * K_comb @ np.linalg.solve(V, resid)   # BLUP

nuis_fit = nuisance @ beta[:n_nuis]            # sex + cohort + intercept
y_adj    = y - (nuis_fit - beta[0]) - u_hat    # keep intercept + genotype + noise
data_m["y_adj"] = y_adj

adj_means = (data_m.groupby("genotype", observed=True)["y_adj"].mean()
             .reset_index().rename(columns={"y_adj": "adjusted_mean"}))
midpoint = 0.5 * (adj_means.loc[adj_means.genotype == homo_ref, "adjusted_mean"].values[0] +
                  adj_means.loc[adj_means.genotype == homo_alt, "adjusted_mean"].values[0])
print("\nAdjusted genotype means:")
print(adj_means.to_string(index=False))
print(f"Homozygote midpoint (additive expectation for HET): {midpoint:.1f}")

# PLOT
palette = {homo_ref: "#4C72B0", het: "#55A868", homo_alt: "#C44E52"}
fig, ax = plt.subplots(figsize=(8.2, 6))
pos = list(range(1, len(geno_order) + 1))

bp = ax.boxplot([data_m.loc[data_m.genotype == g, "y_adj"].values for g in geno_order],
                positions=pos, patch_artist=True, widths=0.5,
                medianprops=dict(color="black", linewidth=2),
                whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2),
                showfliers=False)
for patch, g in zip(bp["boxes"], geno_order):
    patch.set_facecolor(palette.get(g, "#808080")); patch.set_alpha(0.55)

rng = np.random.default_rng(42)
for p_, g in zip(pos, geno_order):
    v = data_m.loc[data_m.genotype == g, "y_adj"].values
    ax.scatter(p_ + rng.uniform(-0.16, 0.16, v.size), v, color=palette.get(g, "#808080"),
               alpha=0.8, s=28, edgecolors="white", linewidths=0.4, zorder=3)
for p_, g in zip(pos, geno_order):
    m = adj_means.loc[adj_means.genotype == g, "adjusted_mean"].values[0]
    ax.scatter(p_, m, marker="D", s=90, color="white",
               edgecolors=palette.get(g, "#808080"), linewidths=2, zorder=5)

# Additive expectation line: connect the two homozygote means; HET deviation = dominance
ax.plot([pos[0], pos[-1]],
        [adj_means.loc[adj_means.genotype == homo_ref, "adjusted_mean"].values[0],
         adj_means.loc[adj_means.genotype == homo_alt, "adjusted_mean"].values[0]],
        "--", color="grey", linewidth=1.4, zorder=2, label="Additive expectation")
het_mean = adj_means.loc[adj_means.genotype == het, "adjusted_mean"].values[0]
het_pos  = pos[geno_order.index(het)]
ax.annotate("", xy=(het_pos, het_mean), xytext=(het_pos, midpoint),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.6))
ax.text(het_pos + 0.12, 0.5 * (het_mean + midpoint),
        f"dominance\nb_d = {b_d:.0f}\nk = {k:.2f}", fontsize=9, va="center")

ax.legend(handles=[Line2D([0], [0], marker="D", color="w", markerfacecolor="white",
                          markeredgecolor="black", markersize=9, label="Adjusted mean"),
                   Line2D([0], [0], color="grey", ls="--", label="Additive expectation")],
          loc="upper right", framealpha=0.85)
ax.set_xticks(pos); ax.set_xticklabels(geno_order, fontsize=11)
ax.set_xlabel("Genotype", fontsize=12)
ax.set_ylabel("Microbe abundance (adjusted)", fontsize=12)
ax.set_title("Microbe abundance by genotype — dominance view\n"
             "Categorical genotype; GRM + cage as random effects",
             fontweight="bold", fontsize=11)
txt = (f"genotype p = {p_geno:.2e}\ndominance p = {p_dom:.2e}\n{dom_class}"
       if p_geno < 0.001 else
       f"genotype p = {p_geno:.3f}\ndominance p = {p_dom:.3f}\n{dom_class}")
ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha="left", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="lightyellow",
                  edgecolor="grey", alpha=0.9))
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("dominance_microbe_abundance.png", dpi=300)
plt.close()
print("\nSaved: dominance_microbe_abundance.png")

# EXPORT
pd.DataFrame([
    {"test": "Overall genotype", "df": 2, "chi2": chi_geno, "p_value": p_geno},
    {"test": "Additive trend",   "df": 1, "chi2": chi_add,  "p_value": p_add},
    {"test": "Dominance",        "df": 1, "chi2": chi_dom,  "p_value": p_dom},
]).to_csv("dominance_lrt_results.csv", index=False)

pd.DataFrame([{"additive_b_a": b_a, "dominance_b_d": b_d, "degree_of_dominance_k": k,
               "classification": dom_class, "het_leans_toward": toward,
               "homozygote_midpoint": midpoint}]).to_csv("dominance_effects.csv", index=False)

pd.DataFrame({"component": ["GRM", "Cage", "Residual"],
              "variance": [vc["sigma2_g"], vc["sigma2_cage"], vc["sigma2_e"]],
              "proportion": [vc["sigma2_g"]/tv, vc["sigma2_cage"]/tv, vc["sigma2_e"]/tv]
              }).to_csv("dominance_variance_components.csv", index=False)

adj_means.to_csv("dominance_genotype_means.csv", index=False)
print("Saved: dominance_lrt_results.csv, dominance_effects.csv, "
      "dominance_variance_components.csv, dominance_genotype_means.csv")
print("\n" + "=" * 70 + "\nDONE\n" + "=" * 70)
