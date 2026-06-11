"""
MIXED MODEL ANALYSIS WITH GRM — glimix_core direct interface (v3)

Phenotype : microbe_abundance
Fixed effects  : genotype, sex, cohort
Random effects : GRM  (σ²_g,    separately estimated)
                 Cage (σ²_cage, separately estimated)
                 Noise (σ²_e,   separately estimated)
"""

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


# FUNCTION: Read GCTA Binary GRM
def read_gcta_grm(grm_prefix):
    print(f"Reading GRM files with prefix: {grm_prefix}")
    id_file  = grm_prefix + ".grm.id"
    bin_file = grm_prefix + ".grm.bin"
    grm_id   = pd.read_csv(id_file, sep="\t", header=None,
                            names=["FID", "IID"], dtype=str)
    n        = len(grm_id)
    n_vals   = n * (n + 1) // 2
    with open(bin_file, "rb") as fh:
        grm_vals = np.array(struct.unpack(f"{n_vals}f", fh.read(n_vals * 4)),
                            dtype=np.float64)
    G = np.zeros((n, n), dtype=np.float64)
    G[np.tril_indices(n)] = grm_vals
    G = G + G.T - np.diag(np.diag(G))
    print(f"  GRM: {n}x{n}, range [{G.min():.4f}, {G.max():.4f}]")
    return G, grm_id["IID"].tolist()


def norm_K(K):
    "Normalise K to unit mean diagonal."
    d = np.diag(K).mean()
    return K / d if d > 0 else K


def fit_lmm_two_re(y, M, K_grm, K_cage, label="", restricted=True):
 
    print(f"\n  [{label}] Fitting LMM with separate GRM + Cage random effects")

    K_grm_n  = norm_K(K_grm)  + 1e-4 * np.eye(len(y))
    K_cage_n = norm_K(K_cage) + 1e-4 * np.eye(len(y))

    def lml_at_w(w):
        w = float(np.clip(w, 1e-6, 1 - 1e-6))
        K_comb = w * K_grm_n + (1 - w) * K_cage_n
        QS     = economic_qs(K_comb)
        lmm    = LMM(y, M, QS, restricted=restricted)
        lmm.fit(verbose=False)
        return lmm.lml(), lmm

    # Coarse grid search
    best_lml, best_lmm, best_w = -np.inf, None, 0.5
    for w in np.linspace(0.05, 0.95, 19):
        lml, lmm = lml_at_w(w)
        if lml > best_lml:
            best_lml, best_lmm, best_w = lml, lmm, w

    # Scalar refinement around best grid point
    lo = max(0.001, best_w - 0.1)
    hi = min(0.999, best_w + 0.1)
    opt = minimize_scalar(lambda w: -lml_at_w(w)[0],
                          bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-4})
    lml_r, lmm_r = lml_at_w(opt.x)
    if lml_r > best_lml:
        best_lml, best_lmm, best_w = lml_r, lmm_r, float(opt.x)

    # Recover variance components from glimix_core parameterisation
    s     = best_lmm.scale
    delta = best_lmm.delta
    w     = best_w

    sg    = s * (1 - delta) * w          # σ²_g
    sc    = s * (1 - delta) * (1 - w)    # σ²_cage
    se    = s * delta                     # σ²_e
    total = sg + sc + se

    print(f"    w_opt (GRM weight)  = {w:.4f}")
    print(f"    σ²_g    (GRM)       = {sg:.6f}  ({100*sg/total:.1f}%)")
    print(f"    σ²_cage (Cage)      = {sc:.6f}  ({100*sc/total:.1f}%)")
    print(f"    σ²_e    (Noise)     = {se:.6f}  ({100*se/total:.1f}%)")
    print(f"    LML                 = {best_lml:.4f}")

    return {"lml": best_lml, "sigma2_g": sg, "sigma2_cage": sc,
            "sigma2_e": se, "total_var": total, "h2": sg / total,
            "beta": np.array(best_lmm.beta), "w_opt": w, "lmm": best_lmm}


# 1. LOAD PHENOTYPE DATA
print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

data = pd.read_csv("microbe_abundance.csv", dtype={"rfid": str})
data = data.dropna(subset=["rfid", "sex", "cohort", "cage",
                            "microbe_abundance", "genotype"])
data["rfid"] = data["rfid"].astype(str)

ref_candidates = [g for g in data["genotype"].unique() if "ref" in g.lower()]
ref_level = ref_candidates[0] if ref_candidates else data["genotype"].unique()[0]
print(f"Reference genotype: {ref_level}")

print("\n=== MICROBE ABUNDANCE BY GENOTYPE ===")
print(data.groupby("genotype")["microbe_abundance"].agg(
    n="count", mean="mean", sd="std", median="median"))

# 2. LOAD GRM
print("\n" + "="*70)
print("LOADING GRM")
print("="*70)

try:
    G_full, grm_ids = read_gcta_grm("my_subset_no_chr10")
    use_grm = True
except FileNotFoundError as e:
    print(f"WARNING: {e} — using identity matrix")
    use_grm = False

# 3. ALIGN SAMPLES
print("\n" + "="*70)
print("ALIGNING SAMPLES")
print("="*70)

if use_grm:
    common_ids = list(set(data["rfid"]) & set(grm_ids))
    print(f"Phenotype : {data['rfid'].nunique()} | GRM: {len(grm_ids)} | Common: {len(common_ids)}")
    if len(common_ids) == 0:
        raise ValueError("No matching IDs!")
    data_matched = data.set_index("rfid").loc[common_ids].reset_index()
    grm_order    = [grm_ids.index(s) for s in common_ids]
    G_sub        = G_full[np.ix_(grm_order, grm_order)]
else:
    data_matched = data.copy()
    common_ids   = data_matched["rfid"].tolist()
    G_sub        = np.eye(len(common_ids))

n = len(common_ids)
print(f"Final dataset: {n} individuals")


# 4. CAGE COVARIANCE  K_cage = Z_cage @ Z_cage.T
le_cage  = LabelEncoder()
cage_enc = le_cage.fit_transform(data_matched["cage"].values)
n_cages  = len(le_cage.classes_)
Z_cage   = np.zeros((n, n_cages), dtype=np.float64)
Z_cage[np.arange(n), cage_enc] = 1.0
K_cage   = Z_cage @ Z_cage.T
print(f"\nCages: {n_cages},  K_cage: {K_cage.shape}")

# 5. COVARIATE MATRICES
sex_dummies      = pd.get_dummies(data_matched["sex"],      drop_first=True, prefix="sex")
cohort_dummies   = pd.get_dummies(data_matched["cohort"],   drop_first=True, prefix="cohort")
genotype_dummies = pd.get_dummies(data_matched["genotype"], drop_first=True, prefix="genotype")

intercept = np.ones((n, 1))
M_null = np.hstack([intercept, sex_dummies.values,
                    cohort_dummies.values]).astype(np.float64)
M_full = np.hstack([intercept, sex_dummies.values,
                    cohort_dummies.values,
                    genotype_dummies.values]).astype(np.float64)
y = data_matched["microbe_abundance"].values.astype(np.float64)

print(f"\nM_null: {M_null.shape}  |  M_full: {M_full.shape}")

# 6. FIT NULL AND FULL MODELS
print("\n" + "="*70)
print("FITTING MODELS (ML for LRT, REML for Variance Components)")
print("="*70)

# 1. Fit with ML (restricted=False) for a statistically valid LRT
res_null_ml = fit_lmm_two_re(y, M_null, G_sub, K_cage, label="null_ML", restricted=False)
res_full_ml = fit_lmm_two_re(y, M_full, G_sub, K_cage, label="full_ML", restricted=False)

# 2. Fit Full Model with REML (restricted=True) for unbiased variance components
res_full_reml = fit_lmm_two_re(y, M_full, G_sub, K_cage, label="full_REML", restricted=True)
res_null_reml = fit_lmm_two_re(y, M_null, G_sub, K_cage, label="null_REML", restricted=True)

# 7. LIKELIHOOD RATIO TEST (Using ML models, df=1)
print("\n" + "="*70)
print("GLOBAL GENOTYPE P-VALUE (LRT on ML Likelihoods, df=1)")
print("="*70)

lml_null       = res_null_ml["lml"]
lml_full       = res_full_ml["lml"]
lrt_stat       = 2.0 * (lml_full - lml_null)
lrt_pval       = chi2_dist.sf(lrt_stat, df=1)

print(f"  LML null (ML) : {lml_null:.4f}")
print(f"  LML full (ML) : {lml_full:.4f}")
print(f"  χ²            : {lrt_stat:.4f}  (df=1)")
print(f"  p-value       : {lrt_pval:.4e}")

sig_label = ("*** HIGHLY SIGNIFICANT" if lrt_pval < 0.001 else
             "** VERY SIGNIFICANT"    if lrt_pval < 0.01  else
             "* SIGNIFICANT"          if lrt_pval < 0.05  else
             "NOT SIGNIFICANT")
print(f"  {sig_label}")

pval_label = (f"p = {lrt_pval:.2e}"           if lrt_pval < 0.001 else
              f"p = {lrt_pval:.4f}"            if lrt_pval < 0.05  else
              f"p = {lrt_pval:.3f} (n.s.)")

# 8. VARIANCE COMPONENTS SUMMARY (Using REML model)
print("\n" + "="*70)
print("VARIANCE COMPONENTS (Full Model - REML)")
print("="*70)

vc = res_full_reml
tv = vc["total_var"]
print(f"  σ²_g    (GRM)   = {vc['sigma2_g']:.6f}  ({100*vc['sigma2_g']/tv:.1f}%)")
print(f"  σ²_cage (Cage)  = {vc['sigma2_cage']:.6f}  ({100*vc['sigma2_cage']/tv:.1f}%)")
print(f"  σ²_e    (Noise) = {vc['sigma2_e']:.6f}  ({100*vc['sigma2_e']/tv:.1f}%)")
print(f"  SNP heritability h² = {vc['h2']:.4f}")


# 9. ADJUSTED PHENOTYPE VALUES
beta_null   = res_null_reml["beta"]
fitted_null = M_null[:, :len(beta_null)] @ beta_null
y_adjusted  = (y - fitted_null) + y.mean()

data_matched = data_matched.copy()
data_matched["y_adjusted"] = y_adjusted

adj_means = (data_matched.groupby("genotype")["y_adjusted"]
             .mean().reset_index()
             .rename(columns={"y_adjusted": "adjusted_mean"}))
print("\nAdjusted genotype means:")
print(adj_means.to_string(index=False))

# 10. VISUALISATION
subtitle  = "glimix_core LMM: GRM + Cage as separate random effects"
all_genos = data_matched["genotype"].unique().tolist()

def find_geno(patterns, candidates):
    for pat in patterns:
        for c in candidates:
            if pat.lower() in c.lower():
                return c
    return None

homo_ref = find_geno(["homo_ref","homref","ref/ref","0/0"], all_genos) or ref_level
het      = find_geno(["het","ref/alt","0/1","1/0"], all_genos)
homo_alt = find_geno(["homo_alt","homalt","alt/alt","1/1"], all_genos)

geno_order = [g for g in [homo_ref, het, homo_alt] if g is not None]
for g in all_genos:
    if g not in geno_order:
        geno_order.append(g)

palette    = ["#4C72B0", "#55A868", "#C44E52", "#808080"]
box_colors = {g: palette[min(i, len(palette)-1)] for i, g in enumerate(geno_order)}

fig, ax  = plt.subplots(figsize=(8, 6))
positions = list(range(1, len(geno_order) + 1))

bp = ax.boxplot(
    [data_matched.loc[data_matched["genotype"] == g, "y_adjusted"].values
     for g in geno_order],
    positions=positions, patch_artist=True, widths=0.45,
    medianprops=dict(color="black", linewidth=2),
    whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2),
    showfliers=False)

for patch, geno in zip(bp["boxes"], geno_order):
    patch.set_facecolor(box_colors[geno])
    patch.set_alpha(0.6)

rng = np.random.default_rng(42)
for pos, geno in zip(positions, geno_order):
    vals   = data_matched.loc[data_matched["genotype"] == geno, "y_adjusted"].values
    jitter = rng.uniform(-0.15, 0.15, size=len(vals))
    ax.scatter(pos + jitter, vals, color=box_colors[geno],
               alpha=0.85, s=30, edgecolors="white", linewidths=0.4, zorder=3)

for pos, geno in zip(positions, geno_order):
    adj_m = adj_means.loc[adj_means["genotype"] == geno, "adjusted_mean"].values[0]
    ax.scatter(pos, adj_m, marker="D", s=80,
               color="white", edgecolors=box_colors[geno], linewidths=2, zorder=5)

ax.legend(handles=[Line2D([0],[0], marker="D", color="w",
                          markerfacecolor="white", markeredgecolor="black",
                          markersize=9, label="Adjusted mean")],
          loc="upper right", framealpha=0.8)
ax.set_xticks(positions)
ax.set_xticklabels(geno_order, fontsize=11)
ax.set_xlabel("Genotype", fontsize=12)
ax.set_ylabel("Microbe Abundance (adjusted)", fontsize=12)
ax.set_title(f"Microbe Abundance by Genotype\n{subtitle}",
             fontweight="bold", fontsize=11)
ax.text(0.97, 0.97, pval_label, transform=ax.transAxes,
        ha="right", va="top", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                  edgecolor="grey", alpha=0.9))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("limix_microbe_abundance_with_grm_df1.png", dpi=300)
print("\nSaved: limix_microbe_abundance_with_grm_df1.png")
plt.close()

# 11. EXPORT
pd.DataFrame([{
    "test": "Global genotype effect (LRT, df=1)",
    "lml_null": lml_null, "lml_full": lml_full,
    "lrt_statistic": lrt_stat, "df": 1,
    "p_value": lrt_pval, "significant": lrt_pval < 0.05
}]).to_csv("limix_genotype_lrt_results_df1.csv", index=False)

pd.DataFrame({
    "component" : ["GRM (σ²_g)", "Cage (σ²_cage)", "Residual (σ²_e)"],
    "variance"  : [vc["sigma2_g"], vc["sigma2_cage"], vc["sigma2_e"]],
    "proportion": [vc["sigma2_g"]/tv, vc["sigma2_cage"]/tv, vc["sigma2_e"]/tv]
}).to_csv("limix_variance_components_df1.csv", index=False)

adj_means.to_csv("limix_genotype_means_df1.csv", index=False)

with open("limix_model_summary_df1.txt", "w") as fh:
    fh.write("="*70 + "\n")
    fh.write("MIXED MODEL — SEPARATE GRM + CAGE (glimix_core LMM)\n")
    fh.write("Matches: Tonnelé et al. 2025, Nature Communications\n")
    fh.write("="*70 + "\n\n")
    fh.write(f"n={n}, cages={n_cages}, genotypes={data_matched['genotype'].nunique()}\n\n")
    fh.write("Variance Components (Full Model):\n")
    fh.write(f"  σ²_g    = {vc['sigma2_g']:.6f} ({100*vc['sigma2_g']/tv:.1f}%)\n")
    fh.write(f"  σ²_cage = {vc['sigma2_cage']:.6f} ({100*vc['sigma2_cage']/tv:.1f}%)\n")
    fh.write(f"  σ²_e    = {vc['sigma2_e']:.6f} ({100*vc['sigma2_e']/tv:.1f}%)\n")
    fh.write(f"  h²      = {vc['h2']:.4f}\n\n")
    fh.write(f"LRT: χ²={lrt_stat:.4f}, df=1, p={lrt_pval:.4e}\n\n")
    fh.write("Adjusted Genotype Means:\n")
    fh.write(adj_means.to_string(index=False) + "\n")

print("Saved: limix_genotype_lrt_results_df1.csv")
print("Saved: limix_variance_components_df1.csv")
print("Saved: limix_genotype_means_df1.csv")
print("Saved: limix_model_summary_df1.txt")

print("\n" + "="*70)
print("COMPLETE")
print("="*70)
print(f"✓ Separate GRM + Cage random effects via glimix_core LMM")
print(f"✓ Global genotype p-value (LRT, df=1): {lrt_pval:.4e}")
print(f"✓ SNP heritability h²: {vc['h2']:.4f}")
