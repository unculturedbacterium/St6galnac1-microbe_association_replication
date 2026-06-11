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
        w      = float(np.clip(w, 1e-6, 1 - 1e-6))
        K_comb = w * K_grm_n + (1 - w) * K_cage_n
        QS     = economic_qs(K_comb)
        lmm    = LMM(y, M, QS, restricted=restricted)
        lmm.fit(verbose=False)
        return lmm.lml(), lmm

    best_lml, best_lmm, best_w = -np.inf, None, 0.5
    for w in np.linspace(0.05, 0.95, 19):
        lml, lmm = lml_at_w(w)
        if lml > best_lml:
            best_lml, best_lmm, best_w = lml, lmm, w

    lo  = max(0.001, best_w - 0.1)
    hi  = min(0.999, best_w + 0.1)
    opt = minimize_scalar(lambda w: -lml_at_w(w)[0],
                          bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-4})
    lml_r, lmm_r = lml_at_w(opt.x)
    if lml_r > best_lml:
        best_lml, best_lmm, best_w = lml_r, lmm_r, float(opt.x)

    s     = best_lmm.scale
    delta = best_lmm.delta
    w     = best_w
    sg    = s * (1 - delta) * w
    sc    = s * (1 - delta) * (1 - w)
    se    = s * delta
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

# 4. CAGE COVARIANCE
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

# 6. FIT MODELS
print("\n" + "="*70)
print("FITTING MODELS (ML for LRT, REML for Variance Components)")
print("="*70)

res_null_ml   = fit_lmm_two_re(y, M_null, G_sub, K_cage, label="null_ML",   restricted=False)
res_full_ml   = fit_lmm_two_re(y, M_full, G_sub, K_cage, label="full_ML",   restricted=False)
res_full_reml = fit_lmm_two_re(y, M_full, G_sub, K_cage, label="full_REML", restricted=True)
res_null_reml = fit_lmm_two_re(y, M_null, G_sub, K_cage, label="null_REML", restricted=True)

# 7. LIKELIHOOD RATIO TEST
print("\n" + "="*70)
print("GLOBAL GENOTYPE P-VALUE (LRT on ML Likelihoods, df=1)")
print("="*70)

lml_null = res_null_ml["lml"]
lml_full = res_full_ml["lml"]
lrt_stat = 2.0 * (lml_full - lml_null)
lrt_pval = chi2_dist.sf(lrt_stat, df=1)

print(f"  LML null (ML) : {lml_null:.4f}")
print(f"  LML full (ML) : {lml_full:.4f}")
print(f"  χ²            : {lrt_stat:.4f}  (df=1)")
print(f"  p-value       : {lrt_pval:.4e}")

sig_label = ("*** HIGHLY SIGNIFICANT" if lrt_pval < 0.001 else
             "** VERY SIGNIFICANT"    if lrt_pval < 0.01  else
             "* SIGNIFICANT"          if lrt_pval < 0.05  else
             "NOT SIGNIFICANT")
print(f"  {sig_label}")

pval_label = (f"p = {lrt_pval:.2e}" if lrt_pval < 0.001 else
              f"p = {lrt_pval:.4f}" if lrt_pval < 0.05  else
              f"p = {lrt_pval:.3f} (n.s.)")

# 8. VARIANCE COMPONENTS SUMMARY
print("\n" + "="*70)
print("VARIANCE COMPONENTS (Full Model - REML)")
print("="*70)

vc = res_full_reml
tv = vc["total_var"]
print(f"  σ²_g    (GRM)   = {vc['sigma2_g']:.6f}  ({100*vc['sigma2_g']/tv:.1f}%)")
print(f"  σ²_cage (Cage)  = {vc['sigma2_cage']:.6f}  ({100*vc['sigma2_cage']/tv:.1f}%)")
print(f"  σ²_e    (Noise) = {vc['sigma2_e']:.6f}  ({100*vc['sigma2_e']/tv:.1f}%)")
print(f"  SNP heritability h² = {vc['h2']:.4f}")

# 9. GENOTYPE ORDER HELPER
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

# 10. TRUE RESIDUALS (remove all fixed + random effects, keep genotype signal)
print("\n" + "="*70)
print("COMPUTING TRUE RESIDUALS")
print("="*70)

lmm_full  = res_full_reml["lmm"]
w_opt     = res_full_reml["w_opt"]
beta_full = np.array(lmm_full.beta)

K_grm_n  = norm_K(G_sub)  + 1e-4 * np.eye(n)
K_cage_n = norm_K(K_cage) + 1e-4 * np.eye(n)
K_comb   = w_opt * K_grm_n + (1 - w_opt) * K_cage_n

Xbeta_full = M_full[:, :len(beta_full)] @ beta_full

# BLUP for combined random effect
s_scale  = lmm_full.scale
delta    = lmm_full.delta
su       = s_scale * (1 - delta)
se_var   = s_scale * delta
V        = su * K_comb + se_var * np.eye(n)
Vinv     = np.linalg.solve(V, np.eye(n))
u_hat    = su * K_comb @ Vinv @ (y - Xbeta_full)

epsilon  = y - Xbeta_full - u_hat

# Add back genotype-only fitted effect
n_base_cols = M_null.shape[1]
geno_cols   = M_full[:, n_base_cols:]
geno_betas  = beta_full[n_base_cols:]
geno_fitted = beta_full[0] + geno_cols @ geno_betas

data_matched = data_matched.copy()
data_matched["y_resid_plot"] = geno_fitted + epsilon

adj_means_resid = (
    data_matched.groupby("genotype")["y_resid_plot"]
    .mean().reset_index()
    .rename(columns={"y_resid_plot": "adjusted_mean"})
)
print("\nGenotype means (true residuals + genotype effect):")
print(adj_means_resid.to_string(index=False))

# 11. BOXPLOT (residuals)
palette    = ["#4C72B0", "#55A868", "#C44E52", "#808080"]
box_colors = {g: palette[min(i, len(palette)-1)] for i, g in enumerate(geno_order)}

fig, ax   = plt.subplots(figsize=(8, 6))
positions = list(range(1, len(geno_order) + 1))

bp = ax.boxplot(
    [data_matched.loc[data_matched["genotype"] == g, "y_resid_plot"].values
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
    vals   = data_matched.loc[data_matched["genotype"] == geno, "y_resid_plot"].values
    jitter = rng.uniform(-0.15, 0.15, size=len(vals))
    ax.scatter(pos + jitter, vals, color=box_colors[geno],
               alpha=0.85, s=30, edgecolors="white", linewidths=0.4, zorder=3)

for pos, geno in zip(positions, geno_order):
    m = adj_means_resid.loc[adj_means_resid["genotype"] == geno, "adjusted_mean"].values[0]
    ax.scatter(pos, m, marker="D", s=80,
               color="white", edgecolors=box_colors[geno], linewidths=2, zorder=5)

ax.legend(handles=[Line2D([0],[0], marker="D", color="w",
                          markerfacecolor="white", markeredgecolor="black",
                          markersize=9, label="Adjusted mean")],
          loc="upper right", framealpha=0.8)
ax.set_xticks(positions)
ax.set_xticklabels(geno_order, fontsize=11)
ax.set_xlabel("Genotype", fontsize=12)
ax.set_ylabel("Microbe Abundance\n(residuals + genotype effect)", fontsize=12)
ax.set_title("Microbe Abundance by Genotype\n"
             "Residuals after GRM, cage, sex & cohort removed (BLUP)",
             fontweight="bold", fontsize=11)
ax.text(0.97, 0.97, pval_label, transform=ax.transAxes,
        ha="right", va="top", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                  edgecolor="grey", alpha=0.9))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("limix_microbe_abundance_true_residuals.png", dpi=300)
print("\nSaved: limix_microbe_abundance_true_residuals.png")
plt.close()

# 12. DOMINANCE PLOT (Cui et al. 2023 style)
print("\n" + "="*70)
print("DOMINANCE ANALYSIS (Cui et al. 2023, Genome Biology)")
print("="*70)

def get_mean(geno):
    return adj_means_resid.loc[
        adj_means_resid["genotype"] == geno, "adjusted_mean"].values[0]

mu_ref = get_mean(homo_ref)
mu_het = get_mean(het)
mu_alt = get_mean(homo_alt)

midpoint = (mu_ref + mu_alt) / 2.0
a        = (mu_ref - mu_alt) / 2.0
d        = mu_het - midpoint
da       = d / a if a != 0 else np.nan

dom_type = ("Additive"                 if abs(da) < 0.2  else
            "Partially dominant (REF)" if  0.2 <= da <  0.8 else
            "Dominant (REF)"           if  0.8 <= da <= 1.2 else
            "Over-dominant"            if  da  >  1.2       else
            "Partially dominant (ALT)" if -0.8 <  da <= -0.2 else
            "Dominant (ALT)"           if -1.2 <= da <= -0.8 else
            "Over-dominant (ALT)")

print(f"  mu homo_REF = {mu_ref:.2f}")
print(f"  mu HET      = {mu_het:.2f}")
print(f"  mu homo_ALT = {mu_alt:.2f}")
print(f"  midpoint    = {midpoint:.2f}")
print(f"  a           = {a:.2f}")
print(f"  d           = {d:.2f}")
print(f"  d/a         = {da:.3f}  →  {dom_type}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                         gridspec_kw={"width_ratios": [1.6, 1]})

palette_dom     = {homo_ref: "#4C72B0", het: "#55A868", homo_alt: "#C44E52"}
x_pos           = [0, 1, 2]
x_labels        = ["homo_REF", "HET", "homo_ALT"]
geno_order_plot = [homo_ref, het, homo_alt]

# LEFT: individual points + observed means + additive expectation line
ax  = axes[0]
rng = np.random.default_rng(0)
for xi, geno in zip(x_pos, geno_order_plot):
    vals   = data_matched.loc[data_matched["genotype"] == geno, "y_resid_plot"].values
    jitter = rng.uniform(-0.12, 0.12, size=len(vals))
    ax.scatter(xi + jitter, vals, color=palette_dom[geno],
               alpha=0.45, s=20, edgecolors="none", zorder=2)

ax.plot(x_pos, [mu_ref, midpoint, mu_alt],
        color="grey", linestyle="--", linewidth=1.8,
        label="Additive expectation", zorder=3)

ax.plot(x_pos, [mu_ref, mu_het, mu_alt],
        color="black", linestyle="-", linewidth=2.2, zorder=4,
        marker="D", markersize=10, markerfacecolor="white",
        markeredgewidth=2, label="Observed mean")

ax.annotate("", xy=(1, mu_het), xytext=(1, midpoint),
            arrowprops=dict(arrowstyle="<->", color="firebrick", lw=2.2))
ax.text(1.10, (mu_het + midpoint) / 2,
        f"  d = {d:.0f}", color="firebrick", fontsize=10, va="center")

ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, fontsize=11)
ax.set_ylabel("Microbe Abundance (LMM-adjusted residuals)", fontsize=11)
ax.set_title(f"Observed vs additive expectation\n(d/a = {da:.2f}, {dom_type})",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
ax.text(0.97, 0.97, pval_label, transform=ax.transAxes,
        ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                  edgecolor="grey", alpha=0.9))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# RIGHT: dominance degree scale
ax2 = axes[1]
ax2.axhline(0, color="lightgrey", linewidth=0.8)

zone_data = [(-1.5, "Over-dom\n(ALT)"), (-1.0, "Dom\n(ALT)"),
             ( 0.0, "Additive"),        ( 1.0, "Dom\n(REF)"),
             ( 1.5, "Over-dom\n(REF)")]
for cx, lab in zone_data:
    ax2.axvline(cx, color="lightgrey", linewidth=0.5, linestyle=":")
    ax2.text(cx, -0.38, lab, ha="center", va="top", fontsize=7.5, color="grey")

ax2.axvspan(-0.2,  0.2, alpha=0.08, color="grey")
ax2.axvspan( 0.8,  1.2, alpha=0.14, color="#4C72B0")
ax2.scatter([da], [0], s=320, color="firebrick", zorder=5,
            edgecolors="black", linewidths=1.5)
ax2.axvline(da, color="firebrick", linewidth=1.8, alpha=0.6)
ax2.text(da, 0.20, f"d/a = {da:.2f}\n{dom_type}",
         ha="center", fontsize=10, color="firebrick", fontweight="bold")

ax2.set_xlim(-2, 2)
ax2.set_ylim(-0.6, 0.55)
ax2.set_xlabel("Dominance degree (d/a)", fontsize=11)
ax2.set_title("Dominance classification\n(Cui et al. 2023, Genome Biol.)",
              fontsize=10, fontweight="bold")
ax2.set_yticks([])
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)

plt.suptitle("Dominant genetic architecture of microbe abundance QTL\n"
             "Mixed model adjusted for GRM + cage + sex + cohort",
             fontsize=11, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("dominance_plot_cui2023_style.png", dpi=300, bbox_inches="tight")
print("\nSaved: dominance_plot_cui2023_style.png")
plt.close()

# 13. EXPORT
pd.DataFrame([{
    "test": "Global genotype effect (LRT, df=1)",
    "lml_null": lml_null, "lml_full": lml_full,
    "lrt_statistic": lrt_stat, "df": 1,
    "p_value": lrt_pval, "significant": lrt_pval < 0.05
}]).to_csv("limix_genotype_lrt_results.csv", index=False)

pd.DataFrame({
    "component" : ["GRM (σ²_g)", "Cage (σ²_cage)", "Residual (σ²_e)"],
    "variance"  : [vc["sigma2_g"], vc["sigma2_cage"], vc["sigma2_e"]],
    "proportion": [vc["sigma2_g"]/tv, vc["sigma2_cage"]/tv, vc["sigma2_e"]/tv]
}).to_csv("limix_variance_components.csv", index=False)

adj_means_resid.to_csv("limix_genotype_means.csv", index=False)

pd.DataFrame({
    "statistic": ["d/a", "d", "a", "mu_homo_REF", "mu_HET", "mu_homo_ALT",
                  "midpoint", "dom_type"],
    "value":     [da, d, a, mu_ref, mu_het, mu_alt, midpoint, dom_type]
}).to_csv("dominance_statistics.csv", index=False)

with open("limix_model_summary.txt", "w") as fh:
    fh.write("="*70 + "\n")
    fh.write("MIXED MODEL — SEPARATE GRM + CAGE (glimix_core LMM)\n")
    fh.write("Tonnelé et al. 2025, Nature Communications\n")
    fh.write("="*70 + "\n\n")
    fh.write(f"n={n}, cages={n_cages}, genotypes={data_matched['genotype'].nunique()}\n\n")
    fh.write("Variance Components (Full Model, REML):\n")
    fh.write(f"  σ²_g    = {vc['sigma2_g']:.6f} ({100*vc['sigma2_g']/tv:.1f}%)\n")
    fh.write(f"  σ²_cage = {vc['sigma2_cage']:.6f} ({100*vc['sigma2_cage']/tv:.1f}%)\n")
    fh.write(f"  σ²_e    = {vc['sigma2_e']:.6f} ({100*vc['sigma2_e']/tv:.1f}%)\n")
    fh.write(f"  h²      = {vc['h2']:.4f}\n\n")
    fh.write(f"LRT: χ²={lrt_stat:.4f}, df=1, p={lrt_pval:.4e}\n\n")
    fh.write("Dominance (Cui et al. 2023):\n")
    fh.write(f"  d/a = {da:.3f}  →  {dom_type}\n\n")
    fh.write("Genotype Means (true residuals):\n")
    fh.write(adj_means_resid.to_string(index=False) + "\n")

print("\nSaved: limix_genotype_lrt_results.csv")
print("Saved: limix_variance_components.csv")
print("Saved: limix_genotype_means.csv")
print("Saved: dominance_statistics.csv")
print("Saved: limix_model_summary.txt")

print("\n" + "="*70)
print("COMPLETE")
print("="*70)
print(f"  Global genotype p-value (LRT, df=1): {lrt_pval:.4e}")
print(f"  SNP heritability h²: {vc['h2']:.4f}")
print(f"  Dominance degree d/a: {da:.3f}  →  {dom_type}")
