import numpy as np
import pandas as pd
from pandas_plink import read_plink1_bin


PLINK_PREFIX = "/tscc/projects/ps-palmer/gwas/databases/rounds/r11.2.1"

TARGET_CHR = "10"
TARGET_POS = 102471774

print(f"Loading PLINK files from: {PLINK_PREFIX}")

G = read_plink1_bin(
    PLINK_PREFIX + ".bed",
    PLINK_PREFIX + ".bim",
    PLINK_PREFIX + ".fam",
    verbose=False,
)

print(G)

# G dimensions are usually: sample x variant
# Variant metadata are stored in G.variant, G.chrom, G.pos, G.a0, G.a1, etc.

chrom = G.chrom.astype(str).values
pos = G.pos.astype(int).values

target_mask = (
    np.char.replace(chrom.astype(str), "chr", "") == str(TARGET_CHR)
) & (pos == TARGET_POS)

idx = np.where(target_mask)[0]

if len(idx) == 0:
    raise RuntimeError(f"No SNP found at {TARGET_CHR}:{TARGET_POS}")

if len(idx) > 1:
    print(f"WARNING: found {len(idx)} variants at {TARGET_CHR}:{TARGET_POS}; reporting all.")

print(f"\nFound {len(idx)} variant(s) at {TARGET_CHR}:{TARGET_POS}")


rows = []

for j in idx:
    snp = G.variant.values[j] if "variant" in G.coords else f"variant_index_{j}"

    a0 = G.a0.values[j] if "a0" in G.coords else "NA"
    a1 = G.a1.values[j] if "a1" in G.coords else "NA"

    # Genotypes are allele counts for one allele, typically values 0/1/2 with NaN missing.
    g = G[:, j].compute().values.astype(float)

    n_nonmissing = np.sum(~np.isnan(g))
    n_missing = np.sum(np.isnan(g))

    if n_nonmissing == 0:
        allele_freq = np.nan
        maf = np.nan
        n0 = n1 = n2 = 0
    else:
        allele_freq = np.nansum(g) / (2.0 * n_nonmissing)
        maf = min(allele_freq, 1.0 - allele_freq)

        n0 = int(np.sum(g == 0))
        n1 = int(np.sum(g == 1))
        n2 = int(np.sum(g == 2))

    rows.append(
        {
            "CHR": TARGET_CHR,
            "POS": TARGET_POS,
            "SNP": snp,
            "A0": a0,
            "A1": a1,
            "N_NONMISSING": int(n_nonmissing),
            "N_MISSING": int(n_missing),
            "N_GENO_0": n0,
            "N_GENO_1": n1,
            "N_GENO_2": n2,
            "COUNTED_ALLELE_FREQ": allele_freq,
            "MAF": maf,
        }
    )

summary = pd.DataFrame(rows)

print("\nMAF summary:")
print(summary.to_string(index=False))

summary.to_csv("maf_10_102471774.tsv", sep="\t", index=False)
print("\nWrote: maf_10_102471774.tsv")
