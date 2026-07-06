
# St6galnac1–Microbiome Association Replication

Code for replicating the analysis of host genetic effects on the gut microbiome in heterogeneous stock (HS) outbred rats, focusing on the *St6galnac1* locus and its association with microbial abundance. Includes mixed-model GWAS, GRM construction, microbiome diversity analyses, and publication-ready plotting scripts.

This pipeline was used in:
> Sadegi et al. 2026 "Replication of Host St6galnac1-Microbiome Association Across Distinct Experimental Conditions in Heterogeneous Stock Rats"

---

## Repository Structure

```
St6galnac1-microbe_association_replication/
├── classification.py       # QIIME2 taxonomy classification & sample matching
├── analysis_and_plots.py   # Alpha/beta diversity analyses & taxonomic composition plots
├── limix.py                # LIMIX/glimix_core LMM — genotype association test
├── dominance_all.py        # Dominance analysis (Cui et al. 2023 style)
├── liftover_func.py        # rn7 → GRCr8 liftover + genotype extraction from PLINK
└── maf.py                  # Minor allele frequency computation at a target SNP
```

---

## Analysis Workflow

### 1. Taxonomy Classification (`classification.py`)

Classifies 16S rRNA amplicon sequences using QIIME2 and the SILVA 138.2 database, then merges classified taxonomy with genotype metadata.

**Inputs:**
- `all.seqs.fa` — representative sequences (FASTA)
- `SILVA138.2_SSURef_NR99_uniform_classifier_full-length.qza` — pre-trained SILVA classifier
- `rarefied_feature_table.qza` — rarefied QIIME2 feature table
- `226721_mapping_file.txt` — sample metadata
- `metadata.csv` — sample metadata with `rfid` and `tube_barcode` columns
- `rarefiedtaxonomy.csv` / `rarefiedtaxonomy_silva.csv` — rarefied taxonomy table with `orig_name` column

**Outputs:**
- `taxonomy_silva.qza` — classified taxonomy artifact
- `feature_table_summary_silva_rarefied.qzv` — feature table summary visualization
- `taxonomy_barplot_silva_rarefied.qzv` — taxonomy barplot visualization
- `matched_allrarefied_silva.csv` — metadata merged with rarefied taxonomy, matched on `tube_barcode`

---

### 2. Diversity Analyses & Composition Plots (`analysis_and_plots.py`)

Computes alpha and beta diversity metrics and generates publication-ready figures.

**Alpha diversity:**
- Shannon entropy, Simpson index, observed OTUs
- Boxplots stratified by genotype

**Beta diversity:**
- Aitchison distance (CLR transformation + Euclidean distance)
- PERMANOVA significance test (999 permutations)
- PCA ordination plot colored by genotype

**Taxonomic composition:**
- Stacked bar plot of average relative abundance per cohort

**Inputs:**
- `matched_genotypes_allbiomes_blind.csv` — OTU table with `genotype` column (alpha diversity)
- `matched_allrarefied_genotypes.csv` — rarefied OTU table with `genotype` column (beta diversity)
- `matched_allrarefied_exposures.csv` — rarefied OTU table with `cohort` column (composition plots)

**Dependencies:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-bio`, `scikit-learn`

---

### 3. LIMIX Mixed-Model Association Test (`limix.py`)

Tests for a global genotype effect on microbe abundance using a linear mixed model (LMM) with separate random effects for polygenic background (GRM) and cage environment.

**Model:**

```
y = Xβ + u_g + u_cage + ε

u_g    ~ N(0, σ²_g · K_GRM)
u_cage ~ N(0, σ²_cage · K_cage)
ε      ~ N(0, σ²_e · I)
```

Fixed effects: intercept, sex, cohort, genotype (dummy-coded)

The GRM weight (`w`) is jointly optimized over a grid + scalar refinement. Genotype significance is assessed via a **likelihood ratio test (LRT, df=1)** comparing ML-fitted null and full models. Variance components (σ²_g, σ²_cage, σ²_e, h²) are estimated under **REML**.

**Inputs:**
- `microbe_abundance.csv` — phenotype file with columns: `rfid`, `microbe_abundance`, `genotype`, `sex`, `cohort`, `cage`
- `my_subset_no_chr10.grm.bin` / `.grm.id` — GCTA binary GRM (falls back to identity matrix if absent)

**Outputs:**
- `limix_microbe_abundance_with_grm_df1.png` — boxplot of adjusted microbe abundance by genotype
- `limix_genotype_lrt_results_df1.csv` — LRT statistic and p-value
- `limix_variance_components_df1.csv` — variance component estimates
- `limix_genotype_means_df1.csv` — adjusted genotype means
- `limix_model_summary_df1.txt` — human-readable model summary

**Dependencies:** `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scipy`, `numpy_sugar`, `glimix_core`

---

### 4. Dominance Analysis

Extends the LMM above with a dominance classification following **Cui et al. 2023 (*Genome Biology*)**.

The dominance degree `d/a` is computed from BLUP-corrected genotype means:

```
a   = (μ_homo_REF − μ_homo_ALT) / 2
d   = μ_HET − midpoint
d/a = dominance ratio
```

| d/a range | Classification |
|---|---|
| \|d/a\| < 0.2 | Additive |
| 0.2 – 0.8 | Partially dominant (REF) |
| 0.8 – 1.2 | Dominant (REF) |
| > 1.2 | Over-dominant |
| Negative equivalents | ALT-direction counterparts |

Uses **BLUP** to remove polygenic background and cage effects before computing genotype means, ensuring residuals reflect only the genotype signal at the tested locus.

**Inputs:** Same as `limix.py`

**Outputs:**
- `limix_microbe_abundance_true_residuals.png` — boxplot with BLUP-corrected residuals
- `dominance_plot_cui2023_style.png` — two-panel figure: observed vs additive expectation + dominance scale
- `dominance_statistics.csv` — a, d, d/a, genotype means
- `limix_genotype_lrt_results.csv`, `limix_variance_components.csv`, `limix_genotype_means.csv`
- `limix_model_summary.txt`

---

### 5. Liftover & Genotype Extraction (`liftover_func.py`)

Extracts genotypes for a target SNP from a large PLINK dataset, lifting coordinates from **rn7 → GRCr8** (via a custom chain file), then merging with sample metadata.

```python
from liftover_func import get_genotypes

get_genotypes(
    chrom=4,
    pos=70_834_123,          # rn7 coordinate
    output_file="metadata_ch4_genotypes.csv",
    metadata="metadata_all.csv",
    plink_prefix="/path/to/r11.2.1",
    chain_file="/path/to/rn7ToGCF_036323735.1.over.chain.gz",
)
```

Genotype codes returned: `Homo_REF`, `HET`, `Homo_ALT`, `NA`.

**Inputs:**
- rn7 chromosome and position
- `metadata_all.csv` — sample metadata with `rfid` column
- PLINK binary files (`.bed`, `.bim`, `.fam`)
- rn7→GRCr8 chain file

**Output:** CSV with all metadata columns plus a `genotype` column.

**Dependencies:** `pandas`, `numpy`, `pandas_plink`, `pyliftover`

---

### 6. Minor Allele Frequency (`maf.py`)

Computes MAF and genotype counts at a single target SNP from a PLINK dataset. Useful for quality-checking variant frequency before association testing.

Edit the constants at the top of the file to change the target:

```python
PLINK_PREFIX = "/tscc/projects/ps-palmer/gwas/databases/rounds/r11.2.1"
TARGET_CHR   = "10"
TARGET_POS   = 102471774
```

**Output:** `maf_10_102471774.tsv` — tab-separated table with N_NONMISSING, N_MISSING, N_GENO_0/1/2, allele frequency, and MAF.

**Dependencies:** `numpy`, `pandas`, `pandas_plink`

---

## Installation

```bash
# Core dependencies
pip install pandas numpy matplotlib seaborn scipy scikit-learn

# Microbiome
pip install scikit-bio
conda install -c qiime2 qiime2   # or follow https://docs.qiime2.org

# Mixed models
pip install glimix-core numpy-sugar

# Genotype / liftover
pip install pandas-plink pyliftover
```

> **Note:** `pandas_plink` requires the PLINK binary files (`.bed`/`.bim`/`.fam`) from the Palmer Lab HS rat GWAS database (`r11.2.1`). These are not redistributed here. The chain file for rn7→GRCr8 liftover is also required and is available through the Palmer Lab.

---

## Required Input Files (not included)

| File | Used by | Description |
|---|---|---|
| `microbe_abundance.csv` | `limix.py`, `dominance_all.py` | Phenotype + metadata table |
| `my_subset_no_chr10.grm.*` | `limix.py`, `dominance_all.py` | GCTA binary GRM (chr10-excluded) |
| `matched_genotypes_allbiomes_blind.csv` | `analysis_and_plots.py` | OTU table for alpha diversity |
| `matched_allrarefied_genotypes.csv` | `analysis_and_plots.py` | OTU table for beta diversity |
| `matched_allrarefied_exposures.csv` | `analysis_and_plots.py` | OTU table for taxonomy plots |
| `metadata.csv` / `metadata_all.csv` | `classification.py`, `liftover_func.py` | Sample metadata with `rfid` |
| PLINK files (r11.2.1) | `liftover_func.py`, `maf.py` | Palmer Lab HS rat genotype data |
| rn7→GRCr8 chain file | `liftover_func.py` | Coordinate liftover |

---

## Citation

If you use this code, please cite:
> Sadegi et al. (2026) "Replication of Host St6galnac1-Microbiome Association Across Distinct Experimental Conditions in Heterogeneous Stock Rats"

> Tonnelé et al. (2025). "Genetic architecture and mechanisms of host-microbiome interactions from a multi-cohort analysis of outbred laboratory rats". Nature Communications, 10126. https://doi.org/10.1038/s41467-025-66105-z

Dominance classification methodology:
> Cui Y et al. (2023). "Dominance genetic variation contributes little to the missing heritability for human complex traits. Genome Biology", 24(1):111. https://doi.org/10.1186/s13059-023-02954-7
