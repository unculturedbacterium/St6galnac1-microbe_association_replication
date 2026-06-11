import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from skbio.diversity import alpha_diversity, beta_diversity
from skbio.stats.ordination import pcoa
from skbio.stats.composition import clr
from skbio.stats.distance import DistanceMatrix, permanova
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

# Alpha Diversity 
df = pd.read_csv("matched_genotypes_allbiomes_blind.csv")

meta = df[['genotype']]
otu = df.drop(columns=['genotype'])

otu = otu.apply(pd.to_numeric, errors='coerce').fillna(0)

metrics = ['shannon', 'simpson', 'observed_otus']
alpha_df = pd.DataFrame({
    m: alpha_diversity(m, otu.values, ids=otu.index) for m in metrics
})
alpha_df['genotype'] = meta['genotype'].values


plt.figure(figsize=(12, 4))
for i, m in enumerate(metrics):
    plt.subplot(1, 3, i+1)
    sns.boxplot(x='genotype', y=m, data=alpha_df, palette='viridis')
    sns.stripplot(x='genotype', y=m, data=alpha_df, color='black', size=3, alpha=0.5)
    plt.title(f'Alpha diversity: {m.capitalize()}')
    plt.xlabel('')
plt.tight_layout()
plt.show()

#Beta Diversity (genotypes) for other parameters, the same code was used, just parameters changed

data_fp = "matched_allrarefied_genotypes.csv"

df = pd.read_csv(data_fp)

assert 'genotype' in df.columns, "No 'genotype' column found!"

metadata = df[['genotype']]
features = df.drop(columns=['genotype'])

features = features + 1e-6

clr_features = pd.DataFrame(
    clr(features.values),
    index=df.index,
    columns=features.columns
)


dist_matrix = squareform(pdist(clr_features, metric='euclidean'))
dm = DistanceMatrix(dist_matrix, ids=clr_features.index)


permanova_result = permanova(dm, metadata['genotype'], permutations=999)
print("\n PERMANOVA Results")
print(permanova_result)

pca = PCA(n_components=2)
coords = pca.fit_transform(clr_features)

pca_df = pd.DataFrame(coords, columns=['PC1', 'PC2'])
pca_df['genotype'] = metadata['genotype'].values

plt.figure(figsize=(8,6))

for group in pca_df['genotype'].unique():
    subset = pca_df[pca_df['genotype'] == group]
    plt.scatter(subset['PC1'], subset['PC2'], label=group, s=80, alpha=0.7)

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.title("Aitchison Beta Diversity (CLR + PCA)")
plt.legend(title="Genotype")
plt.tight_layout()
plt.show()

#Average Taxonomic plots

# Load your rarefied microbiome data
data_fp = "matched_allrarefied_exposures.csv"
df = pd.read_csv(data_fp)

# Check columns
# 'cohort' column + multiple taxa columns
taxa_columns = [c for c in df.columns if c != 'cohort']

# Compute average relative abundance per cohort
avg_abundance = df.groupby('cohort')[taxa_columns].mean()

# Optional: normalize to 100% (sum to 1 or 100%)
avg_abundance = avg_abundance.div(avg_abundance.sum(axis=1), axis=0)

# Sort taxa by total abundance for nicer plotting
taxa_order = avg_abundance.sum(axis=0).sort_values(ascending=False).index
avg_abundance = avg_abundance[taxa_order]

# Plot stacked bar plot
fig, ax = plt.subplots(figsize=(10, 6))

bottom = None
cohorts = avg_abundance.index

for i, taxa in enumerate(avg_abundance.columns):
    if bottom is None:
        ax.bar(cohorts, avg_abundance[taxa], label=taxa)
        bottom = avg_abundance[taxa].values
    else:
        ax.bar(cohorts, avg_abundance[taxa], bottom=bottom, label=taxa)
        bottom += avg_abundance[taxa].values

ax.set_ylabel("Average Relative Abundance")
ax.set_xlabel("Cohort")
ax.set_title("Average Taxonomic Composition per Cohort")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.show()
