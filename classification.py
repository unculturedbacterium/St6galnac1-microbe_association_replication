import pandas as pd
from qiime2 import Artifact, Metadata
from qiime2.plugins.feature_table.visualizers import summarize, barplot
from qiime2.plugins import feature_classifier

# Import the FASTA as FeatureData[Sequence]
sequences = Artifact.import_data("FeatureData[Sequence]", "all.seqs.fa")
classifier = Artifact.load("SILVA138.2_SSURef_NR99_uniform_classifier_full-length.qza") #<--- Note: you can change this, im using Silva because it is most up to date
taxonomy = feature_classifier.methods.classify_sklearn(
    reads=sequences,
    classifier=classifier
)

taxonomy.classification.save("taxonomy_silva.qza")



# Load your feature table and metadata
feature_table = Artifact.load("rarefied_feature_table.qza")
metadata = Metadata.load("226721_mapping_file.txt")

# Generate summary visualization
table_summary = summarize(table=feature_table, sample_metadata=metadata)

# Save the visualization
table_summary.visualization.save("feature_table_summary_silva_rarefied.qzv")


taxonomy = Artifact.load("taxonomy_silva.qza")

tax_barplot = barplot(
    table=feature_table,
    taxonomy=taxonomy,
    metadata=metadata
)

tax_barplot.visualization.save("taxonomy_barplot_silva_rarefied.qzv")


taxonomy = Artifact.load("taxonomy_silva.qza")

tax_barplot = barplot(
    table=feature_table,
    taxonomy=taxonomy,
    metadata=metadata
)

tax_barplot.visualization.save("taxonomy_barplot_silva_rarefied.qzv")


def match_data(genotypes_file, origname_file, output_file):

    try:
        # 1. Load DataFrames
        df_genotype = pd.read_csv(genotypes_file)
        df_origname = pd.read_csv(origname_file)
        
        print(f"Loaded Genotype data with {len(df_genotype)} rows.")
        print(f"Loaded Prevotella OrigName data with {len(df_origname)} rows.")

        if 'orig_name' in df_origname.columns:
            df_origname = df_origname.rename(columns={'orig_name': 'tube_barcode'})
        else:
            print("Error: Could not find the 'orig_name' column in the Prevotella file.")
            return

        df_genotype['tube_barcode'] = df_genotype['tube_barcode'].astype(str)
        df_origname['tube_barcode'] = df_origname['tube_barcode'].astype(str)

        merged_df = pd.merge(
            df_genotype, 
            df_origname, 
            on='tube_barcode', 
            how='inner'
        )

        merged_df.to_csv(output_file, index=False)
        
        print(f"\nSuccessfully merged {len(merged_df)} matching rows.")
        print(f"Saved the result to {output_file}")
        
        print("\nFirst 5 rows of the merged data:")
        print(merged_df.head())

    except FileNotFoundError:
        print("Error: One or both input files were not found. Please ensure 'genotypes.csv' and 'prevotella_origname.csv' are available.")
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")

# Define file paths
GENOTYPES_FILE = 'metadata.csv'
ORIGNAME_FILE = 'rarefiedtaxonomy.csv'
OUTPUT_FILE = 'matched_allrarefied_silva.csv'

# Execute the data matching process
if __name__ == '__main__':
    match_data(GENOTYPES_FILE, ORIGNAME_FILE, OUTPUT_FILE)

def match_data(genotypes_file, origname_file, output_file):

    try:
        # 1. Load DataFrames
        df_genotype = pd.read_csv(genotypes_file)
        df_origname = pd.read_csv(origname_file)
        
        print(f"Loaded Genotype data with {len(df_genotype)} rows.")
        print(f"Loaded Prevotella OrigName data with {len(df_origname)} rows.")

        if 'orig_name' in df_origname.columns:
            df_origname = df_origname.rename(columns={'orig_name': 'tube_barcode'})
        else:
            print("Error: Could not find the 'orig_name' column in the Prevotella file.")
            return

        df_genotype['tube_barcode'] = df_genotype['tube_barcode'].astype(str)
        df_origname['tube_barcode'] = df_origname['tube_barcode'].astype(str)

        merged_df = pd.merge(
            df_genotype, 
            df_origname, 
            on='tube_barcode', 
            how='inner'
        )

        merged_df.to_csv(output_file, index=False)
        
        print(f"\nSuccessfully merged {len(merged_df)} matching rows.")
        print(f"Saved the result to {output_file}")
        
        print("\nFirst 5 rows of the merged data:")
        print(merged_df.head())

    except FileNotFoundError:
        print("Error: One or both input files were not found. Please ensure 'genotypes.csv' and 'prevotella_origname.csv' are available.")
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")

# Define file paths
GENOTYPES_FILE = 'metadata.csv'
ORIGNAME_FILE = 'rarefiedtaxonomy_silva.csv'
OUTPUT_FILE = 'matched_allrarefied_silva.csv'

# Execute the data matching process
if __name__ == '__main__':
    match_data(GENOTYPES_FILE, ORIGNAME_FILE, OUTPUT_FILE)
