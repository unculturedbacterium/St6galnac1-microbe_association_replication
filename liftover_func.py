import pandas as pd
import numpy as np
from pandas_plink import read_plink
from pyliftover import LiftOver


# NCBI accession → chromosome number (GRCr8)
NCBI_TO_NUM = {
    "NC_086019.1": "1",  "NC_086020.1": "2",  "NC_086021.1": "3",
    "NC_086022.1": "4",  "NC_086023.1": "5",  "NC_086024.1": "6",
    "NC_086025.1": "7",  "NC_086026.1": "8",  "NC_086027.1": "9",
    "NC_086028.1": "10", "NC_086029.1": "11", "NC_086030.1": "12",
    "NC_086031.1": "13", "NC_086032.1": "14", "NC_086033.1": "15",
    "NC_086034.1": "16", "NC_086035.1": "17", "NC_086036.1": "18",
    "NC_086037.1": "19", "NC_086038.1": "20", "NC_086039.1": "X",
    "NC_086040.1": "Y",  "NC_001665.3": "MT"
}

GENO_LABELS = {
    "00": "Homo_REF",
    "01": "HET",
    "10": "HET",
    "11": "Homo_ALT"
}

def get_genotypes(
    chrom,                    # e.g. 4
    pos,                      # rn7 position, e.g. 70_834_123
    output_file,              # e.g. "metadata_ch4_genotypes.csv"
    metadata    = "metadata_all.csv",
    plink_prefix= "/tscc/projects/ps-palmer/gwas/databases/rounds/r11.2.1",
    chain_file  = "/tscc/projects/ps-palmer/gwas/projects/baud_legacy/rn7ToGCF_036323735.1.over.chain.gz",
):
    """
    Extract genotypes for a given rn7 chromosome:position,
    lift over to GRCr8, and merge with metadata.

    Parameters
    ----------
    chrom        : int or str  — chromosome number (rn7), e.g. 4
    pos          : int         — position (rn7, 1-based), e.g. 70_834_123
    output_file  : str         — output CSV filename
    metadata     : str         — path to metadata CSV (must have 'rfid' column)
    plink_prefix : str         — path prefix to plink files
    chain_file   : str         — path to rn7→GRCr8 chain file
    """

    # LIFTOVER
    lo     = LiftOver(chain_file)
    result = lo.convert_coordinate(f"chr{chrom}", pos - 1)   # 0-based input

    if not result:
        raise ValueError(f"chr{chrom}:{pos} did not lift over. Check coordinates/chain file.")

    lifted_chr_raw = result[0][0]
    lifted_pos     = result[0][1] + 1   # back to 1-based
    lifted_chr     = NCBI_TO_NUM.get(lifted_chr_raw, lifted_chr_raw)
    print(f"Lifted: chr{chrom}:{pos} (rn7) → chr{lifted_chr}:{lifted_pos} (GRCr8)")

    # LOAD PLINK 
    bim, fam, bed = read_plink(plink_prefix)
    bim["chrom"]  = bim["chrom"].astype(str)

    #  FIND SNP 
    snp_row = bim[(bim["chrom"] == str(lifted_chr)) & (bim["pos"] == lifted_pos)]
    if snp_row.empty:
        snp_row = bim[(bim["chrom"] == str(lifted_chr)) &
                      (bim["pos"].between(lifted_pos - 1, lifted_pos + 1))]
    if snp_row.empty:
        raise ValueError(f"No SNP found at GRCr8 chr{lifted_chr}:{lifted_pos} in the plink file.")

    snp_idx = snp_row.index[0]
    a0      = snp_row.iloc[0]["a0"]
    a1      = snp_row.iloc[0]["a1"]
    snp_id  = snp_row.iloc[0]["snp"]
    print(f"SNP: {snp_id}  |  a0={a0} (REF)  a1={a1} (ALT)")

    #  MATCH RFIDS
    meta       = pd.read_csv(metadata, dtype={"rfid": str})
    your_rfids = set(meta["rfid"].dropna().unique())

    fam["rfid"] = fam["iid"].str.extract(r"([\dA-Z]{10,15})")
    matched     = fam[fam["rfid"].isin(your_rfids)].copy()

    if matched.empty:
        raise ValueError("No RFIDs from metadata matched sample IDs in the plink fam file.")
    print(f"Matched {len(matched)} rats in plink data.")

    # EXTRACT GENOTYPES
    geno_raw = bed[snp_idx, matched.index.tolist()].compute()

    def decode(code, a0, a1):
        if np.isnan(code): return "NA"
        key = str(int(code == 0 or code == 1)) + str(int(code == 1 or code == 2))
        # simpler direct mapping:
        if code == 0:   return f"Homo_REF"   # a0/a0
        elif code == 1: return f"HET"         # a0/a1
        elif code == 2: return f"Homo_ALT"   # a1/a1
        else:           return "NA"

    genotypes = [decode(g, a0, a1) for g in geno_raw]

    geno_df = pd.DataFrame({
        "rfid":     matched["rfid"].values,
        "genotype": genotypes
    })

    # MERGE WITH METADATA & SAVE
    out = meta.merge(geno_df, on="rfid", how="left")
    out.to_csv(output_file, index=False)
    print(f"\nSaved {output_file} — {len(out)} rows")
    print(out["genotype"].value_counts(dropna=False).to_string())

    return out
