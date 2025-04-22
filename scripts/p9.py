import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from anndata import read_h5ad

# ----------------- MARKER GENES DICTIONARY -----------------
marker_genes = {
  "Enterocytes BEST4": [
    "BEST4", "OTOP2", "CA7", "GUCA2A", "GUCA2B", "SPIB", "CFTR"
  ],
  "Goblet cells MUC2 TFF1": [
    "MUC2", "TFF1", "TFF3", "FCGBP", "AGR2", "SPDEF"
  ],
  "Tuft cells": [
    "POU2F3", "DCLK1"
  ],
  "Goblet cells SPINK4": [
    "MUC2", "SPINK4"
  ],
  "Enterocytes TMIGD1 MEP1A": [
    "CA1", "CA2", "TMIGD1", "MEP1A"
  ],
  "Enterocytes CA1 CA2 CA4-": [
    "CA1", "CA2"
  ],
  "Goblet cells MUC2 TFF1-": [
    "MUC2"
  ],
  "Epithelial Cycling cells": [
    "LGR5", "OLFM4", "MKI67"
  ],
  "Enteroendocrine cells": [
    "CHGA", "GCG", "GIP", "CCK"
  ],
  "Stem cells OLFM4": [
    "OLFM4", "LGR5"
  ],
  "Stem cells OLFM4 LGR5": [
    "OLFM4", "LGR5", "ASCL2"
  ],
  "Stem cells OLFM4 PCNA": [
    "OLFM4", "PCNA", "LGR5", "ASCL2", "SOX9", "TERT"
  ],
  "Paneth cells": [
    "LYZ", "DEFA5"
  ]
}

# The imputation methods you want to compare
methods = ["MAGIC", "SAUCIE"]


def expand_magic_matrix(full_original, magic_reduced):
    """
    Re-inserts columns that MAGIC may have dropped. 
    - full_original: shape (N_dt, G) for entire (disease, tissue) subset
    - magic_reduced: shape (N_dt, G_kept) from the MAGIC output
    Returns a shape-(N_dt, G) array with those columns restored.
    """
    expanded = np.zeros_like(full_original)
    
    nonzero_counts = (full_original != 0).sum(axis=0)
    kept_columns = np.where(nonzero_counts >= 5)[0]
    removed_columns = np.where(nonzero_counts < 5)[0]
    
    # Copy values from reduced matrix to their original positions
    for j_reduced, j_original in enumerate(kept_columns):
        expanded[:, j_original] = magic_reduced[:, j_reduced]
        
    # For columns that MAGIC removed, revert to the original values
    for j_original in removed_columns:
        expanded[:, j_original] = full_original[:, j_original]
    
    return expanded


# We build one big DataFrame with columns = [dataset, disease, tissue, cell_type, gene, "original", method1, method2, ...]
rows_list = []

data_dir = Path("./datasets")
files = list(data_dir.glob("*.h5ad"))

for f in files:
    dataset_id = f.stem
    print(f"Processing dataset {dataset_id}")
    adata = read_h5ad(f)

    # Filter out non-primary data if needed
    if "is_primary_data" in adata.obs.columns:
        adata = adata[adata.obs["is_primary_data"] == True]

    # Overwrite var_names with gene symbols from feature_name (strip version suffix)
    if "feature_name" in adata.var.columns:
        adata.var_names = adata.var["feature_name"].astype(str).str.replace(r"\.\d+$", "", regex=True)
    else:
        print("Warning: 'feature_name' not found in adata.var.columns. Gene symbol matching may fail.")

    # We'll gather the sample-level splits by "disease" and "tissue"
    if not {"disease", "tissue"}.issubset(adata.obs.columns):
        # If missing, treat entire adata as one subset
        disease_tissue_df = pd.DataFrame([("Unknown", "Unknown")], columns=["disease", "tissue"])
    else:
        disease_tissue_df = adata.obs[["disease", "tissue"]].drop_duplicates()

    for _, (disease, tissue) in disease_tissue_df.iterrows():
        # Subset to the entire (disease, tissue)
        mask_dt = (adata.obs["disease"] == disease) & (adata.obs["tissue"] == tissue)
        adata_dt = adata[mask_dt]

        if adata_dt.n_obs < 5:
            continue

        X_dt_original = adata_dt.X.toarray()  # shape => (N_dt, G)
        var_names_dt = adata_dt.var_names
        if "Celltype" not in adata_dt.obs.columns:
            continue
        obs_celltype_dt = adata_dt.obs["Celltype"]

        # Load the entire (disease,tissue) imputed matrix for each method
        method_to_imputed = {}
        for m in methods:
            imputed_path = Path("output") / m / dataset_id / disease / (tissue + ".npy")
            if not imputed_path.exists():
                method_to_imputed[m] = None
                continue

            X_dt_imputed = np.load(imputed_path)
            if m == "MAGIC":
                # expand columns
                X_dt_imputed = expand_magic_matrix(X_dt_original, X_dt_imputed)

            if X_dt_imputed.shape != X_dt_original.shape:
                print(f"[Warning: shape mismatch for {m}, skipping.]")
                method_to_imputed[m] = None
                continue

            method_to_imputed[m] = X_dt_imputed

        # For each cell type & reference gene, subset rows, compute fraction
        for ctype_key, gene_list in marker_genes.items():
            ct_mask = (obs_celltype_dt == ctype_key)
            n_ct = ct_mask.sum()
            if n_ct < 5:
                continue

            # We can also measure the original fraction here
            # shape => (n_ct, G)
            X_ct_original = X_dt_original[ct_mask, :]

            for gene in gene_list:
                row_dict = {
                    "dataset": dataset_id,
                    "disease": disease,
                    "tissue": tissue,
                    "cell_type": ctype_key,
                    "gene": gene
                }

                if gene not in var_names_dt:
                    for m in methods:
                        row_dict[m] = np.nan
                    # Also set the 'original' fraction to np.nan if gene not in data
                    row_dict["original"] = np.nan
                    rows_list.append(row_dict)
                    continue

                gene_idx = var_names_dt.get_loc(gene)

                # fraction of cells originally >0 for this gene
                expr_original = X_ct_original[:, gene_idx]
                row_dict["original"] = np.mean(expr_original > 0)

                # fraction of cells after imputation
                for m in methods:
                    X_dt_imputed_full = method_to_imputed[m]
                    if X_dt_imputed_full is None:
                        row_dict[m] = np.nan
                        continue

                    X_ct_imputed = X_dt_imputed_full[ct_mask, :]
                    expr_values = X_ct_imputed[:, gene_idx]
                    fraction_expr = np.mean(expr_values > 0)
                    row_dict[m] = fraction_expr

                rows_list.append(row_dict)

# ------------------- Build the final DataFrame ---------------------
df_recovery = pd.DataFrame(rows_list)
# columns: [dataset, disease, tissue, cell_type, gene, original, MAGIC, SAUCIE, ...]

df_recovery.to_csv("marker_gene_recovery.csv", index=False)
print("Saved marker gene recovery fractions (incl. original) to 'marker_gene_recovery.csv'.")


# ------------------- OPTIONAL: Create Heatmaps ---------------------
# For example, let's do one heatmap per dataset, comparing "original" vs. methods.
# We'll average across disease/tissue again, or do separately if you like.

unique_datasets = df_recovery["dataset"].unique()
for ds in unique_datasets:
    df_ds = df_recovery[df_recovery["dataset"] == ds]
    if df_ds.empty:
        continue

    # group by (cell_type, gene) and average across all disease/tissue
    # (If you prefer each disease/tissue separately, adapt your grouping.)
    agg_cols = ["original"] + methods  # e.g. ["original", "MAGIC", "SAUCIE"]
    df_agg = df_ds.groupby(["cell_type", "gene"])[agg_cols].mean().reset_index()

    df_pivot = df_agg.pivot_table(index=["cell_type", "gene"], values=agg_cols)

    # We'll make a heatmap with more columns now: "original", "MAGIC", "SAUCIE", ...
    plt.figure(figsize=(14, 0.5*len(df_pivot) + 4))
    sns.heatmap(
        df_pivot,
        annot=True,
        fmt=".2f",
        cmap="Purples",
        cbar_kws={"label": "Fraction Expressed"}
    )
    plt.title(f"Marker Recovery: {ds} (original vs. imputed; avg across disease/tissue)")
    plt.tight_layout()
    plt.savefig(f"marker_recovery_{ds}_with_original.png")
    plt.close()

print("Done creating optional heatmaps (including 'original').")
