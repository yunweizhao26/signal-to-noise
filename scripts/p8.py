import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import math

# Marker genes for each cell type
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

# Create a global set of all reference genes (so we only plot these)
reference_gene_set = set()
for gene_list in marker_genes.values():
    reference_gene_set.update(gene_list)

# Load JSON file
with open("./output/analysis/include_random_2.json", "r") as f:
    data = json.load(f)

method_a, method_b = "SAUCIE", "MAGIC"

os.makedirs("plots", exist_ok=True)

# We'll store gene-level & cell-level data to create combined subplots if desired
all_gene_data = []  # list of tuples: (dataset_name, [actual_vals for reference genes only])
all_cell_data = []  # list of tuples: (dataset_name, [actual_vals for all cells])

for dataset_name, dataset_val in data.items():
    pair_data = dataset_val.get(method_a, {}).get(method_b, {})
    if not pair_data:
        continue
    
    output_dir = f"plots/{dataset_name.replace(',', '').replace(' ', '_')}"
    os.makedirs(output_dir, exist_ok=True)

    # ----- A) GENE-LEVEL: filter to reference genes only -----
    gene_concordance_actual = pair_data.get("per_gene_concordance", [])
    # Filter out any genes that are NOT in the reference_gene_set
    gene_concordance_actual_ref = [
        (g, val) for (g, val) in gene_concordance_actual if g in reference_gene_set
    ]
    if gene_concordance_actual_ref:
        # Sort descending by actual value
        gene_concordance_actual_sorted = sorted(
            gene_concordance_actual_ref, key=lambda x: x[1], reverse=True
        )
        xvals = np.arange(len(gene_concordance_actual_sorted))
        actual_vals = [g[1] for g in gene_concordance_actual_sorted]

        # Store distribution for combined subplot
        all_gene_data.append((dataset_name, actual_vals))

        # -- Figure A1: LINE PLOT (only reference genes) --
        plt.figure(figsize=(8, 5))
        plt.plot(xvals, actual_vals, color='blue')
        plt.title(f'Gene-Level (Actual) [Ref Genes]\n{dataset_name}')
        plt.xlabel('Genes (ranked by Actual Concordance)')
        plt.ylabel('Concordance')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/gene_concordance_actual_ref_line.png")
        plt.close()

        # -- Figure A2: DISTRIBUTION (only reference genes) --
        plt.figure(figsize=(8, 5))
        sns.histplot(
            actual_vals, color='blue', bins=30,  # fewer bins, since fewer genes
            element='step', fill=False, stat='count'
        )
        plt.title(f'Gene Dist (Actual) [Ref Genes]\n{dataset_name}')
        plt.xlabel('Concordance')
        plt.ylabel('Num Reference Genes')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/gene_concordance_actual_ref_distribution.png")
        plt.close()

    # ----- B) CELL-LEVEL: (Optional) keep as is, or skip if not needed -----
    cell_concordance_actual = pair_data.get("per_cell_concordance", [])
    if cell_concordance_actual:
        cell_concordance_actual_sorted = sorted(
            cell_concordance_actual, key=lambda x: x[1], reverse=True
        )
        xvals = np.arange(len(cell_concordance_actual_sorted))
        actual_vals = [c[1] for c in cell_concordance_actual_sorted]

        # Store for combined subplot if you want
        all_cell_data.append((dataset_name, actual_vals))

        # -- Figure B1: LINE PLOT --
        plt.figure(figsize=(8, 5))
        plt.plot(xvals, actual_vals, color='blue')
        plt.title(f'Cell-Level (Actual) {dataset_name}')
        plt.xlabel('Cells (ranked by Actual Concordance)')
        plt.ylabel('Concordance')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cell_concordance_actual_line.png")
        plt.close()

        # -- Figure B2: DISTRIBUTION --
        plt.figure(figsize=(8, 5))
        sns.histplot(
            actual_vals, color='blue', bins=100,
            element='step', fill=False, stat='count'
        )
        plt.title(f'Cell Dist (Actual) {dataset_name}')
        plt.xlabel('Concordance')
        plt.ylabel('Num Cells')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cell_concordance_actual_distribution.png")
        plt.close()

# -------------------------------------------------------------------------------------
# Now build a single figure with subplots for ALL gene-level distributions
# (only reference genes). all_gene_data = [(dataset_name, [vals...]), ...]
# -------------------------------------------------------------------------------------
if all_gene_data:
    n = len(all_gene_data)
    ncols = int(math.ceil(n**0.5))  # e.g., sqrt of n, then round up
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), squeeze=False)
    axes = axes.flatten()

    for i, (dset_name, dist_vals) in enumerate(all_gene_data):
        ax = axes[i]
        sns.histplot(dist_vals, color='blue', bins=30,
                     element='step', fill=False, stat='count', ax=ax)
        ax.set_title(dset_name, fontsize=10)
        ax.set_xlabel('Concordance')
        ax.set_ylabel('Ref Genes')

    # Turn off any remaining axes
    for j in range(i+1, nrows*ncols):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig("plots/all_gene_distributions_ref_genes.png")
    plt.close()

# -------------------------------------------------------------------------------------
# Same approach for all_cell_data (cell-level) if you want a combined figure
# -------------------------------------------------------------------------------------
if all_cell_data:
    n = len(all_cell_data)
    ncols = int(math.ceil(n**0.5))
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), squeeze=False)
    axes = axes.flatten()

    for i, (dset_name, dist_vals) in enumerate(all_cell_data):
        ax = axes[i]
        sns.histplot(dist_vals, color='blue', bins=100,
                     element='step', fill=False, stat='count', ax=ax)
        ax.set_title(dset_name, fontsize=10)
        ax.set_xlabel('Concordance')
        ax.set_ylabel('Num Cells')

    for j in range(i+1, nrows*ncols):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig("plots/all_cell_distributions.png")
    plt.close()

# ------------------------------------------------------------------------------
# 2) (Optional) Plot side-by-side contingency tables (Actual vs. Random)
#    This part is unchanged; it doesn't revolve around reference genes
# ------------------------------------------------------------------------------
for dataset_name, dataset_val in data.items():
    pair_data = dataset_val.get(method_a, {}).get(method_b, {})
    if not pair_data:
        continue

    contingency_actual = pair_data.get("contingency")          
    contingency_random = pair_data.get("contingency_random")   

    if not contingency_actual or not contingency_random:
        continue

    mat_actual = np.array([
        [contingency_actual["both_imputed"], contingency_actual["only_method_a_imputed"]],
        [contingency_actual["only_method_b_imputed"], contingency_actual["neither_imputed"]]
    ])
    mat_random = np.array([
        [contingency_random["both_imputed"], contingency_random["only_method_a_imputed"]],
        [contingency_random["only_method_b_imputed"], contingency_random["neither_imputed"]]
    ])

    output_dir = f"plots/{dataset_name.replace(',', '').replace(' ', '_')}"
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(ncols=2, figsize=(12, 5))

    sns.heatmap(
        mat_actual, ax=axes[0],
        annot=True, fmt='d', cmap='twilight',
        xticklabels=[f'{method_b} Imputed', f'{method_b} Not Imputed'],
        yticklabels=[f'{method_a} Imputed', f'{method_a} Not Imputed']
    )
    axes[0].set_title(f'Actual Contingency\n({dataset_name})')
    axes[0].set_xlabel(method_b)
    axes[0].set_ylabel(method_a)

    sns.heatmap(
        mat_random, ax=axes[1],
        annot=True, fmt='d', cmap='twilight',
        xticklabels=[f'{method_b} Imputed', f'{method_b} Not Imputed'],
        yticklabels=[f'{method_a} Imputed', f'{method_a} Not Imputed']
    )
    axes[1].set_title(f'Random Contingency\n({dataset_name})')
    axes[1].set_xlabel(method_b)
    axes[1].set_ylabel(method_a)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/contingency_table_actual_vs_random.png")
    plt.close()

# ------------------------------------------------------------------------------
# 3) Build a 3-column table for overall concordance (Actual vs Random), unchanged
# ------------------------------------------------------------------------------
table_data = []
for dataset_name, dataset_val in data.items():
    pair_data = dataset_val.get(method_a, {}).get(method_b, {})
    if not pair_data:
        continue

    c_actual = pair_data.get("contingency", {})
    c_random = pair_data.get("contingency_random", {})
    if not c_actual or not c_random:
        continue

    total_actual = (c_actual["both_imputed"] +
                    c_actual["only_method_a_imputed"] +
                    c_actual["only_method_b_imputed"] +
                    c_actual["neither_imputed"])
    if total_actual == 0:
        overall_conc_actual = 0.0
    else:
        overall_conc_actual = (
            c_actual["both_imputed"] + c_actual["neither_imputed"]
        ) / total_actual

    total_random = (c_random["both_imputed"] +
                    c_random["only_method_a_imputed"] +
                    c_random["only_method_b_imputed"] +
                    c_random["neither_imputed"])
    if total_random == 0:
        overall_conc_random = 0.0
    else:
        overall_conc_random = (
            c_random["both_imputed"] + c_random["neither_imputed"]
        ) / total_random

    table_data.append([dataset_name, overall_conc_actual, overall_conc_random])

print("=== Overall Concordance Table ===")
print("Dataset Name | Overall Conc (Actual) | Overall Conc (Random)")
for row in table_data:
    print(f"{row[0]} | {row[1]:.4f} | {row[2]:.4f}")

df = pd.DataFrame(table_data, columns=["dataset_name", "overall_conc_actual", "overall_conc_random"])
df.to_csv("plots/overall_concordance_table.csv", index=False)
print("\nSaved table to 'plots/overall_concordance_table.csv'.")
