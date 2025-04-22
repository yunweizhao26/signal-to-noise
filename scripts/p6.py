import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd  # if you want to store the table in a DataFrame

# Load JSON file
with open("./output/analysis/include_random_2.json", "r") as f:
    data = json.load(f)

method_a, method_b = "SAUCIE", "MAGIC"

# Create a folder for plots (if not already present)
os.makedirs("plots", exist_ok=True)

# 1) Remove the random line from the actual line plots
#    We'll show ONLY the actual line & distribution for gene-level and cell-level.

for dataset_name, dataset_val in data.items():
    pair_data = dataset_val.get(method_a, {}).get(method_b, {})
    if not pair_data:
        continue
    
    output_dir = f"plots/{dataset_name.replace(',', '').replace(' ', '_')}"
    os.makedirs(output_dir, exist_ok=True)

    # ----- A) GENE-LEVEL: ONLY ACTUAL -----
    gene_concordance_actual = pair_data.get("per_gene_concordance", [])
    # ignore the random data -> gene_concordance_random = pair_data.get("per_gene_concordance_bayesian", [])

    if gene_concordance_actual:
        # Sort descending by actual value
        gene_concordance_actual_sorted = sorted(
            gene_concordance_actual, key=lambda x: x[1], reverse=True
        )
        xvals = np.arange(len(gene_concordance_actual_sorted))

        # Extract labels & values
        actual_vals = [g[1] for g in gene_concordance_actual_sorted]

        # -- Figure A1: LINE PLOT (Actual only) --
        plt.figure(figsize=(8, 5))
        plt.plot(xvals, actual_vals, color='blue', label='Actual (Sorted)')
        plt.title(f'Gene-Level Concordance (Actual Only)\n{dataset_name}')
        plt.xlabel('Genes (ranked by Actual Concordance)')
        plt.ylabel('Concordance')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/gene_concordance_actual_line.png")
        plt.close()

        # -- Figure A2: DISTRIBUTION (Actual only) --
        plt.figure(figsize=(8, 5))
        sns.histplot(
            actual_vals, color='blue', bins=100,
            element='step', fill=False, stat='count'
        )
        plt.title(f'Gene-Level Concordance Distribution (Actual Only)\n{dataset_name}')
        plt.xlabel('Concordance')
        plt.ylabel('Number of Genes')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/gene_concordance_actual_distribution.png")
        plt.close()

    # ----- B) CELL-LEVEL: ONLY ACTUAL -----
    cell_concordance_actual = pair_data.get("per_cell_concordance", [])
    if cell_concordance_actual:
        cell_concordance_actual_sorted = sorted(
            cell_concordance_actual, key=lambda x: x[1], reverse=True
        )
        xvals = np.arange(len(cell_concordance_actual_sorted))
        actual_vals = [c[1] for c in cell_concordance_actual_sorted]

        # -- Figure B1: LINE PLOT (Actual only) --
        plt.figure(figsize=(8, 5))
        plt.plot(xvals, actual_vals, color='blue', label='Actual (Sorted)')
        plt.title(f'Cell-Level Concordance (Actual Only)\n{dataset_name}')
        plt.xlabel('Cells (ranked by Actual Concordance)')
        plt.ylabel('Concordance')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cell_concordance_actual_line.png")
        plt.close()

        # -- Figure B2: DISTRIBUTION (Actual only) --
        plt.figure(figsize=(8, 5))
        sns.histplot(
            actual_vals, color='blue', bins=100,
            element='step', fill=False, stat='count'
        )
        plt.title(f'Cell-Level Concordance Distribution (Actual Only)\n{dataset_name}')
        plt.xlabel('Concordance')
        plt.ylabel('Number of Cells')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cell_concordance_actual_distribution.png")
        plt.close()

# 2) Plot side-by-side contingency tables (Actual vs. Random) - optional
for dataset_name, dataset_val in data.items():
    pair_data = dataset_val.get(method_a, {}).get(method_b, {})
    if not pair_data:
        continue

    # 1) Retrieve the actual contingency
    contingency_actual = pair_data.get("contingency")          
    # 2) Retrieve the random contingency
    contingency_random = pair_data.get("contingency_random")   

    if not contingency_actual or not contingency_random:
        # skip if missing either
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


# 3) Build a 3-column table: [ dataset_name, overall_concordance_actual, overall_concordance_random ]
#    overall_concordance = (both_imputed + neither_imputed) / sum_of_all_4
table_data = []
for dataset_name, dataset_val in data.items():
    pair_data = dataset_val.get(method_a, {}).get(method_b, {})
    if not pair_data:
        continue

    # "contingency" = actual, "contingency_random" = random
    c_actual = pair_data.get("contingency", {})
    c_random = pair_data.get("contingency_random", {})

    # Make sure both exist
    if not c_actual or not c_random:
        continue

    # For the actual
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

    # For the random
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

    # Store in a list
    table_data.append([dataset_name, overall_conc_actual, overall_conc_random])

# Print out the 3-column table
print("=== Overall Concordance Table ===")
print("Dataset Name | Overall Concordance (Actual) | Overall Concordance (Random)")
for row in table_data:
    print(f"{row[0]} | {row[1]:.4f} | {row[2]:.4f}")

# Optionally save as CSV
df = pd.DataFrame(table_data, columns=["dataset_name", "overall_conc_actual", "overall_conc_random"])
df.to_csv("plots/overall_concordance_table.csv", index=False)
print("\nSaved table to 'plots/overall_concordance_table.csv' (3 columns).")
