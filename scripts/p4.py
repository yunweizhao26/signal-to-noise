import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Load JSON file
with open("./output/analysis/include_random_2.json", "r") as f:
    data = json.load(f)

method_a, method_b = "SAUCIE", "MAGIC"

# Create directory for tables
os.makedirs("plots/contingency_tables", exist_ok=True)

# Initialize for aggregated table
aggregate_contingency = np.zeros((2, 2), dtype=int)

# Aggregated contingency counts
aggregated_contingency = {
    "both_imputed": 0,
    "only_method_a_imputed": 0,
    "only_method_b_imputed": 0,
    "neither_imputed": 0,
    "originally_zero": 0
}

for dataset_name, dataset in data.items():
    pair_data = dataset.get(method_a, {}).get(method_b, {})
    if not pair_data:
        continue

    # Prepare contingency data
    contingency = pair_data.get("contingency")
    if not contingency:
        continue

    # Build 2x2 contingency matrix
    contingency_matrix = np.array([
        [contingency["both_imputed"], contingency["only_method_a_imputed"]],
        [contingency["only_method_b_imputed"], contingency["neither_imputed"]]
    ])

    # Create output directory for this dataset
    output_dir = f"plots/{dataset_name.replace(',', '').replace(' ', '_')}"
    os.makedirs(output_dir, exist_ok=True)

    # Plot heatmap of contingency
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        contingency_matrix,
        annot=True, fmt='d', cmap='twilight',
        xticklabels=[f'{method_b} Imputed', f'{method_b} Not Imputed'],
        yticklabels=[f'{method_a} Imputed', f'{method_a} Not Imputed']
    )
    plt.title(f'Contingency Table ({dataset_name})')
    plt.ylabel(method_a)
    plt.xlabel(method_b)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/contingency_table.png')
    plt.close()

    # =============== 1) PER-GENE Concordance ========================
    gene_concordance_actual = pair_data.get("per_gene_concordance", [])
    gene_concordance_random = pair_data.get("per_gene_concordance_bayesian", [])
    # Each item is (gene_name, value)

    if gene_concordance_actual and gene_concordance_random:
        # ---- 1.1) Plot Actual Concordance, sorted by its own values ----
        # Sort descending by actual value
        gene_concordance_actual_sorted = sorted(
            gene_concordance_actual, key=lambda x: x[1], reverse=True
        )
        xvals_actual = np.arange(len(gene_concordance_actual_sorted))
        actual_vals = [x[1] for x in gene_concordance_actual_sorted]

        plt.figure(figsize=(8, 6))
        plt.plot(xvals_actual, actual_vals, color='blue')
        plt.title(f'Gene-Level Concordance (Actual)\n({dataset_name})')
        plt.xlabel('Genes (ranked by actual concordance)')
        plt.ylabel('Concordance')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/gene_concordance_actual.png")
        plt.close()

        # ---- 1.2) Plot Random Concordance, sorted by its own values ----
        gene_concordance_random_sorted = sorted(
            gene_concordance_random, key=lambda x: x[1], reverse=True
        )
        xvals_random = np.arange(len(gene_concordance_random_sorted))
        random_vals = [x[1] for x in gene_concordance_random_sorted]

        plt.figure(figsize=(8, 6))
        plt.plot(xvals_random, random_vals, color='black')
        plt.title(f'Gene-Level Concordance (Random)\n({dataset_name})')
        plt.xlabel('Genes (ranked by random concordance)')
        plt.ylabel('Concordance')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/gene_concordance_random.png")
        plt.close()

    # =============== 2) PER-CELL Concordance ========================
    cell_concordance_actual = pair_data.get("per_cell_concordance", [])
    cell_concordance_random = pair_data.get("per_cell_concordance_bayesian", [])

    if cell_concordance_actual and cell_concordance_random:
        # ---- 2.1) Plot Actual Concordance in its own sorted order ----
        cell_concordance_actual_sorted = sorted(
            cell_concordance_actual, key=lambda x: x[1], reverse=True
        )
        xvals_actual = np.arange(len(cell_concordance_actual_sorted))
        actual_vals = [x[1] for x in cell_concordance_actual_sorted]

        plt.figure(figsize=(8, 6))
        plt.plot(xvals_actual, actual_vals, color='blue')
        plt.title(f'Cell-Level Concordance (Actual)\n({dataset_name})')
        plt.xlabel('Cells (ranked by actual concordance)')
        plt.ylabel('Concordance')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cell_concordance_actual.png")
        plt.close()

        # ---- 2.2) Plot Random Concordance in its own sorted order ----
        cell_concordance_random_sorted = sorted(
            cell_concordance_random, key=lambda x: x[1], reverse=True
        )
        xvals_random = np.arange(len(cell_concordance_random_sorted))
        random_vals = [x[1] for x in cell_concordance_random_sorted]

        plt.figure(figsize=(8, 6))
        plt.plot(xvals_random, random_vals, color='black')
        plt.title(f'Cell-Level Concordance (Random)\n({dataset_name})')
        plt.xlabel('Cells (ranked by random concordance)')
        plt.ylabel('Concordance')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cell_concordance_random.png")
        plt.close()

    # Accumulate for aggregated contingency
    for key in aggregated_contingency:
        aggregated_contingency[key] += contingency.get(key, 0)

for dataset_name, dataset_val in data.items():
    pair_data = dataset_val.get(method_a, {}).get(method_b, {})
    if not pair_data:
        continue

    # 1) Retrieve the actual contingency
    contingency_actual = pair_data.get("contingency")          # your "both_imputed", etc.
    # 2) Retrieve the random contingency
    contingency_random = pair_data.get("contingency_random")   # your "both_imputed", etc.

    if not contingency_actual or not contingency_random:
        # skip if missing either
        continue

    # Build the 2x2 for actual
    mat_actual = np.array([
        [contingency_actual["both_imputed"], contingency_actual["only_method_a_imputed"]],
        [contingency_actual["only_method_b_imputed"], contingency_actual["neither_imputed"]]
    ])

    # Build the 2x2 for random
    mat_random = np.array([
        [contingency_random["both_imputed"], contingency_random["only_method_a_imputed"]],
        [contingency_random["only_method_b_imputed"], contingency_random["neither_imputed"]]
    ])

    # Create output directory for this dataset
    output_dir = f"plots/{dataset_name.replace(',', '').replace(' ', '_')}"
    os.makedirs(output_dir, exist_ok=True)

    # Make a figure with 2 subplots side by side
    fig, axes = plt.subplots(ncols=2, figsize=(12, 5))

    # Left subplot = Actual
    sns.heatmap(
        mat_actual, ax=axes[0],
        annot=True, fmt='d', cmap='twilight',
        xticklabels=[f'{method_b} Imputed', f'{method_b} Not Imputed'],
        yticklabels=[f'{method_a} Imputed', f'{method_a} Not Imputed']
    )
    axes[0].set_title(f'Actual Contingency\n({dataset_name})')
    axes[0].set_xlabel(method_b)
    axes[0].set_ylabel(method_a)

    # Right subplot = Random
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
