import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load JSON file
with open("./output/analysis/include_random_2.json", "r") as f:
    data = json.load(f)

method_a, method_b = "SAUCIE", "MAGIC"

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load your JSON
with open("./output/analysis/include_random_2.json", "r") as f:
    data = json.load(f)

method_a, method_b = "SAUCIE", "MAGIC"

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load your JSON
with open("./output/analysis/include_random_2.json", "r") as f:
    data = json.load(f)

method_a, method_b = "SAUCIE", "MAGIC"

for dataset_name, dataset_val in data.items():
    pair_data = dataset_val.get(method_a, {}).get(method_b, {})
    if not pair_data:
        continue
    
    # Make an output directory for each dataset
    output_dir = f"plots/{dataset_name.replace(',', '').replace(' ', '_')}"
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------------------
    # 1) GENE-LEVEL: Combine Actual + Random in separate figures
    # -----------------------------------------------------------------
    gene_concordance_actual = pair_data.get("per_gene_concordance", [])
    gene_concordance_random = pair_data.get("per_gene_concordance_bayesian", [])
    
    if gene_concordance_actual and gene_concordance_random:
        # Sort the actual data descending
        gene_concordance_actual_sorted = sorted(
            gene_concordance_actual, key=lambda x: x[1], reverse=True
        )
        xvals = np.arange(len(gene_concordance_actual_sorted))
        
        # Extract gene labels and actual vals in sorted order
        gene_labels_actual = [g[0] for g in gene_concordance_actual_sorted]
        actual_vals = [g[1] for g in gene_concordance_actual_sorted]

        # Re-map random to actual’s order
        random_dict = dict(gene_concordance_random)
        random_vals_reordered = [random_dict.get(g, 0.0) for g in gene_labels_actual]

        # Full random distribution (for the distribution plot)
        random_vals_full = [val for (_, val) in gene_concordance_random]

        # ----------- Figure A: LINE PLOT (Actual + Random) -----------
        plt.figure(figsize=(8,5))
        plt.plot(xvals, actual_vals, color='blue', label='Actual (Sorted)')
        plt.plot(xvals, random_vals_reordered, color='black', label='Random (Reordered)')
        plt.title(f'Gene-Level Concordance\n(Actual vs Random)\n{dataset_name}')
        plt.xlabel('Genes (ranked by Actual Concordance)')
        plt.ylabel('Concordance')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/gene_concordance_combined_line.png")
        plt.close()

        # ----------- Figure B: DISTRIBUTION (frequency) -----------
        plt.figure(figsize=(8,5))
        sns.histplot(
            actual_vals, color='blue', bins=30,
            element='step', fill=False, stat='count', label='Actual'
        )
        sns.histplot(
            random_vals_full, color='black', bins=30,
            element='step', fill=False, stat='count', label='Random'
        )
        plt.title(f'Gene-Level Concordance Distribution\n(Actual vs Random)\n{dataset_name}')
        plt.xlabel('Concordance')
        plt.ylabel('Number of Genes')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/gene_concordance_combined_distribution.png")
        plt.close()

    # -----------------------------------------------------------------
    # 2) CELL-LEVEL: Combine Actual + Random in separate figures
    # -----------------------------------------------------------------
    cell_concordance_actual = pair_data.get("per_cell_concordance", [])
    cell_concordance_random = pair_data.get("per_cell_concordance_bayesian", [])

    if cell_concordance_actual and cell_concordance_random:
        # Sort the actual data descending
        cell_concordance_actual_sorted = sorted(
            cell_concordance_actual, key=lambda x: x[1], reverse=True
        )
        xvals = np.arange(len(cell_concordance_actual_sorted))
        
        cell_labels_actual = [c[0] for c in cell_concordance_actual_sorted]
        actual_vals = [c[1] for c in cell_concordance_actual_sorted]

        # Build dictionary for random: cell -> random_val
        random_dict = dict(cell_concordance_random)
        random_vals_reordered = [random_dict.get(lbl, 0.0) for lbl in cell_labels_actual]

        # Full random distribution
        random_vals_full = [val for (_, val) in cell_concordance_random]

        # ----------- Figure A: LINE PLOT (Actual + Random) -----------
        plt.figure(figsize=(8,5))
        plt.plot(xvals, actual_vals, color='blue', label='Actual (Sorted)')
        plt.plot(xvals, random_vals_reordered, color='black', label='Random (Reordered)')
        plt.title(f'Cell-Level Concordance\n(Actual vs Random)\n{dataset_name}')
        plt.xlabel('Cells (ranked by Actual Concordance)')
        plt.ylabel('Concordance')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cell_concordance_combined_line.png")
        plt.close()

        # ----------- Figure B: DISTRIBUTION (frequency) -----------
        plt.figure(figsize=(8,5))
        sns.histplot(
            actual_vals, color='blue', bins=30,
            element='step', fill=False, stat='count', label='Actual'
        )
        sns.histplot(
            random_vals_full, color='black', bins=30,
            element='step', fill=False, stat='count', label='Random'
        )
        plt.title(f'Cell-Level Concordance Distribution\n(Actual vs Random)\n{dataset_name}')
        plt.xlabel('Concordance')
        plt.ylabel('Number of Cells')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cell_concordance_combined_distribution.png")
        plt.close()


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
