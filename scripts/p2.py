import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load JSON file
with open("./output/analysis/new_info.json", "r") as f:
    data = json.load(f)

method_a, method_b = "SAUCIE", "MAGIC"

# Prepare to collect aggregated scores
gene_scores_all = []
cell_scores_all = []

# Loop through each dataset
for dataset_name, dataset in data.items():
    pair_data = dataset.get(method_a, {}).get(method_b, {})
    if not pair_data:
        continue

    gene_scores = [score for _, score in pair_data.get("per_gene_concordance", [])]
    cell_scores = [score for _, score in pair_data.get("per_cell_concordance", [])]

    # Add to aggregated lists
    gene_scores_all.extend(gene_scores)
    cell_scores_all.extend(cell_scores)

    # Create output directory for each dataset
    output_dir = f"plots/{dataset_name.replace(',', '').replace(' ', '_')}"
    os.makedirs(output_dir, exist_ok=True)

    # Sorted line plots
    plt.figure(figsize=(10, 6))
    plt.plot(sorted(gene_scores, reverse=True), color='skyblue')
    plt.xlabel('Genes (sorted high to low concordance)')
    plt.ylabel('Gene Concordance Score')
    plt.title(f'Gene-Level Concordance Sorted ({dataset_name})')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Gene_Concordance_Sorted.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(sorted(cell_scores, reverse=True), color='lightcoral')
    plt.xlabel('Cells (sorted high to low concordance)')
    plt.ylabel('Cell Concordance Score')
    plt.title(f'Cell-Level Concordance Sorted ({dataset_name})')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Cell_Concordance_Sorted.png')
    plt.close()

    # Histogram percentage plots
    plt.figure(figsize=(10, 6))
    sns.histplot(gene_scores, bins=50, kde=True, color='skyblue', stat='percent')
    plt.xlabel('Gene Concordance Score')
    plt.ylabel('Percentage (%)')
    plt.title(f'Gene-Level Concordance Distribution ({dataset_name})')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Gene_Concordance_Histogram_Percent.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(cell_scores, bins=50, kde=True, color='lightcoral', stat='percent')
    plt.xlabel('Cell Concordance Score')
    plt.ylabel('Percentage (%)')
    plt.title(f'Cell-Level Concordance Distribution ({dataset_name})')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Cell_Concordance_Histogram_Percent.png')
    plt.close()

    # Violin plots
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=[gene_scores, cell_scores], palette=['skyblue', 'lightcoral'])
    plt.xticks([0, 1], ['Gene Concordance', 'Cell Concordance'])
    plt.ylabel('Concordance Score')
    plt.title(f'Gene vs. Cell Concordance ({dataset_name})')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Gene_vs_Cell_Concordance_Violin.png')
    plt.close()

# Aggregated plots (saved outside)
os.makedirs("plots", exist_ok=True)

# Sorted line plots aggregated
plt.figure(figsize=(10, 6))
plt.plot(sorted(gene_scores_all, reverse=True), color='skyblue')
plt.xlabel('Genes (sorted high to low concordance)')
plt.ylabel('Gene Concordance Score')
plt.title('Aggregated Gene-Level Concordance Sorted')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plots/Aggregated_Gene_Concordance_Sorted.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(sorted(cell_scores_all, reverse=True), color='lightcoral')
plt.xlabel('Cells (sorted high to low concordance)')
plt.ylabel('Cell Concordance Score')
plt.title('Aggregated Cell-Level Concordance Sorted')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plots/Aggregated_Cell_Concordance_Sorted.png')
plt.close()

# Histogram percentage plots aggregated
plt.figure(figsize=(10, 6))
sns.histplot(gene_scores_all, bins=50, kde=True, color='skyblue', stat='percent')
plt.xlabel('Gene Concordance Score')
plt.ylabel('Percentage (%)')
plt.title('Aggregated Gene-Level Concordance Distribution')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plots/Aggregated_Gene_Concordance_Histogram_Percent.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.histplot(cell_scores_all, bins=50, kde=True, color='lightcoral', stat='percent')
plt.xlabel('Cell Concordance Score')
plt.ylabel('Percentage (%)')
plt.title('Aggregated Cell-Level Concordance Distribution')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plots/Aggregated_Cell_Concordance_Histogram_Percent.png')
plt.close()

# Aggregated violin plot
plt.figure(figsize=(8, 6))
sns.violinplot(data=[gene_scores_all, cell_scores_all], palette=['skyblue', 'lightcoral'])
plt.xticks([0, 1], ['Gene Concordance', 'Cell Concordance'])
plt.ylabel('Concordance Score')
plt.title('Aggregated Gene vs. Cell Concordance')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plots/Aggregated_Gene_vs_Cell_Concordance_Violin.png')
plt.close()