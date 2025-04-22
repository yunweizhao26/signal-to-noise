import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load JSON file
with open("./output/analysis/post_analysis.json", "r") as f:
    data = json.load(f)

method_a, method_b = "SAUCIE", "MAGIC"

# Extract concordance scores
gene_scores = []
cell_scores = []

for sample in data.values():
    pair_data = sample.get(method_a, {}).get(method_b, {})
    if pair_data:
        gene_scores.extend([score for _, score in pair_data.get("per_gene_concordance", [])])
        cell_scores.extend([score for _, score in pair_data.get("per_cell_concordance", [])])

# Plot histogram for gene-level concordance (percentage)
plt.figure(figsize=(10, 6))
sns.histplot(gene_scores, bins=50, kde=True, color='skyblue', stat='percent')
plt.xlabel('Gene Concordance Score')
plt.ylabel('Percentage (%)')
plt.title(f'Gene-Level Concordance Score Distribution ({method_a} vs. {method_b})')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('Gene_Level_Concordance_Histogram_Percent.png')
plt.show()

# Plot histogram for cell-level concordance (percentage)
plt.figure(figsize=(10, 6))
sns.histplot(cell_scores, bins=50, kde=True, color='lightcoral', stat='percent')
plt.xlabel('Cell Concordance Score')
plt.ylabel('Percentage (%)')
plt.title(f'Cell-Level Concordance Score Distribution ({method_a} vs. {method_b})')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('Cell_Level_Concordance_Histogram_Percent.png')
plt.show()

# Violin plot comparison
plt.figure(figsize=(8, 6))
sns.violinplot(data=[gene_scores, cell_scores], palette=['skyblue', 'lightcoral'])
plt.xticks([0, 1], ['Gene Concordance', 'Cell Concordance'])
plt.ylabel('Concordance Score')
plt.title(f'Gene vs. Cell Concordance Score Comparison ({method_a} vs. {method_b})')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('Gene_Cell_Concordance_Violin.png')
plt.show()

# Calculate and print average total concordance
total_concordances = [
    sample.get(method_a, {}).get(method_b, {}).get("total_concordance")
    for sample in data.values()
    if sample.get(method_a, {}).get(method_b, {})
]
avg_total = np.mean([c for c in total_concordances if c is not None])
print(f"Average Total Concordance ({method_a} vs. {method_b}): {avg_total:.2e}")