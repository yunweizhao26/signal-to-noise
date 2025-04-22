import json
import numpy as np
import matplotlib.pyplot as plt

# Load JSON file
with open("./output/analysis/post_analysis.json", "r") as f:
    data = json.load(f)

method_a, method_b = "SAUCIE", "MAGIC"

# Extract and sort gene concordance scores
gene_scores = []
cell_scores = []

for sample in data.values():
    pair_data = sample.get(method_a, {}).get(method_b, {})
    if pair_data:
        gene_scores.extend([score for _, score in pair_data.get("per_gene_concordance", [])])
        cell_scores.extend([score for _, score in pair_data.get("per_cell_concordance", [])])

# Sort scores from high to low
gene_sorted = np.sort(gene_scores)[::-1]
cell_sorted = np.sort(cell_scores)[::-1]

# Plot gene-level concordance
plt.figure(figsize=(10, 6))
plt.plot(gene_sorted, linewidth=2, color='skyblue')
plt.xlabel('Genes (sorted high to low concordance)')
plt.ylabel('Gene Concordance Score')
plt.title(f'Gene-Level Concordance Scores Sorted ({method_a} vs. {method_b})')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('Gene_Level_Concordance_Line.png')
plt.show()

# Plot cell-level concordance
plt.figure(figsize=(10, 6))
plt.plot(cell_sorted, linewidth=2, color='lightcoral')
plt.xlabel('Cells (sorted high to low concordance)')
plt.ylabel('Cell Concordance Score')
plt.title(f'Cell-Level Concordance Scores Sorted ({method_a} vs. {method_b})')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('Cell_Level_Concordance_Line.png')
plt.show()

# Calculate and print average total concordance
total_concordances = [
    sample.get(method_a, {}).get(method_b, {}).get("total_concordance")
    for sample in data.values()
    if sample.get(method_a, {}).get(method_b, {})
]
avg_total = np.mean([c for c in total_concordances if c is not None])
print(f"Average Total Concordance ({method_a} vs. {method_b}): {avg_total:.2e}")
