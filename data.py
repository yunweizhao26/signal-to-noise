import json
from collections import defaultdict

with open("./output/analysis/include_random_2.json", "r") as f:
    data = json.load(f)

# We'll store:
#   unique_genes_per_dataset[dataset_name] = {set of gene names}
#   unique_celltypes_per_dataset[dataset_name] = {set of cell types}
unique_genes_per_dataset = defaultdict(set)
unique_celltypes_per_dataset = defaultdict(set)

for dataset_name, dataset_val in data.items():
    # dataset_name is e.g. "Crohn disease, lamina propria of mucosa of colon"

    # The top-level keys in 'dataset_val' might be method names: "SAUCIE", "MAGIC", etc.
    # Each method sub-dict might have "contingency", "per_gene_concordance", ...
    for method_a, sub_a in dataset_val.items():
        # e.g. method_a = "SAUCIE"
        # sub_a might have "SAUCIE", "MAGIC", etc. or might be a single dict
        # Because you have structures like:
        # "SAUCIE": {
        #    "SAUCIE": 1,
        #    "MAGIC": {
        #       "contingency": {...},
        #       "per_gene_concordance": [...],
        #       ...
        #     }
        # }
        # we'll iterate further:
        if isinstance(sub_a, dict):
            for method_b, sub_b in sub_a.items():
                # If sub_b is not a dict, skip
                if not isinstance(sub_b, dict):
                    continue

                # 1. Extract genes from "per_gene_concordance" if present
                gene_concord_list = sub_b.get("per_gene_concordance", [])
                # gene_concord_list is a list of [ [gene_name, value], ... ]
                for (gene_name, val) in gene_concord_list:
                    unique_genes_per_dataset[dataset_name].add(gene_name)

                # 2. Extract cell types from "per_cell_concordance" if present
                cell_concord_list = sub_b.get("per_cell_concordance", [])
                # cell_concord_list is a list of [ [cell_label, value], ... ]
                for (cell_label, val) in cell_concord_list:
                    # If the cell label has a parentheses portion, you can parse out the real cell type
                    # e.g. "Paneth cells (N105446_L-AGTAACCGTTAAGGGC)"
                    if "(" in cell_label:
                        celltype = cell_label.split("(")[0].strip()
                    else:
                        celltype = cell_label.strip()
                    unique_celltypes_per_dataset[dataset_name].add(celltype)

# Convert each set to a sorted list, if you like
for ds in unique_genes_per_dataset:
    unique_genes_per_dataset[ds] = sorted(unique_genes_per_dataset[ds])
for ds in unique_celltypes_per_dataset:
    unique_celltypes_per_dataset[ds] = sorted(unique_celltypes_per_dataset[ds])

# Now you have:
# unique_genes_per_dataset["Crohn disease, lamina propria of mucosa of colon"] -> sorted list of genes
# unique_celltypes_per_dataset["Crohn disease, lamina propria of mucosa of colon"] -> sorted list of cell types

# Example printout:
for ds, gene_list in unique_genes_per_dataset.items():
    print(ds, "=> # Genes:", len(gene_list))
    # print(gene_list)  # or do something else

for ds, cell_list in unique_celltypes_per_dataset.items():
    print(ds, "=> # Cell types:", len(cell_list))
    # print(cell_list)

# write the results into a file
with open("unique_celltypes_per_dataset.json", "w") as f:
    json.dump(unique_celltypes_per_dataset, f, indent=4)
with open("unique_genes_per_dataset.json", "w") as f:
    json.dump(unique_genes_per_dataset, f, indent=4)