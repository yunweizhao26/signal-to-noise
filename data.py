import json
from collections import defaultdict
with open("./output/analysis/include_random_2.json", "r") as f:
    data = json.load(f)
unique_genes_per_dataset = defaultdict(set)
unique_celltypes_per_dataset = defaultdict(set)
for dataset_name, dataset_val in data.items():
    for method_a, sub_a in dataset_val.items():
        if isinstance(sub_a, dict):
            for method_b, sub_b in sub_a.items():
                if not isinstance(sub_b, dict):
                    continue
                gene_concord_list = sub_b.get("per_gene_concordance", [])
                for (gene_name, val) in gene_concord_list:
                    unique_genes_per_dataset[dataset_name].add(gene_name)
                cell_concord_list = sub_b.get("per_cell_concordance", [])
                for (cell_label, val) in cell_concord_list:
                    if "(" in cell_label:
                        celltype = cell_label.split("(")[0].strip()
                    else:
                        celltype = cell_label.strip()
                    unique_celltypes_per_dataset[dataset_name].add(celltype)
for ds in unique_genes_per_dataset:
    unique_genes_per_dataset[ds] = sorted(unique_genes_per_dataset[ds])
for ds in unique_celltypes_per_dataset:
    unique_celltypes_per_dataset[ds] = sorted(unique_celltypes_per_dataset[ds])
for ds, gene_list in unique_genes_per_dataset.items():
    print(ds, "=> # Genes:", len(gene_list))
for ds, cell_list in unique_celltypes_per_dataset.items():
    print(ds, "=> # Cell types:", len(cell_list))
with open("unique_celltypes_per_dataset.json", "w") as f:
    json.dump(unique_celltypes_per_dataset, f, indent=4)
with open("unique_genes_per_dataset.json", "w") as f:
    json.dump(unique_genes_per_dataset, f, indent=4)
