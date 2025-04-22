import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("disagreement.csv")

# Make an output folder for subdataset-level plots
os.makedirs("subdataset_disagreement_plots", exist_ok=True)

# Group by (dataset, disease, tissue)
group_cols = ["dataset", "disease", "tissue"]
# for (ds, dis, tis), subdf in df.groupby(group_cols):
#     if subdf.empty:
#         continue
    
#     # Combine cell_type + gene into one label on x-axis
#     subdf["cg_label"] = subdf["cell_type"] + "_" + subdf["gene"]
    
#     plt.figure(figsize=(12, 6))
#     sns.barplot(data=subdf, x="cg_label", y="disagreement", color="purple")
#     plt.xticks(rotation=90)
#     plt.title(f"Disagreement per (CellType, Gene)\n{dis}, {tis}")
#     plt.xlabel("Cell Type - Reference Gene")
#     plt.ylabel("Disagreement Score")
#     plt.ylim(0, 1)
#     plt.tight_layout()

#     # Save figure name with safe characters
#     outname = f"subdataset_disagreement_plots/{dis}_{tis}_disagreement_bar.png"
#     outname = outname.replace(" ", "_")  # remove spaces
#     plt.savefig(outname, dpi=120)
#     plt.close()

for (ds, dis, tis), subdf in df.groupby(["dataset", "disease", "tissue"]):
    if subdf.empty:
        continue
    
    # pivot => index=cell_type, columns=gene, values=disagreement
    pivot = subdf.pivot_table(
        index="cell_type",
        columns="gene",
        values="disagreement",
        aggfunc="mean"  # if duplicates
    )
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(
        pivot, annot=True, fmt=".2f",
        cmap="Purples", vmin=0, vmax=1,
        cbar_kws={"label": "Disagreement Score"}
    )
    plt.title(f"Disagreement Heatmap\n{dis}, {tis}")
    plt.tight_layout()

    outname = f"subdataset_disagreement_plots/{dis}_{tis}_disagreement_heatmap.png"
    outname = outname.replace(" ", "_")
    plt.savefig(outname, dpi=120)
    plt.close()
