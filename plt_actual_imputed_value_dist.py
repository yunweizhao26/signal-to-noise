import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import read_h5ad

DATASETS_DIR = Path("datasets")
IMPUTE_DIR = Path("output")
METHODS = ["SAUCIE", "MAGIC",  "deepImpute", "scScope", "scVI", "knn_smoothing"]
PLOT_DIR = Path("plots")
MAX_POINTS = 3_000_000

def log(msg): 
    print(msg, flush=True)

def load_original(dataset_h5ad, disease, tissue):
    adata = read_h5ad(dataset_h5ad)
    adata = adata[adata.obs["is_primary_data"] == True]
    mask = (adata.obs["disease"] == disease) & (adata.obs["tissue"] == tissue)
    X = adata[mask].X
    return X.toarray() if hasattr(X, "toarray") else X

def expand_magic_matrix(y, reduced_matrix):
    expanded = np.zeros(y.shape)
    nonzero_counts = (y != 0).sum(axis=0)
    kept_columns = np.where(nonzero_counts >= 5)[0]
    removed_columns = np.where(nonzero_counts < 5)[0]
    
    for j_reduced, j_original in enumerate(kept_columns):
        expanded[:, j_original] = reduced_matrix[:, j_reduced]
        
    for j_original in removed_columns:
        expanded[:, j_original] = y[:, j_original]
    
    return expanded

def collect_imputed(dataset_id, disease, tissue, original):
    n_zero_mask = original == 0
    per_method = {}
    for m in METHODS:
        f = IMPUTE_DIR / m / dataset_id / disease / f"{tissue}.npy"
        if not f.exists():
            log(f"  – {m}: missing, skip")
            continue
        X_imp = np.load(f)
        if m == "MAGIC":
            X_imp = expand_magic_matrix(original, X_imp)
        new_vals = X_imp[n_zero_mask]
        new_vals = new_vals[new_vals > 0]
        if len(new_vals):
            per_method[m] = new_vals
            log(f"  – {m}: {len(new_vals):,} newly imputed values")
    return per_method

def subsample(arr, cap=MAX_POINTS):
    return np.random.choice(arr, cap, replace=False) if arr.size > cap else arr

def plot_distributions(dataset_id, disease, tissue, per_method):
    if not per_method: 
        return
    
    PLOT_DIR.mkdir(exist_ok=True, parents=True)
    fname = PLOT_DIR / f"{disease}_{tissue}_imputed_dist.png"
    
    plt.figure(figsize=(10, 6))  # Horizontal layout
    
    for m, vals in per_method.items():
        vals = subsample(vals)
        plt.hist(vals, bins=100, histtype='step', label=m, alpha=0.8)
    
    plt.xlabel("Imputed value")  # Linear scale
    plt.ylabel("Count")
    plt.title(f"Distribution of NEWLY imputed values\n {disease} / {tissue}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    
    log(f"  → saved {fname}")

def main():
    PLOT_DIR.mkdir(exist_ok=True, parents=True)
    
    for dataset_id in {p.parent.parent.name for p in IMPUTE_DIR.glob("*/*/*/*.npy")}:
        h5ad_files = list(DATASETS_DIR.glob(f"{dataset_id}*.h5ad"))
        if not h5ad_files:
            log(f"*** No h5ad for {dataset_id}, skipping")
            continue
            
        adata = read_h5ad(h5ad_files[0])
        combos = adata.obs[["disease", "tissue"]].drop_duplicates()
        
        for disease, tissue in combos.itertuples(index=False):
            log(f"\n{dataset_id} – {disease} / {tissue}")
            original = load_original(h5ad_files[0], disease, tissue)
            per_method = collect_imputed(dataset_id, disease, tissue, original)
            plot_distributions(dataset_id, disease, tissue, per_method)

if __name__ == "__main__":
    main()