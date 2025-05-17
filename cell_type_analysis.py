import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from sklearn.metrics import accuracy_score, adjusted_rand_score
import matplotlib.pyplot as plt
from tqdm import tqdm

methods = ["SAUCIE", "MAGIC", "deepImpute", "scScope", "scVI", "knn_smoothing"]
ZERO_TOL = 0
DATA_DIR = Path("./datasets")
OUT_DIR = Path("./results")
OUT_DIR.mkdir(exist_ok=True)

mark_df = pd.read_csv("./data/marker_genes.csv")
marker = {r["Cell subset name"].strip().lower(): eval(r["Markers"]) for _, r in mark_df.iterrows()}
cell_types = list(marker.keys())

def build_magic_map(X):
    nz = X.getnnz(axis=0) if hasattr(X, "getnnz") else (X != 0).sum(0).A1
    keep = np.where(nz >= 5)[0]
    return {g: i for i, g in enumerate(keep)}


# def log1p_cpm(vec):
#     lib = vec.sum() or 1
#     return np.log1p(vec * 1e6 / lib)

def load_imp_column(method, ds, disease, tissue, gene_idx, magic_map):
    p = Path("output") / method / ds / disease / f"{tissue}.npy"
    if not p.exists():
        return None
    arr = np.load(p, mmap_mode="r")
    if method == "MAGIC":
        pos = magic_map.get(gene_idx)
        if pos is None or pos >= arr.shape[1]:
            return np.zeros(arr.shape[0])
        return arr[:, pos]
    return arr[:, gene_idx]


def get_orig_column(X, idx):
    return X[:, idx].toarray().ravel().astype(float) if hasattr(X, "tocsr") else X[:, idx].astype(float)

# ---------- main loop ----------
rows = []
for h5 in DATA_DIR.glob("*.h5ad"):
    print("\n=== Processing", h5.name, "===")
    dsid = h5.stem
    adata_full = ad.read_h5ad(h5)
    # match the exact cell set used during imputation
    if "is_primary_data" in adata_full.obs:
        adata_full = adata_full[adata_full.obs["is_primary_data"]]
        print("  filtered to primary data –", adata_full.n_obs, "cells")
    print("  loaded AnnData – cells:", adata_full.n_obs, "genes:", adata_full.n_vars)

    if "feature_name" in adata_full.var:
        adata_full.var_names = adata_full.var["feature_name"].astype(str).str.replace(r"\.\d+$", "", regex=True)
    name_to_idx = {g: i for i, g in enumerate(adata_full.var_names)}

    magic_map_global = build_magic_map(adata_full.X)
    print("  built MAGIC map –", len(magic_map_global), "genes kept by MAGIC")

    combos = adata_full.obs[["disease", "tissue"]].drop_duplicates()
    for dis, tis in tqdm(combos.itertuples(index=False), total=len(combos), desc=f"{dsid} combos"):
        sub = adata_full[(adata_full.obs["disease"] == dis) & (adata_full.obs["tissue"] == tis)].copy()
        n_cells = sub.n_obs
        if n_cells == 0:
            continue

        print(f"    ▶ {dis} / {tis} – {n_cells} cells")
        feats = np.zeros((n_cells, len(cell_types)))
        gene_counts = np.zeros(len(cell_types))

        for ct_idx, ct in enumerate(tqdm(cell_types, desc="    marker sets", leave=False)):
            genes = [g for g in marker[ct] if g in name_to_idx]
            if not genes:
                continue
            gene_counts[ct_idx] = len(genes)
            for g in genes:
                gi = name_to_idx[g]
                orig_col = get_orig_column(sub.X, gi)
                zeros = orig_col == 0

                vals = np.zeros(n_cells)
                used = 0
                for m in methods:
                    col = load_imp_column(m, dsid, dis, tis, gi, magic_map_global)
                    if col is None:
                        continue
                    # col = log1p_cpm(col.reshape(-1, 1)).ravel()
                    if col.shape[0] != n_cells:
                        print(f"          ⚠︎ skip {m} (len {len(col)} ≠ {n_cells}) for gene {g}")
                        continue
                    vals += col
                    used += 1
                consensus = vals / used if used else np.zeros(n_cells)
                filled = orig_col.copy()
                filled[zeros] = consensus[zeros]
                feats[:, ct_idx] += filled

        valid = gene_counts > 0
        feats[:, valid] = feats[:, valid] / gene_counts[valid]
        preds = np.array(cell_types)[feats.argmax(1)]
        sub.obs["PredCelltype"] = preds

        if "Celltype" in sub.obs.columns:
            true = sub.obs["Celltype"].str.lower().str.strip().values
            acc = accuracy_score(true, preds)
            ari = adjusted_rand_score(true, preds)
            print(f"      accuracy={acc:.3f}  ARI={ari:.3f}")
        else:
            acc = np.nan
            ari = np.nan
            print("      no ground‑truth Celltype column – skipping accuracy/ARI")

        rows.append({"dataset": dsid, "disease": dis, "tissue": tis,
                     "cells": n_cells, "accuracy": acc, "ARI": ari})

        if "Celltype" in sub.obs.columns:
            labels = np.unique(np.concatenate([true, preds]))
            cm = pd.crosstab(pd.Series(true, name="true", dtype="category"),
                              pd.Series(preds, name="pred", dtype="category"), dropna=False)
            cm = cm.reindex(index=labels, columns=labels, fill_value=0)
            plt.figure(figsize=(4, 4))
            plt.imshow(cm.values, interpolation="nearest", cmap="Blues")
            plt.xticks(range(len(labels)), labels, rotation=90)
            plt.yticks(range(len(labels)), labels)
            plt.title(f"{dis} / {tis}\nacc={acc:.2f}")
            plt.tight_layout()
            plt.savefig(OUT_DIR / f"cm_{dsid}_{dis}_{tis}.png", dpi=250)
            plt.close()

res_df = pd.DataFrame(rows)
res_df.to_csv(OUT_DIR / "prediction_evaluation.csv", index=False)
print("\n=== Summary ===")
print(res_df)
print("Finished – results in", OUT_DIR)
