import os
import math
from pathlib import Path
from functools import lru_cache

import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.preprocessing import normalize
from sklearn.decomposition import IncrementalPCA
import phate
from sklearn.cluster import KMeans

MODE = os.environ.get("MODE", "filled").lower()
assert MODE in {"filled", "imputed_only", "unsupervised"}

DATA_DIR = Path("./datasets")
OUT_DIR = Path("./results")
OUT_DIR.mkdir(exist_ok=True)

def zscore(col: np.ndarray):
    mu = col.mean()
    sd = col.std()
    if sd < 1e-8:
        return np.zeros_like(col, dtype=float)
    return (col - mu) / sd

@lru_cache(maxsize=64)
def _open_npy(path: str):
    return np.load(path, mmap_mode="r")

mark_df = pd.read_csv("./data/marker_genes.csv")
marker = {
    r["Cell subset name"].strip().lower(): eval(r["Markers"])
    for _, r in mark_df.iterrows()
}
cell_types = list(marker.keys())

def build_magic_map(X):
    nz = X.getnnz(axis=0) if hasattr(X, "getnnz") else (X != 0).sum(0).A1
    keep = np.where(nz >= 5)[0]
    return {g: i for i, g in enumerate(keep)}

def load_imp_column(method: str, ds: str, dis: str, tis: str, gi: int, magic_map: dict):
    p = Path("output") / method / ds / dis / f"{tis}.npy"
    if not p.exists():
        return None
    arr = _open_npy(str(p))
    if method == "MAGIC":
        pos = magic_map.get(gi)
        if pos is None or pos >= arr.shape[1]:
            return None
        return arr[:, pos]
    return arr[:, gi]

def get_raw_col(X, idx):
    return X[:, idx].toarray().ravel() if hasattr(X, "tocsr") else X[:, idx]

def short(label: str, maxlen: int = 25):
    return label[: maxlen - 1] + "…" if len(label) > maxlen else label

def cpm_norm(v: np.ndarray):
    lib = v.sum() or 1.0
    return v * 1e6 / lib

import textwrap
def wrap(lbl, width=18):
    if len(lbl) <= width:
        return lbl
    short = textwrap.fill(lbl[:width-1] + '…', break_long_words=False)
    return short.replace('\n', ' ')

def summarise_method_ranges(ds: str, dis: str, tis: str, magic_map: dict, n_show: int = 10):
    rows = []
    for m in methods:
        p = Path("output") / m / ds / dis / f"{tis}.npy"
        if not p.exists():
            continue
        arr = _open_npy(str(p))
        n_genes = min(n_show, arr.shape[1])
        sample = arr[:, :n_genes].astype(float).ravel()
        pcts = np.nanpercentile(sample, [0, 50, 95])
        rows.append([m, *pcts])
    if rows:
        pd.DataFrame(rows, columns=["method", "min", "median", "p95"]).to_csv(
            OUT_DIR / f"range_{ds}_{dis}_{tis}.csv", index=False
        )

methods = [
    "SAUCIE",
    "MAGIC",
    "deepImpute",
    "scScope",
    "scVI",
    "knn_smoothing",
]

rows = []
for h5 in DATA_DIR.glob("*.h5ad"):
    dsid = h5.stem
    print(f"\n=== processing {dsid}  (mode={MODE}) ===")

    adata = ad.read_h5ad(h5)
    if "is_primary_data" in adata.obs:
        adata = adata[adata.obs["is_primary_data"]]
    if "feature_name" in adata.var:
        adata.var_names = (
            adata.var["feature_name"].astype(str).str.replace(r"\.\d+$", "", regex=True)
        )
    name2idx = {g: i for i, g in enumerate(adata.var_names)}

    magic_map = build_magic_map(adata.X)

    combos = adata.obs[["disease", "tissue"]].drop_duplicates()
    for dis, tis in tqdm(combos.itertuples(index=False), total=len(combos), desc=dsid):
        sub = adata[(adata.obs["disease"] == dis) & (adata.obs["tissue"] == tis)]
        n_cells = sub.n_obs
        if n_cells == 0:
            continue

        summarise_method_ranges(dsid, dis, tis, magic_map)

        if MODE == "unsupervised":
            cols = []
            for gi in range(sub.n_vars):
                col = load_imp_column("MAGIC", dsid, dis, tis, gi, magic_map)
                if col is None or len(col) != n_cells:
                    col = get_raw_col(sub.X, gi)
                cols.append(col)
            M = np.vstack(cols).T.astype(float)
            M = normalize(M, norm="l1", axis=1) * 1e6
            phate_op = phate.PHATE(n_components=30, random_state=0)
            emb = phate_op.fit_transform(M)
            
            k = max(2, int(math.sqrt(n_cells / 50)))
            preds = KMeans(k, n_init=10, random_state=0).fit_predict(emb).astype(str)

        else:
            feats = np.zeros((n_cells, len(cell_types)))
            gene_counts = np.zeros(len(cell_types))

            for ct_idx, ct in enumerate(cell_types):
                genes = [g for g in marker[ct] if g in name2idx]
                if not genes:
                    continue
                gene_counts[ct_idx] = len(genes)

                for g in genes:
                    gi = name2idx[g]
                    raw = get_raw_col(sub.X, gi)
                    zeros = raw == 0

                    imp_sum = np.zeros(n_cells)
                    used = 0
                    for m in methods:
                        col = load_imp_column(m, dsid, dis, tis, gi, magic_map)
                        if col is None or len(col) != n_cells:
                            continue
                        imp_sum += zscore(col.astype(float))
                        used += 1

                    if used == 0:
                        continue
                    consensus = imp_sum / used

                    if MODE == "filled":
                        final = raw.copy()
                        final[zeros] = consensus[zeros]
                    else:
                        final = np.zeros(n_cells)
                        final[zeros] = consensus[zeros]

                    feats[:, ct_idx] += final

            valid = gene_counts > 0
            feats[:, valid] /= gene_counts[valid]
            preds = np.array(cell_types)[feats.argmax(1)]

        true_available = "Celltype" in sub.obs.columns
        if true_available:
            true = sub.obs["Celltype"].str.lower().str.strip().values
            acc = accuracy_score(true, preds)
            ari = adjusted_rand_score(true, preds)
            print(f"  {dis}/{tis}: acc={acc:.3f}   ARI={ari:.3f}")
        else:
            acc = ari = np.nan

        rows.append(
            dict(dataset=dsid, disease=dis, tissue=tis, cells=n_cells, accuracy=acc, ARI=ari)
        )

        if true_available:
            labels = np.unique(np.concatenate([true, preds]))[:20]

            cm = pd.crosstab(
                pd.Categorical(true, categories=labels),
                pd.Categorical(preds, categories=labels),
                dropna=False,
            )

            fig, ax = plt.subplots(figsize=(2.6, 2.6))
            vmax = cm.values.max() * 0.9 or 1
            sns.heatmap(cm, cmap="Blues", ax=ax, cbar=False,
                        linewidths=0.4, linecolor='white',
                        vmin=0, vmax=vmax)

            n = cm.shape[1]
            ax.set_xticks(np.arange(n) + .5)
            ax.set_yticks(np.arange(n) + .5)

            ax.set_xticklabels([wrap(l) for l in labels],
                            rotation=90, ha='center', fontsize=5)
            ax.set_yticklabels([wrap(l) for l in labels],
                            rotation=0,  va='center', fontsize=5)

            ax.set(xlabel='', ylabel='')

            ax.set_xlim(0, cm.shape[1])
            ax.set_ylim(cm.shape[0], 0)

            ax.set_title(f"{short(dis)} / {short(tis)}\nacc={acc:.2f}",
                        fontsize=7, pad=3)

            plt.tight_layout(pad=0.2)
            fig.savefig(OUT_DIR / f"cm_{dsid}_{dis}_{tis}.png", dpi=250)
            plt.close(fig)

res = pd.DataFrame(rows)
res.to_csv(OUT_DIR / f"prediction_evaluation_{MODE}.csv", index=False)
print("\n=== Summary ===")
print(res)
print("Results saved to", OUT_DIR)

if MODE == "filled" and len(res):
    fig, axes = plt.subplots(3, 3, figsize=(11, 10))
    axes = axes.flatten()
    for idx, (d, t) in enumerate(res[["disease", "tissue"]].itertuples(index=False)):
        if idx >= 9:
            break
        p = OUT_DIR / f"cm_{dsid}_{d}_{t}.png"
        if p.exists():
            axes[idx].imshow(plt.imread(p))
        axes[idx].axis("off")
    for j in range(idx + 1, 9):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"nine_cm_grid_{MODE}.png", dpi=200)
    plt.close()
