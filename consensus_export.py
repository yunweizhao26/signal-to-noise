import os, re
from pathlib import Path

import numpy as np, pandas as pd, anndata as ad
from scipy import sparse
from tqdm import tqdm

DATA_DIR  = Path("datasets")
IMP_DIR   = Path("output")
EMB_DIR   = Path("embedding")
RAW_ROOT  = EMB_DIR / "raw"
CONS_ROOT = EMB_DIR / "consensus"

RAW_ROOT.mkdir(parents=True, exist_ok=True)
CONS_ROOT.mkdir(parents=True, exist_ok=True)

methods = ["SAUCIE", "MAGIC", "deepImpute", "scScope", "scVI", "knn_smoothing"]

def zscore_rows(mat, eps=1e-8):
    mu = mat.mean(axis=1, keepdims=True)
    sd = mat.std (axis=1, keepdims=True)
    return (mat - mu) / (sd + eps)

def build_magic_map(X):
    nz = X.getnnz(axis=0) if sparse.issparse(X) else (X != 0).sum(0)
    keep = np.where(nz >= 5)[0]
    return {g: i for i, g in enumerate(keep)}

def load_imp(method, ds, dis, tis):
    p = IMP_DIR / method / ds / dis / f"{tis}.npy"
    return (None, p) if not p.exists() else (np.load(p, mmap_mode="r"), p)

def norm_fname(s):
    return re.sub(r"\s+", "_", s)

def export_consensus(mode="filled"):
    assert mode in {"filled", "imputed_only"}
    for h5 in tqdm(list(DATA_DIR.glob("*.h5ad")), desc="datasets"):
        dsid  = h5.stem
        adata = ad.read_h5ad(h5)
        if "is_primary_data" in adata.obs:
            adata = adata[adata.obs["is_primary_data"]]
        if "feature_name" in adata.var:
            adata.var_names = (adata.var["feature_name"]
                               .astype(str)
                               .str.replace(r"\.\d+$", "", regex=True))
        magic_map = build_magic_map(adata.X)
        combos = adata.obs[["disease", "tissue"]].drop_duplicates()
        for dis, tis in tqdm(combos.itertuples(index=False),
                             total=len(combos),
                             desc=dsid,
                             leave=False):
            sub = adata[(adata.obs["disease"] == dis) & (adata.obs["tissue"] == tis)]
            if sub.n_obs == 0:
                continue
            n_genes, n_cells = sub.n_vars, sub.n_obs
            consensus = np.zeros((n_genes, n_cells), dtype=np.float32)
            n_used = 0
            for m in methods:
                imp, imp_path = load_imp(m, dsid, dis, tis)
                if imp is None:
                    continue
                if imp.shape[0] == n_cells:
                    imp = imp.T
                elif imp.shape[1] != n_cells:
                    raise ValueError(f"{imp_path} shape {imp.shape}")
                if m == "MAGIC":
                    full = np.zeros((n_genes, n_cells), dtype=np.float32)
                    for gi, pos in magic_map.items():
                        if pos < imp.shape[0]:
                            full[gi] = imp[pos]
                    imp = full
                consensus += zscore_rows(imp)
                n_used += 1
            if n_used == 0:
                continue
            consensus /= n_used
            raw = sub.X
            raw = raw.toarray() if sparse.issparse(raw) else raw
            raw = raw.T
            zeros = (raw == 0)
            if mode == "filled":
                final = raw.copy(); final[zeros] = consensus[zeros]
            else:
                final = np.zeros_like(raw, dtype=np.float32); final[zeros] = consensus[zeros]
            subdir_raw  = RAW_ROOT  / dsid / norm_fname(dis)
            subdir_cons = CONS_ROOT / dsid / norm_fname(dis)
            subdir_raw.mkdir (parents=True, exist_ok=True)
            subdir_cons.mkdir(parents=True, exist_ok=True)
            fname = f"{norm_fname(tis)}.csv"
            pd.DataFrame(raw,   index=sub.var_names, columns=sub.obs_names).to_csv(subdir_raw  / fname)
            pd.DataFrame(final, index=sub.var_names, columns=sub.obs_names).to_csv(subdir_cons / fname)

if __name__ == "__main__":
    export_consensus(os.environ.get("MODE", "filled").lower())
