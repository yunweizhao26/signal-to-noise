import os
import json
import numpy as np

from pathlib import Path
from anndata import read_h5ad
from sklearn.cluster import KMeans

def force_gc():
    """
    Optional utility to force a garbage collection and
    print memory usage. Only needed if you suspect memory leaks.
    """
    import gc, psutil
    process = psutil.Process(os.getpid())
    before = process.memory_info().rss / (1024 * 1024)
    collected = gc.collect()
    after = process.memory_info().rss / (1024 * 1024)
    print(f"GC collected {collected} objects, memory: {before:.2f}MB -> {after:.2f}MB")

def run_pipeline_on_sets():
    """
    Master pipeline:
      1) Locate all h5ad files under ./datasets/*.h5ad
      2) For each file:
         a) Read adata
         b) Filter to is_primary_data
         c) Run imputation (for all methods)
         d) Run analysis (including truthfulness, golden clustering, etc.)
    """
    files = list(Path("./datasets").glob("*.h5ad"))
    
    # Define your methods (function references) here.
    # Example placeholders. You can replace with your actual _run_xxx functions.
    methods = {
        "SAUCIE": _run_saucie,
        "MAGIC": _run_magic,
    }

    for f in files:
        dataset_id = f.stem
        print(f"\n=== Processing dataset {dataset_id} ===")

        adata = read_h5ad(f)
        # filter out cells that originate from other datasets
        adata = adata[adata.obs["is_primary_data"] == True]

        # 1) Run imputation
        run_imputation(adata, dataset_id, methods)

        # 2) Run extended analysis
        run_analysis(adata, dataset_id, methods)

        # Optional GC
        force_gc()

def run_imputation(adata, dataset_id, methods):
    """
    For each combination of disease + tissue in adata,
    run each imputation method, store results to output directory.
    """
    metadata_markers = adata.obs[["disease", "tissue"]].drop_duplicates()
    i = 0
    for _, (disease, tissue) in metadata_markers.iterrows():
        i += 1
        print(f"[Imputation] {i}/{len(metadata_markers)}: disease={disease}, tissue={tissue}")

        mask = (adata.obs["disease"] == disease) & (adata.obs["tissue"] == tissue)
        matrix_input = adata[mask].X.toarray()  # shape: (cells, genes)
        print(f" - shape={matrix_input.shape}")

        for name, method_func in methods.items():
            outpath = os.path.join("output", name, dataset_id, disease, tissue + ".npy")
            if os.path.exists(outpath):
                print(f"   - Method={name} already done.")
            else:
                os.makedirs(os.path.dirname(outpath), exist_ok=True)
                print(f"   - Running method={name} ...")
                result = method_func(matrix_input)
                np.save(outpath, result)
                print(f"   - Saved results to {outpath}")

def _run_saucie(y):
    """
    Placeholder for the real SAUCIE code or import usage.
    """
    import importlib
    tf = importlib.import_module('tensorflow.compat.v1')
    tf.disable_v2_behavior()

    import baselines.SAUCIE.SAUCIE as SAUCIE
    from baselines.SAUCIE.SAUCIE import Loader

    tf.reset_default_graph()
    saucie_model = SAUCIE.SAUCIE(y.shape[1])
    saucie_model.train(Loader(y, shuffle=True), steps=1000)

    rec_y = saucie_model.get_reconstruction(Loader(y, shuffle=False))
    return rec_y

def _run_magic(y):
    """
    Placeholder for the real MAGIC code or usage.
    """
    import baselines.MAGIC.magic as magic
    import scprep

    # Filter & sqrt transform
    y_filtered = scprep.filter.filter_rare_genes(y, min_cells=5)
    y_norm = scprep.transform.sqrt(
        scprep.normalize.library_size_normalize(y_filtered)
    )

    magic_op = magic.MAGIC(t=7, n_pca=20, n_jobs=-1)
    y_hat = magic_op.fit_transform(y_norm, genes='all_genes')

    # Expand back to original shape (some columns might have been dropped)
    expanded = expand_magic_matrix(y, y_hat)
    return expanded

def expand_magic_matrix(y_original, y_magic):
    """
    If MAGIC filtered out columns, re-insert them with
    the original values to keep consistent shape.
    """
    expanded = np.zeros_like(y_original)
    nonzero_counts = (y_original != 0).sum(axis=0)
    kept_columns = np.where(nonzero_counts >= 5)[0]
    removed_columns = np.where(nonzero_counts < 5)[0]

    # Place the magic-imputed columns back
    col_i = 0
    for c in range(y_original.shape[1]):
        if c in kept_columns:
            expanded[:, c] = y_magic[:, col_i]
            col_i += 1
        else:
            # revert to original
            expanded[:, c] = y_original[:, c]
    return expanded

def run_analysis(adata, dataset_id, methods):
    """
    Extended analysis with:
      - Load each method's imputed results
      - Compute TruthfulnessHypothesis matrix
      - Compute per-gene and per-cell averages
      - Perform Golden Clustering experiment
      - Save everything in a JSON
    """
    analysis_out_dir = os.path.join("output", "analysis")
    os.makedirs(analysis_out_dir, exist_ok=True)
    analysis_data_path = os.path.join(analysis_out_dir, dataset_id + ".json")

    # attempt to load existing analysis (if any)
    if os.path.exists(analysis_data_path):
        with open(analysis_data_path, "r") as in_f:
            analysis_data = json.load(in_f)
    else:
        analysis_data = {}

    metadata_markers = adata.obs[["disease", "tissue"]].drop_duplicates()

    for idx, (disease, tissue) in enumerate(metadata_markers.values, start=1):
        print(f"\n[Analysis] {idx}/{len(metadata_markers)}: disease={disease}, tissue={tissue}")

        # subselect
        mask = (adata.obs["disease"] == disease) & (adata.obs["tissue"] == tissue)
        adata_sub = adata[mask]
        input_matrix = adata_sub.X.toarray()  # shape: (cells, genes)
        n_cells, n_genes = input_matrix.shape

        sample_key = f"{disease},{tissue}"
        if sample_key not in analysis_data:
            analysis_data[sample_key] = {}

        sample_analysis = analysis_data[sample_key]
        sample_analysis["shape"] = [n_cells, n_genes]

        # -----------------------------------------------------
        # 1) Load method predictions
        # -----------------------------------------------------
        # We'll store each method's predicted matrix so we can compare
        method_predictions = []
        method_names = list(methods.keys())

        for method_name in method_names:
            path = os.path.join("output", method_name, dataset_id, disease, tissue + ".npy")
            if not os.path.exists(path):
                print(f"   - Method={method_name} does not have results. Skipping.")
                continue
            pred_matrix = np.load(path)
            if pred_matrix.shape != (n_cells, n_genes):
                print(f"   - WARNING: method={method_name} shape mismatch {pred_matrix.shape}, expected {(n_cells, n_genes)}")
            method_predictions.append((method_name, pred_matrix))

        if len(method_predictions) == 0:
            print("   - No methods found for this subset. Skipping analysis.")
            continue

        # -----------------------------------------------------
        # 2) Compute TruthfulnessHypothesis (TH)
        # -----------------------------------------------------
        # TH[i,j] = 1 if X[i,j] > 0
        #         = (count of methods that predicted 0) / num_methods if X[i,j] == 0
        # For efficiency, we'll do:
        #   - Initialize TH with ones
        #   - Stack all method preds, count how many are zero at each (i,j)
        X_is_zero = (input_matrix == 0)  # shape (cells, genes)
        TH = np.ones_like(input_matrix, dtype=float)

        # Stack the predictions: shape (num_methods, n_cells, n_genes)
        all_preds = np.stack([pm[1] for pm in method_predictions], axis=0)
        num_methods = all_preds.shape[0]

        # For each (cell, gene), count how many methods predicted 0
        zero_count = np.sum(all_preds == 0, axis=0)  # shape (n_cells, n_genes)

        # only assign fraction to TH where original X is zero
        TH[X_is_zero] = zero_count[X_is_zero] / float(num_methods)

        # store TH in analysis_data or partial stats
        # We'll store just the shape or if you want the entire TH matrix, it can be huge
        # sample_analysis["TH_matrix"] = TH.tolist()  # optional, be cautious with large data

        # per-gene average: shape = (genes,)
        TH_gene_averages = TH.mean(axis=0)  # averaging over cells
        # per-cell average: shape = (cells,)
        TH_cell_averages = TH.mean(axis=1)  # averaging over genes

        sample_analysis["TH_gene_averages"] = TH_gene_averages.tolist()
        sample_analysis["TH_cell_averages"] = TH_cell_averages.tolist()

        # -----------------------------------------------------
        # 3) Golden Clustering
        # -----------------------------------------------------
        # Steps:
        #   a) Flatten TH, sort by descending value
        #   b) for k in [1..some limit], union top-k with M_p
        # We interpret M_p as the union over a single method's binary matrix OR an all-zero matrix, 
        # depending on your usage. Let's assume we do it on an all-zero start for simplicity.

        # Flatten and sort indices by TH, descending
        flat_TH = TH.ravel()  # shape = n_cells*n_genes
        sorted_indices_desc = np.argsort(-flat_TH)  # descending order
        # (i, j) pairs
        idx_pairs = [np.unravel_index(fi, (n_cells, n_genes)) for fi in sorted_indices_desc]

        # Let's define M_p(0) = all-zero matrix
        M_pk = np.zeros_like(input_matrix, dtype=np.uint8)  # shape (cells, genes)

        # We'll do a step-based approach (since doing for all k can be huge)
        # For demonstration, let's do up to some maximum or step in increments
        max_k = min(len(idx_pairs), 5000)  # pick a cap to not overdo computations
        step_size = 500
        cluster_results = []

        # pick number of clusters (N). Let's guess from adata_sub.obs['Celltype'] if available
        if "Celltype" in adata_sub.obs.columns:
            N = len(adata_sub.obs["Celltype"].unique())
        else:
            N = 5  # fallback

        for k in range(step_size, max_k+1, step_size):
            # union top-k with the current M_pk
            # We only need to add the new chunk from [k-step_size, k)
            for add_idx in range(k-step_size, k):
                if add_idx >= len(idx_pairs):
                    break
                (r, c) = idx_pairs[add_idx]
                M_pk[r, c] = 1

            # Now cluster the cells (row=cell, col=gene). We treat each cell as a vector of gene features
            # shape of M_pk is (cells, genes), so each row is a cell.
            # KMeans wants shape (samples, features) => M_pk
            # We'll do standard K-means with random_state for reproducibility
            kmeans = KMeans(n_clusters=N, n_init=10, random_state=42)
            labels = kmeans.fit_predict(M_pk)

            # Evaluate cluster radius
            # centers: shape (N, genes)
            centers = kmeans.cluster_centers_
            # compute largest distance from center within each cluster
            max_radius = 0.0
            for cluster_id in range(N):
                cell_indices = np.where(labels == cluster_id)[0]
                if len(cell_indices) == 0:
                    continue
                cluster_cells = M_pk[cell_indices]  # shape (X, genes)
                diffs = cluster_cells - centers[cluster_id]  # shape (X, genes)
                # L2 norm across axis=1
                dists = np.linalg.norm(diffs, axis=1)
                cluster_max = dists.max()
                if cluster_max > max_radius:
                    max_radius = cluster_max
            
            cluster_results.append({
                "k": k,
                "radius": float(max_radius)
            })
            print(f"   [GoldenClustering] k={k}, radius={max_radius:.4f}")

        sample_analysis["golden_clustering"] = {
            "n_clusters": N,
            "step_size": step_size,
            "max_k": max_k,
            "cluster_results": cluster_results
        }

        # -----------------------------------------------------
        # done analyzing this subset. Save partial results
        # -----------------------------------------------------
        analysis_data[sample_key] = sample_analysis
        with open(analysis_data_path, "w") as out_f:
            json.dump(analysis_data, out_f, indent=2)

    print(f"Analysis finished for dataset {dataset_id}.\n")
