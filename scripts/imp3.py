from pathlib import Path
import importlib
import os
import numpy as np
from anndata import read_h5ad
#from scipy.sparse import dok_matrix
import json
from operator import itemgetter


import baselines.SAUCIE.SAUCIE as SAUCIE
def _run_saucie(
    y,
):
    tf = importlib.import_module('tensorflow.compat.v1')
    tf.disable_v2_behavior()
    
    tf.reset_default_graph()
    saucie_model = SAUCIE.SAUCIE(y.shape[1])
    saucie_model.train(SAUCIE.Loader(y, shuffle=True), steps=1000)
    
    rec_y = saucie_model.get_reconstruction(SAUCIE.Loader(y, shuffle=False))
    
    return rec_y


import baselines.MAGIC.magic as magic
import scprep
def _run_magic(
    y,
):
    # Preprocess data
    y_filtered = scprep.filter.filter_rare_genes(y, min_cells=5)
    y_norm = scprep.transform.sqrt(
        scprep.normalize.library_size_normalize(y_filtered)
    )
    
    magic_op = magic.MAGIC(
        # knn=5,
        # knn_max=None,
        # decay=1,
        # Variable changed in paper
        t=7, #[2,7,'auto']
        n_pca=20,
        # solver="exact",
        # knn_dist="euclidean",
        n_jobs=-1,
        # random_state=None,
        # verbose=1,
    )
    y_hat = magic_op.fit_transform(y_norm, genes='all_genes')
    
    return y_hat

def expand_magic_matrix(y, reduced_matrix):
    expanded = np.zeros(y.shape)
    
    nonzero_counts = (y != 0).sum(axis=0)
    kept_columns = np.where(nonzero_counts >= 5)[0]
    removed_columns = np.where(nonzero_counts < 5)[0]
    
    # Copy values from reduced matrix to their original positions
    for j_reduced, j_original in enumerate(kept_columns):
        expanded[:, j_original] = reduced_matrix[:, j_reduced]
        
    for _, j_original in enumerate(removed_columns):
        expanded[:, j_original] = y[:, j_original]
    
    return expanded


import baselines.scScope.scscope.scscope as scScope
def _run_scscope(
    y,
) -> None:
    model = scScope.train(
          y,
          15,
          use_mask=True,
          batch_size=64,
          max_epoch=1000,
          epoch_per_check=100,
          T=2,
          exp_batch_idx_input=[],
          encoder_layers=[],
          decoder_layers=[],
          learning_rate=0.0001,
          beta1=0.05,
          num_gpus=1)
    
    _, rec_y, _ = scScope.predict(y, model, batch_effect=[])
    return rec_y


from pandas import DataFrame
from baselines.deepimpute.deepimpute import multinet
def _run_deepimpute(
    y,
):  
    print(1)
    multinet = importlib.import_module('baselines.deepimpute.deepimpute.multinet')
    print(2)

    tf = importlib.import_module('tensorflow.compat.v1')
    print(3)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print(4)

    y_pd = DataFrame(y)
    print("y_pd.shape", y_pd.shape)
    print(5)

    model = multinet.MultiNet()
    print(6)

    # Get killed
    model.fit(y_pd, cell_subset=1, minVMR=0.5)
    print(7)

    imputed_data = model.predict(y_pd)
    print(8)

    return imputed_data.to_numpy()


from baselines.scvi.scvi import run_scvi
def _run_scvi(
    y,
) -> None:
    return run_scvi(y)


def check_mem_usage():
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Current memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")

def force_gc():
    import gc
    check_mem_usage()
    print(f"collecting {gc.collect()} objects")
    check_mem_usage()
    

def run_pipeline_on_sets():
    files = list(Path("./datasets").glob("*.h5ad"))
    
    methods = {
        # "deepImpute": _run_deepimpute,
        # "scScope": _run_scscope, 
        # "scVI": _run_scvi,
        "SAUCIE": _run_saucie,
        "MAGIC": _run_magic, 
    }
    
    for f in files:
        print(f"Processing dataset {f.stem}")
        adata = read_h5ad(f)
        
        # filter out cells that originate from other datasets
        adata = adata[adata.obs["is_primary_data"] == True]
        
        # run_imputation(adata, f.stem, methods)
        
        run_analysis(adata, f.stem, methods)

# for each of the disease+tissue subsets in this dataset, apply the imputation methods
# and save the results to output to "output/imputation/dataset/disease/tissue.npy"
def run_imputation(adata, dataset_id, methods):
    metadata_markers = adata.obs[["disease", "tissue"]].drop_duplicates()
    
    i = 0
    for _, (disease, tissue) in metadata_markers.iterrows():
        i += 1
        print(f"Imputing sample {i}/{len(metadata_markers)}: {tissue} ({disease})")
        
        mask = (adata.obs["disease"] == disease) & (adata.obs["tissue"] == tissue)
        input = adata[mask].X.toarray()
        print(f"- Sample matrix has shape {input.shape}")
        
        for name, method in methods.items():
            path = os.path.join("output", name, dataset_id, disease, tissue + ".npy")
            
            if os.path.exists(path):
                print(f"- Imputation method {name} already ran")
                result = None
            
            else:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                print(f"- Running imputation method {name}")
                
                result = method(input)
                np.save(path, result)
                force_gc()

def compute_delta(imputed_values, bins=100):
    if len(imputed_values) == 0:
        return 0.0
    hist, bin_edges = np.histogram(imputed_values, bins=bins)
    print(f"Histogram: {hist}")
    print(f"Bin edges: {bin_edges}")
    max_bin = np.argmax(hist)
    return bin_edges[max_bin]

# calculate concordance measures
def run_analysis(adata, dataset_id, methods, delta_per_j=0.0):
    import json, os, numpy as np
    import gc
    
    def check_mem_usage():
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        print(f"Current memory usage: {memory_info.rss / (1024*1024):.2f} MB")

    def force_gc():
        check_mem_usage()
        print(f"collecting {gc.collect()} objects")
        check_mem_usage()
    
    def expand_magic_matrix(original_y, reduced_matrix):
        expanded = np.zeros(original_y.shape)
        nonzero_counts = (original_y != 0).sum(axis=0)
        kept_columns = np.where(nonzero_counts >= 5)[0]
        removed_columns = np.where(nonzero_counts < 5)[0]
        for j_reduced, j_original in enumerate(kept_columns):
            expanded[:, j_original] = reduced_matrix[:, j_reduced]
        for j_original in removed_columns:
            expanded[:, j_original] = original_y[:, j_original]
        return expanded
    
    # Number of random draws per subset
    NUM_SIMULATIONS = 100
    
    # Gather subsets
    metadata_markers = adata.obs[["disease", "tissue"]].drop_duplicates()

    # Attempt to load prior analysis data
    analysis_data = {}
    analysis_data_path = os.path.join("output", "analysis", dataset_id + ".json")
    if os.path.exists(analysis_data_path):
        with open(analysis_data_path, "r") as in_f:
            analysis_data = json.load(in_f)
    else:
        os.makedirs(os.path.dirname(analysis_data_path), exist_ok=True)
    
    # Column in .obs for labeling cells (if none, pick a fallback)
    celltype_col = "Celltype" if "Celltype" in adata.obs.columns else adata.obs.columns[0]

    i = 0
    for _, (disease, tissue) in metadata_markers.iterrows():
        i += 1
        print(f"Analyzing sample {i}/{len(metadata_markers)}: {tissue} ({disease})")
        
        mask = (adata.obs["disease"] == disease) & (adata.obs["tissue"] == tissue)
        adata_sample = adata[mask]
        input_arr = adata_sample.X.toarray()
        print(f"- Sample matrix has shape {input_arr.shape}")
        
        sample_key = f"{disease}, {tissue}"
        sample_analysis = analysis_data.setdefault(sample_key, {})
        sample_analysis["shape"] = [int(x) for x in input_arr.shape]
        
        originally_zero = (input_arr == 0)
        num_orig_zeros = np.sum(originally_zero)
        
        # Load each method’s imputed results
        method_imputations = {}
        for name in methods.keys():
            path = os.path.join("output", name, dataset_id, disease, tissue + ".npy")
            if not os.path.exists(path):
                print(f"WARNING: missing imputation result for {name} at {path}")
                continue
            mat = np.load(path)
            if name == "MAGIC":
                mat = expand_magic_matrix(input_arr, mat)
            method_imputations[name] = mat
        
        all_method_names = list(method_imputations.keys())
        for idx_a, method_a in enumerate(all_method_names):
            if method_a not in sample_analysis:
                sample_analysis[method_a] = {}
            
            imputed_a = method_imputations[method_a]
            
            for idx_b in range(idx_a, len(all_method_names)):
                method_b = all_method_names[idx_b]
                
                if method_b not in sample_analysis[method_a]:
                    sample_analysis[method_a][method_b] = {}
                if method_a == method_b:
                    # same method => skip
                    sample_analysis[method_a][method_b] = 1
                    continue
                
                imputed_b = method_imputations[method_b]
                
                # ==================== Actual Overlap ======================
                both_confident_d = (
                    originally_zero & 
                    (imputed_a > delta_per_j) & 
                    (imputed_b > delta_per_j)
                )
                only_a_confident_d = (
                    originally_zero & 
                    (imputed_a > delta_per_j) & 
                    ~(imputed_b > delta_per_j)
                )
                only_b_confident_d = (
                    originally_zero & 
                    ~(imputed_a > delta_per_j) & 
                    (imputed_b > delta_per_j)
                )
                neither_confident_d = (
                    originally_zero & 
                    ~(imputed_a > delta_per_j) & 
                    ~(imputed_b > delta_per_j)
                )

                both_imputed_n = (
                    originally_zero & 
                    (imputed_a > 0) & 
                    (imputed_b > 0)
                )
                only_a_imputed_n = (
                    originally_zero & 
                    (imputed_a > 0) & 
                    ~(imputed_b > 0)
                )
                only_b_imputed_n = (
                    originally_zero & 
                    ~(imputed_a > 0) & 
                    (imputed_b > 0)
                )
                neither_imputed_n = (
                    originally_zero & 
                    ~(imputed_a > 0) & 
                    ~(imputed_b > 0)
                )
                
                contingency_actual = {
                    "both_confident": int(np.sum(both_confident_d)),
                    "only_method_a_confident": int(np.sum(only_a_confident_d)),
                    "only_method_b_confident": int(np.sum(only_b_confident_d)),
                    "neither_confident": int(np.sum(neither_confident_d)),
                    
                    "both_imputed": int(np.sum(both_imputed_n)),
                    "only_method_a_imputed": int(np.sum(only_a_imputed_n)),
                    "only_method_b_imputed": int(np.sum(only_b_imputed_n)),
                    "neither_imputed": int(np.sum(neither_imputed_n)),
                    
                    "originally_zero": int(num_orig_zeros)
                }
                sample_analysis[method_a][method_b]["contingency"] = contingency_actual
                
                if num_orig_zeros > 0:
                    total_concordance = (
                        contingency_actual["both_confident"] / num_orig_zeros
                    )
                else:
                    total_concordance = 0
                sample_analysis[method_a][method_b]["total_concordance"] = total_concordance
                print(f" - {method_a} vs {method_b}, total_concordance = {100*total_concordance:.2f}%")
                
                # ============ Per-Gene Concordance (Actual) ============
                if "per_gene_concordance" not in sample_analysis[method_a][method_b]:
                    orig_zero_by_gene = np.sum(originally_zero, axis=0)
                    both_confident_by_gene = np.sum(both_confident_d, axis=0)
                    per_gene = np.divide(
                        both_confident_by_gene,
                        orig_zero_by_gene,
                        out=np.zeros_like(both_confident_by_gene, dtype=float),
                        where=(orig_zero_by_gene != 0)
                    )
                    gene_labels = adata_sample.var["feature_name"]
                    labeled = list(zip(gene_labels, per_gene))
                    sample_analysis[method_a][method_b]["per_gene_concordance"] = labeled
                
                # ============ Per-Cell Concordance (Actual) ============
                if "per_cell_concordance" not in sample_analysis[method_a][method_b]:
                    orig_zero_by_cell = np.sum(originally_zero, axis=1)
                    both_confident_by_cell = np.sum(both_confident_d, axis=1)
                    per_cell = np.divide(
                        both_confident_by_cell,
                        orig_zero_by_cell,
                        out=np.zeros_like(both_confident_by_cell, dtype=float),
                        where=(orig_zero_by_cell != 0)
                    )
                    cell_labels = [
                        f"{ctype} ({cid})"
                        for ctype, cid in zip(adata_sample.obs[celltype_col], adata_sample.obs.index)
                    ]
                    labeled = list(zip(cell_labels, per_cell))
                    sample_analysis[method_a][method_b]["per_cell_concordance"] = labeled
                
                # ============== MULTIPLE Bayesian Overlap ================
                total_imputed_by_a = (
                    contingency_actual["only_method_a_confident"] +
                    contingency_actual["both_confident"]
                )
                total_imputed_by_b = (
                    contingency_actual["only_method_b_confident"] +
                    contingency_actual["both_confident"]
                )
                
                if num_orig_zeros > 0:
                    p_a = total_imputed_by_a / num_orig_zeros
                    p_b = total_imputed_by_b / num_orig_zeros
                else:
                    p_a, p_b = 0.0, 0.0
                
                # We'll accumulate random contingency counts over many draws
                random_both_sum = 0
                random_only_a_sum = 0
                random_only_b_sum = 0
                random_neither_sum = 0
                
                # For gene/cell-level sums
                random_both_by_gene_sum = np.zeros(input_arr.shape[1], dtype=int)
                random_both_by_cell_sum = np.zeros(input_arr.shape[0], dtype=int)
                
                for _ in range(NUM_SIMULATIONS):
                    random_a = np.zeros_like(input_arr, dtype=bool)
                    random_b = np.zeros_like(input_arr, dtype=bool)
                    
                    rand_vals_a = np.random.rand(num_orig_zeros)
                    rand_vals_b = np.random.rand(num_orig_zeros)
                    
                    random_a[originally_zero] = (rand_vals_a < p_a)
                    random_b[originally_zero] = (rand_vals_b < p_b)
                    
                    random_both_confident = random_a & random_b & originally_zero
                    random_only_a_confident = random_a & (~random_b) & originally_zero
                    random_only_b_confident = random_b & (~random_a) & originally_zero
                    random_neither_confident = (~random_a) & (~random_b) & originally_zero
                    
                    # Tally for "contingency_random"
                    random_both_sum    += np.sum(random_both_confident)
                    random_only_a_sum  += np.sum(random_only_a_confident)
                    random_only_b_sum  += np.sum(random_only_b_confident)
                    random_neither_sum += np.sum(random_neither_confident)
                    
                    # Tally for gene/cell-level
                    random_both_by_gene_sum += np.sum(random_both_confident, axis=0)
                    random_both_by_cell_sum += np.sum(random_both_confident, axis=1)
                
                # Average them out (round to int for final contingency)
                random_both_avg    = int(np.round(random_both_sum / NUM_SIMULATIONS))
                random_only_a_avg  = int(np.round(random_only_a_sum / NUM_SIMULATIONS))
                random_only_b_avg  = int(np.round(random_only_b_sum / NUM_SIMULATIONS))
                random_neither_avg = int(np.round(random_neither_sum / NUM_SIMULATIONS))
                
                # Build final "contingency_random"
                contingency_random = {
                    "both_imputed": random_both_avg,
                    "only_method_a_imputed": random_only_a_avg,
                    "only_method_b_imputed": random_only_b_avg,
                    "neither_imputed": random_neither_avg
                }
                sample_analysis[method_a][method_b]["contingency_random"] = contingency_random
                
                # Average for gene/cell
                random_both_by_gene_avg = random_both_by_gene_sum / NUM_SIMULATIONS
                random_both_by_cell_avg = random_both_by_cell_sum / NUM_SIMULATIONS
                
                # Convert to fraction of zeros for total
                avg_both_total = np.sum(random_both_by_gene_avg)  # sum across all genes
                if num_orig_zeros > 0:
                    random_concordance = avg_both_total / num_orig_zeros
                else:
                    random_concordance = 0.0
                sample_analysis[method_a][method_b]["total_concordance_bayesian"] = random_concordance
                
                # =========== Per-Gene Concordance (Bayesian) ============
                orig_zero_by_gene = np.sum(originally_zero, axis=0)
                per_gene_bayes = np.divide(
                    random_both_by_gene_avg,
                    orig_zero_by_gene,
                    out=np.zeros_like(random_both_by_gene_avg, dtype=float),
                    where=(orig_zero_by_gene != 0)
                )
                gene_labels = adata_sample.var["feature_name"]
                labeled_gene_bayes = list(zip(gene_labels, per_gene_bayes))
                sample_analysis[method_a][method_b]["per_gene_concordance_bayesian"] = labeled_gene_bayes
                
                # =========== Per-Cell Concordance (Bayesian) ============
                orig_zero_by_cell = np.sum(originally_zero, axis=1)
                per_cell_bayes = np.divide(
                    random_both_by_cell_avg,
                    orig_zero_by_cell,
                    out=np.zeros_like(random_both_by_cell_avg, dtype=float),
                    where=(orig_zero_by_cell != 0)
                )
                cell_labels = [
                    f"{ctype} ({cid})"
                    for ctype, cid in zip(adata_sample.obs[celltype_col], adata_sample.obs.index)
                ]
                labeled_cell_bayes = list(zip(cell_labels, per_cell_bayes))
                sample_analysis[method_a][method_b]["per_cell_concordance_bayesian"] = labeled_cell_bayes
        
        # Save partial JSON
        with open(analysis_data_path, "w") as out_f:
            json.dump(analysis_data, out_f, indent=4)
        
        force_gc()
        print("")





def calculate_avg_concordance(files, methods):
    for f in files:
        with open(f, "r") as inp:
            analysis_data = json.load(inp)
        
        combined_values = {}
        for sample in analysis_data.values():
            for method_a in methods.keys():
                combined_values.setdefault(method_a, {})
                for method_b, concordance_val in sample[method_a].items():
                    combined_values[method_a].setdefault(method_b, []).append(concordance_val)

        avg = analysis_data["avg"] = {}
        for method_a, sub_concordances in combined_values.items():
            avg[method_a] = {}
            for method_b, values in sub_concordances.items():
                avg[method_a][method_b] = sum(values) / len(values)
                
        with open(f, "w") as outp:
            json.dump(analysis_data, outp)
        

def extract_counts_from_h5ad():
    h5ad_files = list(Path("./datasets").glob("**/*.h5ad"))
    extracted = []
    extracted_path = "./datasets/extracted.json"
    if os.path.exists(extracted_path):
        with open(extracted_path, "r") as f:
            extracted = json.load(f)
    
    for f in h5ad_files:
        if f in extracted:
            print(f"Dataset {f.stem} already extracted")
            continue
        
        print(f"Extracting dataset {f.stem}")
        
        adata = ad.read_h5ad(f)
        
        metadata_marker = "biosample_id"
        if metadata_marker not in adata.obs_keys():
            print(f"Dataset {f.stem} doesn't contain biosample_id metadata, quitting")
            continue
            metadata_marker = ["donor_id", "tissue", "disease"]
        
        # filter out cells that actually originate from other datasets
        adata = adata[adata.obs["is_primary_data"] == True]
        
        path = f"./datasets/chunked/{f.stem}"
        os.makedirs(path, exist_ok=True)
        for sample in adata.obs[metadata_marker].unique():
            file_path = path + f"/{sample}.npy"
            if os.path.exists(file_path):
                continue
            
            mask = adata.obs[metadata_marker] == sample
            np.save(file_path, adata[mask].X)
        
        extracted.append(f.stem)
        print(f"Finished extracting dataset {f.stem}")
        
    with open(extracted_path) as f:
        json.dump(extracted, extracted_path)

run_pipeline_on_sets()