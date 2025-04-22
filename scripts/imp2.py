from collections import defaultdict
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
        
        run_imputation(adata, f.stem, methods)
        
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
def run_analysis(adata, dataset_id, methods):
    metadata_markers = adata.obs[["disease", "tissue"]].drop_duplicates()
    
    analysis_data = {}
    analysis_data_path = os.path.join("output", "analysis", dataset_id + ".json")
    if os.path.exists(analysis_data_path):
        with open(analysis_data_path, "r") as in_f:
            analysis_data = json.load(in_f)
    
    else:
        os.makedirs(os.path.dirname(analysis_data_path), exist_ok=True)
    
    # assume that the imputation methods were all run correctly and the subset results are present, splitting
    i = 0
    for _, (disease, tissue) in metadata_markers.iterrows():
        method_imputed = {}
        method_deltas = {}
        
        i += 1
        print(f"Analyzing sample {i}/{len(metadata_markers)}: {tissue} ({disease})")
        
        mask = (adata.obs["disease"] == disease) & (adata.obs["tissue"] == tissue)
        adata_sample = adata[mask]
        input = adata_sample.X.toarray()
        print(f"- Sample matrix has shape {input.shape}")
        
        # `sample_analysis` refers to data about the entire sample
        sample_analysis = analysis_data.setdefault(f"{disease}, {tissue}", {})
        sample_analysis["shape"] = input.shape
        
        nonzero_sums = np.sum(input, axis=0)
        nonzero_counts = np.sum(input != 0, axis=0)
        nonzero_counts = np.where(nonzero_counts > 0, nonzero_counts, 1)
        delta_per_j = nonzero_sums / nonzero_counts
        
        originally_zero = input == 0
        for method_name in methods.keys():
            if method_name not in sample_analysis:
                sample_analysis[method_name] = {}  # <-- ADD THIS

            imputed = np.load(os.path.join("output", method_name, dataset_id, disease, tissue + ".npy"))
            if method_name == "MAGIC":
                imputed = expand_magic_matrix(input, imputed)
            # Compute delta for this method and sample
            imputed_values = imputed[originally_zero]
            delta = compute_delta(imputed_values)
            method_imputed[method_name] = imputed
            method_deltas[method_name] = delta

        # Compute sum_masks for truthfulness (original zeros)
        sum_masks = np.zeros(originally_zero.shape, dtype=float)
        for method_name in methods.keys():
            if method_name not in sample_analysis:
                sample_analysis[method_name] = {}  # <-- ADD THIS

            mask = (method_imputed[method_name] <= method_deltas[method_name])
            sum_masks += mask.astype(float)
        truthfulness_zero = sum_masks[originally_zero] / len(methods)
        avg_truthfulness_zero = truthfulness_zero.mean()

        # Calculate per-method truthfulness
        for method_name in methods.keys():
            if method_name not in sample_analysis:
                sample_analysis[method_name] = {}  # <-- ADD THIS

            # For non-zero entries
            mask_non_zero = (method_imputed[method_name] > method_deltas[method_name]) & (input > 0)
            truthfulness_non_zero = mask_non_zero.mean()
            # Combined average
            total_cells = input.size
            avg_truthfulness = (truthfulness_non_zero * np.sum(input > 0) + avg_truthfulness_zero * np.sum(originally_zero)) / total_cells
            sample_analysis[method_name]["truthfulness"] = avg_truthfulness

        # Pairwise calculations for Concordance and Thresh_Conc
        for index_a, method_a in enumerate(methods.keys()):
            for index_b, method_b in enumerate(methods.keys()):
                if method_a not in sample_analysis:
                    sample_analysis[method_a] = {}
                if index_b <= index_a:
                    continue
                # Get imputed matrices and deltas
                imputed_a = method_imputed[method_a]
                delta_a = method_deltas[method_a]
                imputed_b = method_imputed[method_b]
                delta_b = method_deltas[method_b]
                
                # the matrix should be a list of lists
                # first index indicates row, so by comparing to a 1d array, we effectively compare each row
                # to the means array, which then compares each element of the row to each element of the means
                # both_imputed_d = originally_zero & (imputed_a > delta_per_j) & (imputed_b > delta_per_j)
                # only_a_imputed_d = originally_zero & (imputed_a > delta_per_j) & ~(imputed_b > delta_per_j)
                # only_b_imputed_d = originally_zero & ~(imputed_a > delta_per_j) & (imputed_b > delta_per_j)
                # neither_imputed_d = originally_zero & ~(imputed_a > delta_per_j) & ~(imputed_b > delta_per_j)
                both_imputed_d = originally_zero & (imputed_a > delta_a) & (imputed_b > delta_b)
                only_a_imputed_d = originally_zero & (imputed_a > delta_a) & ~(imputed_b > delta_b)
                only_b_imputed_d = originally_zero & ~(imputed_a > delta_a) & (imputed_b > delta_b)
                neither_imputed_d = originally_zero & ~(imputed_a > delta_a) & ~(imputed_b > delta_b)

                both_imputed_0 = originally_zero & (imputed_a > 0) & (imputed_b > 0)
                only_a_imputed_0 = originally_zero & (imputed_a > 0) & ~(imputed_b > 0)
                only_b_imputed_0 = originally_zero & ~(imputed_a > 0) & (imputed_b > 0)
                neither_imputed_0 = originally_zero & ~(imputed_a > 0) & ~(imputed_b > 0)

                # Calculate Concordance (0,0)
                both_imputed_0 = originally_zero & (imputed_a > 0) & (imputed_b > 0)
                print(both_imputed_0)
                conc_p = np.sum(both_imputed_0) / np.sum(originally_zero)
                # Calculate conc_p_d (delta_a, delta_b)
                both_imputed_d = originally_zero & (imputed_a > delta_a) & (imputed_b > delta_b)
                conc_p_d = np.sum(both_imputed_d) / np.sum(originally_zero)
                # Mutual Concordance p
                mutual_0_or_0 = np.sum(both_imputed_0) + np.sum(only_a_imputed_0) + np.sum(only_b_imputed_0)
                mutual_d_or_d = np.sum(both_imputed_d) + np.sum(only_a_imputed_d) + np.sum(only_b_imputed_d)
                print("both_imputed_d: ", both_imputed_d)
                ### HERE IS CONCORDANCE CALCULATIONS
                # mutual concordance: and/or
                mutual_00_and_or = both_imputed_0 / mutual_0_or_0
                mutual_dd_and_or = both_imputed_d / mutual_d_or_d

                sample_analysis[method_a][method_b]["contingency"] = {
                    "conc_p": conc_p,
                    "thresh_conc_p": conc_p_d,
                    "mutual_conc_0_or_0": mutual_0_or_0,
                    "mutual_conc_0_and_0": int(np.sum(both_imputed_0)),
                    "mutual_conc_d_or_d": mutual_d_or_d,
                    "mutual_conc_d_and_d": int(np.sum(both_imputed_d)),
                    "both_confident": int(np.sum(both_imputed_d)),
                    "only_method_a_confident": int(np.sum(only_a_imputed_d)),
                    "only_method_b_confident": int(np.sum(only_b_imputed_d)),
                    "neither_confident": int(np.sum(neither_imputed_d)),
                    "both_imputed": int(np.sum(both_imputed_0)),
                    "only_method_a_imputed": int(np.sum(only_a_imputed_0)),
                    "only_method_b_imputed": int(np.sum(only_b_imputed_0)),
                    "neither_imputed": int(np.sum(neither_imputed_0)),
                    "originally_zero": int(np.sum(originally_zero)),
                }
                print("- Finding concordance values")
                
                if sample_analysis[method_a][method_b].get("per_gene_concordance") == None:
                    print(" - Calculating per-gene concordance values")
                    # this adds the booleans vertically, generating a list of length n-genes where each value
                    # is the number of cells that got imputed divided by the number of original zeroes
                    # for that specific gene column
                    concordance_per_gene = np.sum(both_imputed_d, axis=0) / np.sum(originally_zero, axis=0)
                    
                    # here we just figure out what the original gene names were,
                    labeled = list(zip(adata_sample.var["feature_name"], concordance_per_gene))
                    
                    # filter out any genes that didn't get imputed above the threshold at all,
                    filtered = [t for t in labeled if t[1] > 0]
                    
                    # and then sort, with the most concordant at the top
                    # UNTESTED
                    #sorted_list = sorted(filtered, key=itemgetter(1), reverse=True)
                    
                    sample_analysis[method_a][method_b]["per_gene_concordance"] = filtered
                        
                if sample_analysis[method_a][method_b].get("per_cell_concordance") == None:
                    print(" - Calculating per-cell concordance values")
                    concordance_per_cell = np.sum(both_imputed_d, axis=1) / np.sum(originally_zero, axis=1)
                    
                    # do the same thing as above, just with more care since we are summing across columns,
                    # not rows
                    cell_labels = [f"{m} ({n})" for m, n in 
                                    zip(adata_sample.obs["Celltype"], adata_sample.obs.index)]
                    labeled = list(zip(cell_labels, concordance_per_cell))
                    filtered = [t for t in labeled if t[1] > 0]
                    #sorted_list = sorted(filtered, key=itemgetter(1), reverse=True)
                    sample_analysis[method_a][method_b]["per_cell_concordance"] = filtered
        
        # prematurely dump the json after testing every sample for easy monitoring
        with open(analysis_data_path, "w") as out_f:
            json.dump(analysis_data, out_f, indent=4)
            
        force_gc()
        print("")

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

# run_pipeline_on_sets()

# Add this after the analysis data is saved
def print_distributions(analysis_data_path):
    with open(analysis_data_path, "r") as f:
        data = json.load(f)
    
    for sample_name, sample in data.items():
        if sample_name == "avg":
            continue
            
        print(f"\nSample: {sample_name}")
        print(f"Original - Non-zero: {sample['original_distribution']['non_zero']:,} | Zero: {sample['original_distribution']['zero']:,}")
        
        for method in sample:
            if method == "original_distribution" or method == "shape":
                continue
                
            stats = sample[method].get("imputation_distribution")
            if stats:
                print(f"{method}:")
                print(f"  Imputed from zero: {stats['from_zero']:,} ({stats['from_zero']/sample['original_distribution']['zero']:.1%})")
                print(f"  Over-imputed: {stats['from_non_zero']:,} ({stats['from_non_zero']/sample['original_distribution']['non_zero']:.1%})")

# Call this after analysis completes
print_distributions("./datasets/extracted.json")