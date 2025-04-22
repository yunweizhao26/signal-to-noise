from pathlib import Path
import importlib
import os
import numpy as np
from anndata import read_h5ad
#from scipy.sparse import dok_matrix
import json
from operator import itemgetter
import baselines.SAUCIE.SAUCIE as SAUCIE
import pandas as pd

marker_genes = {
  "Enterocytes BEST4": [
    "BEST4", "OTOP2", "CA7", "GUCA2A", "GUCA2B", "SPIB", "CFTR"
  ],
  "Goblet cells MUC2 TFF1": [
    "MUC2", "TFF1", "TFF3", "FCGBP", "AGR2", "SPDEF"
  ],
  "Tuft cells": [
    "POU2F3", "DCLK1"
  ],
  "Goblet cells SPINK4": [
    "MUC2", "SPINK4"
  ],
  "Enterocytes TMIGD1 MEP1A": [
    "CA1", "CA2", "TMIGD1", "MEP1A"
  ],
  "Enterocytes CA1 CA2 CA4-": [
    "CA1", "CA2"
  ],
  "Goblet cells MUC2 TFF1-": [
    "MUC2"
  ],
  "Epithelial Cycling cells": [
    "LGR5", "OLFM4", "MKI67"
  ],
  "Enteroendocrine cells": [
    "CHGA", "GCG", "GIP", "CCK"
  ],
  "Stem cells OLFM4": [
    "OLFM4", "LGR5"
  ],
  "Stem cells OLFM4 LGR5": [
    "OLFM4", "LGR5", "ASCL2"
  ],
  "Stem cells OLFM4 PCNA": [
    "OLFM4", "PCNA", "LGR5", "ASCL2", "SOX9", "TERT"
  ],
  "Paneth cells": [
    "LYZ", "DEFA5"
  ]
}



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

def compute_disagreement_per_gene(
    adata, 
    dataset_id, 
    disease, 
    tissue, 
    method_a, 
    method_b, 
    X_original, 
    X_imputed_a, 
    X_imputed_b, 
    celltype, 
    gene
):
    """
    For a single (disease, tissue) subset, given the original matrix and
    two imputed matrices (method_a, method_b),
    calculate the disagreement for one celltype + one gene.
    Returns a dictionary of counts & the final disagreement fraction.
    """
    
    # 1) Subset to the celltype
    mask_ct = (adata.obs["Celltype"] == celltype)
    X_ct_orig = X_original[mask_ct, :]
    X_ct_a = X_imputed_a[mask_ct, :]
    X_ct_b = X_imputed_b[mask_ct, :]

    # 2) Find the gene index, skip if not present
    # print("gene: ", gene)
    # print("adata.var_names: ", adata.var_names)
    if gene not in adata.var_names:
        return None
    gene_idx = adata.var_names.get_loc(gene)

    # 3) Among these cells, which are originally zero for that gene?
    expr_orig = X_ct_orig[:, gene_idx]
    originally_zero_mask = (expr_orig == 0)
    n_zero = originally_zero_mask.sum()
    # if n_zero == 0:
    #     return None  # no zero cells => no disagreement measure

    # 4) Among originally zero cells, see how each method imputes
    expr_a = X_ct_a[originally_zero_mask, gene_idx] > 0  # bool array
    expr_b = X_ct_b[originally_zero_mask, gene_idx] > 0

    onlyA = np.sum(expr_a & ~expr_b)
    onlyB = np.sum(~expr_a & expr_b)
    both  = np.sum(expr_a & expr_b)
    neither = np.sum(~expr_a & ~expr_b)

    # 5) Disagreement = (onlyA + onlyB) / n_zero
    #    You could also store concordance = both / n_zero
    disagreement = (onlyA + onlyB) / n_zero
    
    # Return a dictionary of all we want
    return {
        "dataset": dataset_id,
        "disease": disease,
        "tissue": tissue,
        "cell_type": celltype,
        "gene": gene,
        "both": int(both),
        "onlyA": int(onlyA),
        "onlyB": int(onlyB),
        "neither": int(neither),
        "originally_zero": int(n_zero),
        "disagreement": disagreement
    }

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

disagreement_rows = []

# calculate concordance measures
def run_analysis(adata, dataset_id, methods):
    all_disagreements = []
    metadata_markers = adata.obs[["disease", "tissue"]].drop_duplicates()
    if "feature_name" in adata.var.columns:
        adata.var_names = adata.var["feature_name"].astype(str).str.replace(r"\.\d+$", "", regex=True)
    else:
        print("Warning: 'feature_name' not found in adata.var.columns. Gene symbol matching may fail.")

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
        
        for index_a, method_a in enumerate(methods.keys()):
            if method_a not in sample_analysis:
                sample_analysis[method_a] = {}
            
            imputed_a = np.load(
                os.path.join("output", 
                                method_a, 
                                dataset_id, 
                                disease, 
                                tissue + ".npy"
                                )
                )
            # MAGIC runs some pre-processing that changes the shape of the output matrix
            # by removing any genes that aren't expressed by 5 or more cells
            if method_a == "MAGIC":
                imputed_a = expand_magic_matrix(input, imputed_a)
                
            # imputation_rate is just per-imputation, number of zeros imputed / original zeroes
            # UNTESTED
            # if sample_analysis[method_a].get("imputation_rate") == None:
            #     imputed_count = originally_zero & (imputed_a > delta_per_j)
            #     sample_analysis[method_a]["imputation_rate"] = np.sum(imputed_count) / np.sum(originally_zero)

            for index_b, method_b in enumerate(methods.keys()):
                if index_b < index_a:
                    #sample_analysis[method_a][method_b] = sample_analysis[method_b][method_a]
                    continue
                if index_a == index_b:
                    sample_analysis[method_a][method_b] = 1
                    continue
                if method_b not in sample_analysis[method_a]:
                    sample_analysis[method_a][method_b] = {}
                
                imputed_b = np.load(
                    os.path.join("output", 
                                    method_b, 
                                    dataset_id, 
                                    disease, 
                                    tissue + ".npy"
                                    )
                    )
                
                if method_b == "MAGIC":
                    imputed_b = expand_magic_matrix(input, imputed_b)
                
                results = []
                for celltype, genes in marker_genes.items():
                    for gene in genes:
                        row = compute_disagreement_per_gene(
                            adata_sample, dataset_id, disease, tissue,
                            method_a, method_b,
                            input, imputed_a, imputed_b,
                            celltype, gene
                        )
                        if row is not None:
                            results.append(row)
                # print("disagreement: ", results)
                all_disagreements.extend(results)

                ### HERE IS CONCORDANCE CALCULATIONS
                
                # the matrix should be a list of lists
                # first index indicates row, so by comparing to a 1d array, we effectively compare each row
                # to the means array, which then compares each element of the row to each element of the means
                both_imputed_d = originally_zero & (imputed_a > delta_per_j) & (imputed_b > delta_per_j)
                only_a_imputed_d = originally_zero & (imputed_a > delta_per_j) & ~(imputed_b > delta_per_j)
                only_b_imputed_d = originally_zero & ~(imputed_a > delta_per_j) & (imputed_b > delta_per_j)
                neither_imputed_d = originally_zero & ~(imputed_a > delta_per_j) & ~(imputed_b > delta_per_j)
                
                both_imputed_n = originally_zero & (imputed_a > 0) & (imputed_b > 0)
                only_a_imputed_n = originally_zero & (imputed_a > 0) & ~(imputed_b > 0)
                only_b_imputed_n = originally_zero & ~(imputed_a > 0) & (imputed_b > 0)
                neither_imputed_n = originally_zero & ~(imputed_a > 0) & ~(imputed_b > 0)

                sample_analysis[method_a][method_b]["contingency"] = {
                    "both_confident": int(np.sum(both_imputed_d)),
                    "only_method_a_confident": int(np.sum(only_a_imputed_d)),
                    "only_method_b_confident": int(np.sum(only_b_imputed_d)),
                    "neither_confident": int(np.sum(neither_imputed_d)),
                    "both_imputed": int(np.sum(both_imputed_n)),
                    "only_method_a_imputed": int(np.sum(only_a_imputed_n)),
                    "only_method_b_imputed": int(np.sum(only_b_imputed_n)),
                    "neither_imputed": int(np.sum(neither_imputed_n)),
                    "originally_zero": int(np.sum(originally_zero)),
                }
                
                print("- Finding concordance values")
                
                total_concordance = sample_analysis[method_a].setdefault(method_b, {}).get("total_concordance")
                if total_concordance == None:
                    total_concordance = sample_analysis[method_a][method_b]["total_concordance"] = np.sum(both_imputed_d) / np.sum(originally_zero)
                print(f"- Total concordance between {method_a} and {method_b} calculated to be {total_concordance*100}%")
                
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
    df = pd.DataFrame(all_disagreements)
    df.to_csv(f"{dataset_id}_disagreements.csv", index=False)
    print(f"Saved disagreement scores to {dataset_id}_disagreements.csv")

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
