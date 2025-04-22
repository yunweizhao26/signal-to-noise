from pathlib import Path
import os
import importlib
import json
from operator import itemgetter
import numpy as np
import pandas as pd
from pandas import DataFrame
from anndata import read_h5ad
import baselines.SAUCIE.SAUCIE as SAUCIE
import baselines.MAGIC.magic as magic
import scprep
from baselines.deepimpute.deepimpute import multinet
import baselines.scScope.scscope.scscope as scScope
from baselines.scvi.scvi import run_scvi

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

def _run_saucie(y):
    """
    Run the SAUCIE imputation method on input data.
    
    Args:
        y: Input data matrix (cells x genes)
        
    Returns:
        Imputed data matrix
    """
    tf = importlib.import_module('tensorflow.compat.v1')
    tf.disable_v2_behavior()
    
    tf.reset_default_graph()
    saucie_model = SAUCIE.SAUCIE(y.shape[1])
    saucie_model.train(SAUCIE.Loader(y, shuffle=True), steps=1000)
    
    rec_y = saucie_model.get_reconstruction(SAUCIE.Loader(y, shuffle=False))
    
    return rec_y


def _run_magic(y):
    """
    Run the MAGIC imputation method on input data.
    
    Args:
        y: Input data matrix (cells x genes)
        
    Returns:
        Imputed data matrix
    """
    # Preprocess data - filter rare genes and normalize
    y_filtered = scprep.filter.filter_rare_genes(y, min_cells=5)
    y_norm = scprep.transform.sqrt(
        scprep.normalize.library_size_normalize(y_filtered)
    )
    
    # Initialize and run MAGIC
    magic_op = magic.MAGIC(
        # knn=5,
        # knn_max=None,
        # decay=1,
        # Variable changed in paper
        t=7,  # [2,7,'auto']
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
    """
    Expand a reduced matrix from MAGIC back to the original dimensions.
    
    Since MAGIC filters out rarely expressed genes, we need to reinsert those
    columns in the result matrix.
    
    Args:
        y: Original data matrix
        reduced_matrix: MAGIC output matrix (which has fewer columns)
        
    Returns:
        Expanded matrix with same dimensions as y
    """
    expanded = np.zeros(y.shape)
    
    # Identify which columns were kept/removed in the MAGIC processing
    nonzero_counts = (y != 0).sum(axis=0)
    kept_columns = np.where(nonzero_counts >= 5)[0]
    removed_columns = np.where(nonzero_counts < 5)[0]
    
    # Copy values from reduced matrix to their original positions
    for j_reduced, j_original in enumerate(kept_columns):
        expanded[:, j_original] = reduced_matrix[:, j_reduced]
        
    # Keep original values for removed columns
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
    Calculate the disagreement between two imputation methods for a specific gene in a celltype.
    
    For a single (disease, tissue) subset, given the original matrix and
    two imputed matrices (method_a, method_b), calculate the disagreement 
    for one celltype + one gene.
    
    Args:
        adata: AnnData object containing the dataset
        dataset_id: ID of the dataset
        disease: Disease condition
        tissue: Tissue type
        method_a: Name of first imputation method
        method_b: Name of second imputation method
        X_original: Original count matrix
        X_imputed_a: Imputed matrix from method A
        X_imputed_b: Imputed matrix from method B
        celltype: Cell type to analyze
        gene: Gene to analyze
        
    Returns:
        Dictionary of counts & the final disagreement fraction, or None if gene not found
    """
    
    # 1) Subset to the celltype
    mask_ct = (adata.obs["Celltype"] == celltype)
    X_ct_orig = X_original[mask_ct, :]
    X_ct_a = X_imputed_a[mask_ct, :]
    X_ct_b = X_imputed_b[mask_ct, :]

    # 2) Find the gene index, skip if not present
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

    # Count different imputation outcomes
    onlyA = np.sum(expr_a & ~expr_b)
    onlyB = np.sum(~expr_a & expr_b)
    both = np.sum(expr_a & expr_b)
    neither = np.sum(~expr_a & ~expr_b)

    # 5) Disagreement = (onlyA + onlyB) / n_zero
    disagreement = (onlyA + onlyB) / n_zero if n_zero > 0 else 0
    
    # Return a dictionary of all results
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


def _run_scscope(y):
    """
    Run the scScope imputation method on input data.
    
    Args:
        y: Input data matrix (cells x genes)
        
    Returns:
        Imputed data matrix
    """
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


def _run_deepimpute(y):
    """
    Run the DeepImpute method on input data.
    
    Args:
        y: Input data matrix (cells x genes)
        
    Returns:
        Imputed data matrix
    """
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


def _run_scvi(y):
    """
    Run the scVI imputation method on input data.
    
    Args:
        y: Input data matrix (cells x genes)
        
    Returns:
        Imputed data matrix
    """
    return run_scvi(y)


def check_mem_usage():
    """Monitor memory usage of the current process."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Current memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")


def force_gc():
    """Force garbage collection to free up memory."""
    import gc
    check_mem_usage()
    print(f"collecting {gc.collect()} objects")
    check_mem_usage()
    

def run_pipeline_on_sets():
    """
    Main function to run the imputation pipeline on all datasets.
    
    1. Finds all h5ad files in the datasets directory
    2. Defines the imputation methods to run
    3. For each dataset, runs imputation and analysis
    """
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


def run_imputation(adata, dataset_id, methods):
    """
    For each disease+tissue subset in the dataset, apply imputation methods.
    
    Results are saved to "output/imputation/dataset/disease/tissue.npy"
    
    Args:
        adata: AnnData object containing the dataset
        dataset_id: ID of the dataset
        methods: Dictionary of imputation methods {name: function}
    """
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
    """
    Compute the threshold delta value for confident imputation.
    
    Args:
        imputed_values: Array of imputed values
        bins: Number of histogram bins
        
    Returns:
        Threshold value based on the histogram peak
    """
    if len(imputed_values) == 0:
        return 0.0
    hist, bin_edges = np.histogram(imputed_values, bins=bins)
    print(f"Histogram: {hist}")
    print(f"Bin edges: {bin_edges}")
    max_bin = np.argmax(hist)
    return bin_edges[max_bin]


# Global storage for disagreement analysis
disagreement_rows = []


def run_analysis(adata, dataset_id, methods):
    """
    Calculate concordance and disagreement measures between imputation methods.
    
    For each disease+tissue subset and each pair of methods, computes:
    - Overall concordance statistics
    - Per-gene concordance
    - Per-cell concordance
    - Disagreement scores for marker genes
    
    Args:
        adata: AnnData object containing the dataset
        dataset_id: ID of the dataset
        methods: Dictionary of imputation methods {name: function}
    """
    all_disagreements = []
    metadata_markers = adata.obs[["disease", "tissue"]].drop_duplicates()
    
    # Ensure gene names are properly formatted
    if "feature_name" in adata.var.columns:
        adata.var_names = adata.var["feature_name"].astype(str).str.replace(r"\.\d+$", "", regex=True)
    else:
        print("Warning: 'feature_name' not found in adata.var.columns. Gene symbol matching may fail.")

    # Load or initialize analysis data
    analysis_data = {}
    analysis_data_path = os.path.join("output", "analysis", dataset_id + ".json")
    if os.path.exists(analysis_data_path):
        with open(analysis_data_path, "r") as in_f:
            analysis_data = json.load(in_f)
    else:
        os.makedirs(os.path.dirname(analysis_data_path), exist_ok=True)
    
    # Process each disease+tissue sample
    i = 0
    for _, (disease, tissue) in metadata_markers.iterrows():
        i += 1
        print(f"Analyzing sample {i}/{len(metadata_markers)}: {tissue} ({disease})")
        
        # Extract the sample data
        mask = (adata.obs["disease"] == disease) & (adata.obs["tissue"] == tissue)
        adata_sample = adata[mask]
        input = adata_sample.X.toarray()
        print(f"- Sample matrix has shape {input.shape}")
        
        # Initialize sample analysis data
        sample_analysis = analysis_data.setdefault(f"{disease}, {tissue}", {})
        sample_analysis["shape"] = input.shape
        
        # Calculate delta values for confident imputation thresholds
        nonzero_sums = np.sum(input, axis=0)
        nonzero_counts = np.sum(input != 0, axis=0)
        nonzero_counts = np.where(nonzero_counts > 0, nonzero_counts, 1)
        delta_per_j = nonzero_sums / nonzero_counts
        
        # Identify originally zero values
        originally_zero = input == 0
        
        # Compare each pair of methods
        for index_a, method_a in enumerate(methods.keys()):
            if method_a not in sample_analysis:
                sample_analysis[method_a] = {}
            
            # Load imputation results for method A
            imputed_a = np.load(
                os.path.join("output", 
                            method_a, 
                            dataset_id, 
                            disease, 
                            tissue + ".npy"
                            )
            )
            
            # Handle MAGIC output which has special shape requirements
            if method_a == "MAGIC":
                imputed_a = expand_magic_matrix(input, imputed_a)
                
            # Calculate imputation rate if needed
            # UNTESTED
            # if sample_analysis[method_a].get("imputation_rate") == None:
            #     imputed_count = originally_zero & (imputed_a > delta_per_j)
            #     sample_analysis[method_a]["imputation_rate"] = np.sum(imputed_count) / np.sum(originally_zero)

            # Compare with all other methods
            for index_b, method_b in enumerate(methods.keys()):
                # Skip if we've already done this comparison (A vs B = B vs A)
                if index_b < index_a:
                    #sample_analysis[method_a][method_b] = sample_analysis[method_b][method_a]
                    continue
                    
                # A method perfectly agrees with itself
                if index_a == index_b:
                    sample_analysis[method_a][method_b] = 1
                    continue
                    
                if method_b not in sample_analysis[method_a]:
                    sample_analysis[method_a][method_b] = {}
                
                # Load imputation results for method B
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
                
                # Calculate disagreement for marker genes in each cell type
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
                
                all_disagreements.extend(results)

                # Calculate concordance statistics
                
                # For "confident" imputations (above delta threshold)
                both_imputed_d = originally_zero & (imputed_a > delta_per_j) & (imputed_b > delta_per_j)
                only_a_imputed_d = originally_zero & (imputed_a > delta_per_j) & ~(imputed_b > delta_per_j)
                only_b_imputed_d = originally_zero & ~(imputed_a > delta_per_j) & (imputed_b > delta_per_j)
                neither_imputed_d = originally_zero & ~(imputed_a > delta_per_j) & ~(imputed_b > delta_per_j)
                
                # For any non-zero imputations
                both_imputed_n = originally_zero & (imputed_a > 0) & (imputed_b > 0)
                only_a_imputed_n = originally_zero & (imputed_a > 0) & ~(imputed_b > 0)
                only_b_imputed_n = originally_zero & ~(imputed_a > 0) & (imputed_b > 0)
                neither_imputed_n = originally_zero & ~(imputed_a > 0) & ~(imputed_b > 0)

                # Store contingency table statistics
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
                
                # Calculate total concordance if not already done
                total_concordance = sample_analysis[method_a].setdefault(method_b, {}).get("total_concordance")
                if total_concordance == None:
                    total_concordance = sample_analysis[method_a][method_b]["total_concordance"] = np.sum(both_imputed_d) / np.sum(originally_zero)
                print(f"- Total concordance between {method_a} and {method_b} calculated to be {total_concordance*100}%")
                
                # Calculate per-gene concordance if not already done
                if sample_analysis[method_a][method_b].get("per_gene_concordance") == None:
                    print(" - Calculating per-gene concordance values")
                    # Calculate concordance for each gene (column)
                    concordance_per_gene = np.sum(both_imputed_d, axis=0) / np.sum(originally_zero, axis=0)
                    
                    # Label genes with their names
                    labeled = list(zip(adata_sample.var["feature_name"], concordance_per_gene))
                    
                    # Filter out genes with no concordance
                    filtered = [t for t in labeled if t[1] > 0]
                    
                    # Sort genes by concordance (optional)
                    # UNTESTED
                    #sorted_list = sorted(filtered, key=itemgetter(1), reverse=True)
                    
                    sample_analysis[method_a][method_b]["per_gene_concordance"] = filtered
                
                # Calculate per-cell concordance if not already done
                if sample_analysis[method_a][method_b].get("per_cell_concordance") == None:
                    print(" - Calculating per-cell concordance values")
                    concordance_per_cell = np.sum(both_imputed_d, axis=1) / np.sum(originally_zero, axis=1)
                    
                    # Label cells with their type and ID
                    cell_labels = [f"{m} ({n})" for m, n in 
                                  zip(adata_sample.obs["Celltype"], adata_sample.obs.index)]
                    labeled = list(zip(cell_labels, concordance_per_cell))
                    filtered = [t for t in labeled if t[1] > 0]
                    #sorted_list = sorted(filtered, key=itemgetter(1), reverse=True)
                    sample_analysis[method_a][method_b]["per_cell_concordance"] = filtered
        
        # Save analysis results after each sample
        with open(analysis_data_path, "w") as out_f:
            json.dump(analysis_data, out_f, indent=4)
            
        force_gc()
        print("")
    
    # Save disagreement scores as CSV
    df = pd.DataFrame(all_disagreements)
    df.to_csv(f"{dataset_id}_disagreements.csv", index=False)
    print(f"Saved disagreement scores to {dataset_id}_disagreements.csv")


def calculate_avg_concordance(files, methods):
    """
    Calculate average concordance across all samples in the analysis files.
    
    Args:
        files: List of analysis JSON files
        methods: Dictionary of imputation methods
    """
    for f in files:
        with open(f, "r") as inp:
            analysis_data = json.load(inp)
        
        # Collect all concordance values across samples
        combined_values = {}
        for sample in analysis_data.values():
            for method_a in methods.keys():
                combined_values.setdefault(method_a, {})
                for method_b, concordance_val in sample[method_a].items():
                    combined_values[method_a].setdefault(method_b, []).append(concordance_val)

        # Calculate average concordance
        avg = analysis_data["avg"] = {}
        for method_a, sub_concordances in combined_values.items():
            avg[method_a] = {}
            for method_b, values in sub_concordances.items():
                avg[method_a][method_b] = sum(values) / len(values)
                
        # Save updated analysis data
        with open(f, "w") as outp:
            json.dump(analysis_data, outp)


def extract_counts_from_h5ad():
    """
    Extract count matrices from h5ad files and save as numpy arrays.
    """
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
        
        adata = read_h5ad(f)
        
        metadata_marker = "biosample_id"
        if metadata_marker not in adata.obs_keys():
            print(f"Dataset {f.stem} doesn't contain biosample_id metadata, quitting")
            continue
            metadata_marker = ["donor_id", "tissue", "disease"]
        
        # Filter out cells that actually originate from other datasets
        adata = adata[adata.obs["is_primary_data"] == True]
        
        # Create directory for chunked data
        path = f"./datasets/chunked/{f.stem}"
        os.makedirs(path, exist_ok=True)
        
        # Save each sample as a separate file
        for sample in adata.obs[metadata_marker].unique():
            file_path = path + f"/{sample}.npy"
            if os.path.exists(file_path):
                continue
            
            mask = adata.obs[metadata_marker] == sample
            np.save(file_path, adata[mask].X)
        
        extracted.append(f.stem)
        print(f"Finished extracting dataset {f.stem}")
        
    with open(extracted_path, "w") as f:
        json.dump(extracted, f)


if __name__ == "__main__":
    run_pipeline_on_sets()