import os
import gc
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from anndata import read_h5ad

try:
    import baselines.SAUCIE.SAUCIE as SAUCIE
    import baselines.MAGIC.magic as magic
    import scprep
    from baselines.deepimpute.deepimpute import multinet
    import baselines.scScope.scscope.scscope as scScope
    from baselines.scvi.scvi import run_scvi
    IMPORTED_METHODS = True
except ImportError:
    print("Warning: Not all imputation methods could be imported.")
    IMPORTED_METHODS = False

# Create directories
def setup_directories(args):
    """Create all necessary directories for the experiment."""
    for d in [args.output_dir, args.figures_dir, args.cache_dir]:
        os.makedirs(d, exist_ok=True)
        
    # Create subdirectories for each method
    for method in args.methods.split(','):
        os.makedirs(os.path.join(args.cache_dir, method), exist_ok=True)
        
    print(f"Created directory structure")

# Utility functions
def log_message(message, log_file=None):
    """Log a message to console and optionally to a file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    
    if log_file:
        with open(log_file, "a") as f:
            f.write(full_message + "\n")

def force_gc():
    """Force garbage collection to free up memory."""
    import psutil
    process = psutil.Process(os.getpid())
    before = process.memory_info().rss / (1024 * 1024)
    
    gc.collect()
    
    after = process.memory_info().rss / (1024 * 1024)
    print(f"Memory usage: {before:.2f} MB → {after:.2f} MB (freed {before-after:.2f} MB)")

# Load dataset and markers
def load_data(args):
    """Load the scRNA-seq dataset and marker genes."""
    log_message(f"Loading dataset from {args.input_file}")
    
    # Load AnnData object
    adata = read_h5ad(args.input_file)
    
    # Filter primary data if needed
    if "is_primary_data" in adata.obs.columns:
        adata = adata[adata.obs["is_primary_data"] == True]
    
    # Load marker genes
    if args.markers_file:
        with open(args.markers_file, 'r') as f:
            marker_genes = json.load(f)
    else:
        # Default marker genes (if not provided)
        marker_genes = {
            "Enterocytes": ["BEST4", "OTOP2", "CA7"],
            "Goblet cells": ["MUC2", "TFF1", "TFF3"],
            "Tuft cells": ["POU2F3", "DCLK1"],
            "Stem cells": ["OLFM4", "LGR5", "ASCL2"],
            "Paneth cells": ["LYZ", "DEFA5"]
        }
    
    log_message(f"Loaded dataset with {adata.shape[0]} cells and {adata.shape[1]} genes")
    log_message(f"Loaded {len(marker_genes)} cell types with marker genes")
    
    return adata, marker_genes

# Run imputation methods
def run_imputation_methods(adata, args):
    """Run all imputation methods and cache the results."""
    methods = {}
    all_methods = args.methods.split(',')
    
    # Define all available methods
    if IMPORTED_METHODS:
        all_available_methods = {
            "SAUCIE": run_saucie,
            # "MAGIC": run_magic,
            "scScope": run_scscope,
            "DeepImpute": run_deepimpute,
            "scVI": run_scvi
        }
    else:
        def dummy_method(X):
            print(f"Running dummy method on {X.shape}")
            return X
            
        all_available_methods = {
            "SAUCIE": dummy_method,
            "MAGIC": dummy_method,
            "scScope": dummy_method,
            "DeepImpute": dummy_method,
            "scVI": dummy_method
        }
    
    # Filter to only requested methods
    for method_name in all_methods:
        if method_name in all_available_methods:
            methods[method_name] = all_available_methods[method_name]
        else:
            log_message(f"Warning: Method {method_name} not found, skipping")
    
    # Process all disease/tissue combinations
    if "disease" in adata.obs.columns and "tissue" in adata.obs.columns:
        metadata_markers = adata.obs[["disease", "tissue"]].drop_duplicates()
        
        for idx, (disease, tissue) in enumerate(metadata_markers.values):
            log_message(f"Processing sample {idx+1}/{len(metadata_markers)}: {tissue} ({disease})")
            
            # Extract the subsample
            mask = (adata.obs["disease"] == disease) & (adata.obs["tissue"] == tissue)
            adata_sample = adata[mask]
            X = adata_sample.X.toarray() if hasattr(adata_sample.X, 'toarray') else adata_sample.X
            
            # Run each method
            for method_name, method_func in methods.items():
                # Define cache path
                cache_file = os.path.join(args.cache_dir, method_name, f"{disease}_{tissue}.npy")
                
                # Check if already cached
                if os.path.exists(cache_file) and not args.force_recompute:
                    log_message(f"  Using cached {method_name} results from {cache_file}")
                    continue
                
                # Run the method
                log_message(f"  Running {method_name} on {X.shape}")
                imputed = method_func(X)
                
                # Handle MAGIC special case
                if method_name == "MAGIC":
                    imputed = expand_magic_matrix(X, imputed)
                
                # Save results
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                np.save(cache_file, imputed)
                log_message(f"  Saved {method_name} results to {cache_file}")
                
                # Free memory
                force_gc()
    else:
        # Process the entire dataset at once
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        
        for method_name, method_func in methods.items():
            # Define cache path
            cache_file = os.path.join(args.cache_dir, method_name, "full_dataset.npy")
            
            # Check if already cached
            if os.path.exists(cache_file) and not args.force_recompute:
                log_message(f"  Using cached {method_name} results from {cache_file}")
                continue
            
            # Run the method
            log_message(f"  Running {method_name} on {X.shape}")
            imputed = method_func(X)
            
            # Handle MAGIC special case
            if method_name == "MAGIC":
                imputed = expand_magic_matrix(X, imputed)
            
            # Save results
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            np.save(cache_file, imputed)
            log_message(f"  Saved {method_name} results to {cache_file}")
            
            # Free memory
            force_gc()
    
    return methods.keys()

# Calculate the TH-score for each zero entry
def calculate_th_scores(adata, method_names, args):
    """Calculate TruthfulHypothesis scores for all zero entries."""
    log_message("Calculating TruthfulHypothesis scores for all zero entries")
    
    # Process all disease/tissue combinations
    if "disease" in adata.obs.columns and "tissue" in adata.obs.columns:
        metadata_markers = adata.obs[["disease", "tissue"]].drop_duplicates()
        
        results = []
        for idx, (disease, tissue) in enumerate([metadata_markers.values[0]]):
            log_message(f"Processing sample {idx+1}/{len(metadata_markers)}: {tissue} ({disease})")
            
            # Extract the subsample
            mask = (adata.obs["disease"] == disease) & (adata.obs["tissue"] == tissue)
            adata_sample = adata[mask]
            X = adata_sample.X.toarray() if hasattr(adata_sample.X, 'toarray') else adata_sample.X
            
            # Find zero entries
            zero_mask = X == 0
            zero_indices = np.where(zero_mask)
            
            if len(zero_indices[0]) == 0:
                log_message(f"  No zero entries found in this sample, skipping")
                continue
                
            log_message(f"  Found {len(zero_indices[0])} zero entries")
            
            # Load method weights (reliability scores)
            if args.weights_file and os.path.exists(args.weights_file):
                with open(args.weights_file, 'r') as f:
                    weights = json.load(f)
            else:
                # Default: equal weights for all methods
                weights = {method: 1.0 for method in method_names}
            
            # Normalize weights
            total_weight = sum(weights.values())
            weights = {m: w/total_weight for m, w in weights.items()}
            
            # Calculate TH scores and consensus values
            log_message(f"  Calculating consensus scores using {len(method_names)} methods with weights: {weights}")
            
            scores = np.zeros(len(zero_indices[0]))
            values = np.zeros(len(zero_indices[0]))
            
            for method_name in method_names:
                # Load imputed matrix
                dataset_id = os.path.basename(args.input_file).split('.')[0]
                if dataset_id.endswith("_raw") or dataset_id.endswith("_pre"):
                    dataset_id = dataset_id[:-4]

                imputed_file = get_imputation_files(dataset_id, method_name, disease, tissue)
                if imputed_file is None:
                    log_message(f"Skipping method {method_name} for {disease}, {tissue} - file not found")
                    continue
                imputed = np.load(imputed_file)
                print("method: ", method_name)
                print("imputed size: ", imputed.shape)
                
                # Get method weight
                weight = weights.get(method_name, 0.0)
                
                # # Extract values at zero positions
                imputed_values = imputed[zero_indices]
                
                # Update scores (1 if imputed > 0, else 0)
                scores += weight * (imputed_values > 0).astype(float)
                
                # Update consensus values
                values += weight * imputed_values
            
            # Store results
            results.append({
                "disease": disease,
                "tissue": tissue,
                "zero_indices": zero_indices,
                "scores": scores,
                "values": values,
                "original_matrix": X
            })
            
            log_message(f"  Completed TH score calculation for this sample")
            
        return results
    else:
        # Process the entire dataset at once
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        
        # Find zero entries
        zero_mask = X == 0
        zero_indices = np.where(zero_mask)
        
        log_message(f"Found {len(zero_indices[0])} zero entries in the dataset")
        
        # Load method weights (reliability scores)
        if args.weights_file and os.path.exists(args.weights_file):
            with open(args.weights_file, 'r') as f:
                weights = json.load(f)
        else:
            # Default: equal weights for all methods
            weights = {method: 1.0 for method in method_names}
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {m: w/total_weight for m, w in weights.items()}
        
        # Calculate TH scores and consensus values
        log_message(f"Calculating consensus scores using {len(method_names)} methods with weights: {weights}")
        
        scores = np.zeros(len(zero_indices[0]))
        values = np.zeros(len(zero_indices[0]))
        
        for method_name in method_names:
            # Load imputed matrix
            imputed_file = os.path.join(args.cache_dir, method_name, "full_dataset.npy")
            imputed = np.load(imputed_file)
            
            # Get method weight
            weight = weights.get(method_name, 0.0)
            
            # Extract values at zero positions
            imputed_values = imputed[zero_indices]
            
            # Update scores (1 if imputed > 0, else 0)
            scores += weight * (imputed_values > 0).astype(float)
            
            # Update consensus values
            values += weight * imputed_values
        
        # Store results
        results = [{
            "disease": "all",
            "tissue": "all",
            "zero_indices": zero_indices,
            "scores": scores,
            "values": values,
            "original_matrix": X
        }]
        
        log_message(f"Completed TH score calculation")
        
        return results

def calculate_marker_based_weights(original_matrix, imputed_matrices, marker_genes, adata):
    """Calculate weights for each imputation method based on marker gene agreement improvement."""
    weights = []
    
    # Get cell type annotations
    if "Celltype" in adata.obs.columns:
        cell_types = adata.obs["Celltype"].values
    else:
        # If no cell type annotation, create a dummy one
        from sklearn.cluster import KMeans
        n_clusters = min(10, original_matrix.shape[0] // 10)  # Reasonable number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cell_types = kmeans.fit_predict(original_matrix)
        cell_types = np.array([f"Cluster_{i}" for i in cell_types])
    
    # Calculate baseline marker agreement score
    baseline_score = calculate_marker_agreement(original_matrix, marker_genes, adata)
    
    # Calculate improvement for each method
    for imputed in imputed_matrices:
        score = calculate_marker_agreement(imputed, marker_genes, adata)
        improvement = max(0, score - baseline_score)  # Only consider positive improvements
        weights.append(improvement + 0.01)  # Add small constant to avoid zero weights
    
    # Normalize weights
    weights = np.array(weights)
    return weights / np.sum(weights)


# Calculate marker agreement score
def calculate_marker_agreement(X, marker_genes, adata):
    """Calculate the marker gene agreement score across all cells."""
    # Create gene index lookup
    gene2idx = {g: i for i, g in enumerate(adata.var_names)}
    
    # Cell type to index
    if "Celltype" in adata.obs.columns:
        cell_types = adata.obs["Celltype"].values
        unique_cell_types = np.unique(cell_types)
    else:
        # If no cell type annotation, use clustering or just assign unknown
        cell_types = np.array(["unknown"] * X.shape[0])
        unique_cell_types = np.array(["unknown"])
    
    # Initialize score components
    specificity_score = 0.0
    expression_score = 0.0
    
    # Track number of valid marker cell-type pairs for normalization
    valid_pairs = 0
    
    # For each cell type with markers
    for cell_type, markers in marker_genes.items():
        # If this cell type isn't in annotations, skip
        if cell_type not in unique_cell_types:
            continue
            
        # Get indices of cells of this type
        type_mask = cell_types == cell_type
        if sum(type_mask) == 0:
            continue
        
        # Get indices of the marker genes present in our dataset
        marker_indices = [gene2idx[g] for g in markers if g in gene2idx]
        if len(marker_indices) == 0:
            continue
            
        # For each marker gene
        for marker_idx in marker_indices:
            # Expression in correct cell type
            expr_in_type = np.mean(X[type_mask, marker_idx])
            
            # Expression in other cell types
            expr_in_others = np.mean(X[~type_mask, marker_idx]) if sum(~type_mask) > 0 else 0
            
            # 1. Specificity: How much more expressed in correct type
            if expr_in_others > 0:
                ratio = expr_in_type / expr_in_others
                specificity = np.tanh(ratio - 1)  # Bounded growth function
            else:
                specificity = 1.0 if expr_in_type > 0 else 0.0
            
            # 2. Expression level: What fraction of cells of this type express the marker
            expression = np.mean(X[type_mask, marker_idx] > 0)
            
            # Combine scores
            specificity_score += specificity
            expression_score += expression
            valid_pairs += 1
    
    # Return combined score, normalized by number of marker-celltype pairs
    if valid_pairs > 0:
        return (0.7 * specificity_score + 0.3 * expression_score) / valid_pairs
    else:
        return 0.0

def plot_incremental_results(all_results, args):
    """Generate plots showing how marker agreement changes with incremental imputation."""
    for result in all_results:
        disease = result["disease"].strip().replace(" ", "_")
        tissue = result["tissue"].strip().replace(" ", "_")
        curve_data = result["imputation_curve"]
        max_score_idx = result["max_score_idx"]
        
        # Convert to DataFrame
        df = pd.DataFrame(curve_data)
        
        # Plot marker agreement score vs. percentage of zeros imputed
        plt.figure(figsize=(10, 6))
        plt.plot(df["imputed_fraction"] * 100, df["score"], marker='o', markersize=4)
        plt.axvline(x=df.iloc[max_score_idx]["imputed_fraction"] * 100, color='r', linestyle='--')
        
        plt.title(f"Marker Score vs. Percentage of Zeros Imputed - {disease}, {tissue}")
        plt.xlabel("Percentage of Zeros Imputed (%)")
        plt.ylabel("Marker Agreement Score")
        plt.grid(True)
        
        # Add annotation for the optimal point
        max_point = df.iloc[max_score_idx]
        plt.annotate(f"Optimal: {max_point['imputed_fraction']*100:.1f}%\nScore: {max_point['score']:.4f}",
                     xy=(max_point["imputed_fraction"]*100, max_point["score"]),
                     xytext=(max_point["imputed_fraction"]*100 + 5, max_point["score"]),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
        
        plt.tight_layout()
        
        # Save the figure
        figure_file = os.path.join(args.figures_dir, f"incremental_curve_{disease}_{tissue}.png")
        plt.savefig(figure_file, dpi=300)
        plt.close()
        
        # Also plot score improvement relative to baseline
        baseline_score = df.iloc[0]["score"]
        df["improvement"] = df["score"] - baseline_score
        
        plt.figure(figsize=(10, 6))
        plt.plot(df["imputed_fraction"] * 100, df["improvement"], marker='o', markersize=4)
        plt.axvline(x=df.iloc[max_score_idx]["imputed_fraction"] * 100, color='r', linestyle='--')
        
        plt.title(f"Score Improvement vs. Percentage of Zeros Imputed - {disease}, {tissue}")
        plt.xlabel("Percentage of Zeros Imputed (%)")
        plt.ylabel("Score Improvement")
        plt.grid(True)
        
        # Add annotation for the optimal point
        plt.annotate(f"Optimal: {max_point['imputed_fraction']*100:.1f}%\nImprovement: {max_point['improvement']:.4f}",
                     xy=(max_point["imputed_fraction"]*100, max_point["improvement"]),
                     xytext=(max_point["imputed_fraction"]*100 + 5, max_point["improvement"]),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
        
        plt.tight_layout()
        
        # Save the improvement figure
        figure_file = os.path.join(args.figures_dir, f"improvement_curve_{disease}_{tissue}.png")
        plt.savefig(figure_file, dpi=300)
        plt.close()
    
    # Create summary plot if multiple samples
    if len(all_results) > 1:
        plt.figure(figsize=(12, 8))
        
        for result in all_results:
            disease = result["disease"]
            tissue = result["tissue"]
            df = pd.DataFrame(result["imputation_curve"])
            plt.plot(df["imputed_fraction"]*100, df["score"], 
                     label=f"{tissue} ({disease})", marker='o', markersize=3)
            
            # Mark optimal point
            max_idx = result["max_score_idx"]
            plt.scatter(df.iloc[max_idx]["imputed_fraction"]*100, 
                        df.iloc[max_idx]["score"], 
                        marker='X', s=100, c='red')
        
        plt.title("Marker Agreement Score vs. Percentage of Zeros Imputed")
        plt.xlabel("Percentage of Zeros Imputed (%)")
        plt.ylabel("Marker Agreement Score")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        # Save summary figure
        figure_file = os.path.join(args.figures_dir, f"summary_incremental_curves.png")
        plt.savefig(figure_file, dpi=300)
        plt.close()
        
        # Create a histogram of optimal imputation percentages
        optimal_percentages = [result["max_score_fraction"]*100 for result in all_results]
        
        plt.figure(figsize=(10, 6))
        plt.hist(optimal_percentages, bins=10, edgecolor='black')
        plt.axvline(x=np.mean(optimal_percentages), color='r', linestyle='--',
                   label=f"Mean: {np.mean(optimal_percentages):.1f}%")
        
        plt.title("Distribution of Optimal Imputation Percentages")
        plt.xlabel("Optimal Percentage of Zeros Imputed (%)")
        plt.ylabel("Count")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        # Save histogram
        figure_file = os.path.join(args.figures_dir, f"optimal_percentage_histogram.png")
        plt.savefig(figure_file, dpi=300)
        plt.close()

# Incremental imputation loop to find minimal imputation
def incremental_imputation(adata, marker_genes, th_results, args):
    """
    Incrementally add imputed values based on TH scores and track marker agreement.
    Only modifies zero entries, preserving original non-zero values.
    """
    all_results = []
    
    for sample_idx, sample_data in enumerate(th_results):
        # Unpack sample data
        disease = sample_data["disease"]
        tissue = sample_data["tissue"]
        zero_indices = sample_data["zero_indices"]
        scores = sample_data["scores"]
        values = sample_data["values"]
        X_original = sample_data["original_matrix"]
        
        # Create test matrix (copy of original)
        X_test = X_original.copy()
        
        # Sort indices by TH scores (descending)
        sorted_indices = np.argsort(-scores)
        
        # Determine total zeros and step size for 1% increments
        total_zeros = len(sorted_indices)
        step_size = max(1, int(0.01 * total_zeros))  # 1% of zeros
        
        # Initialize tracking
        imputation_curve = []
        
        # Calculate initial marker agreement score
        initial_score = calculate_marker_agreement(X_test, marker_genes, adata)
        imputation_curve.append({
            "iteration": 0,
            "imputed_entries": 0,
            "imputed_fraction": 0.0,
            "score": initial_score
        })
        
        # Main imputation loop - add 1% at a time
        for i in range(0, total_zeros, step_size):
            # Get next batch of indices (1% increment)
            batch_indices = sorted_indices[i:i+step_size]
            
            if len(batch_indices) == 0:
                break
                
            # Get coordinates to update
            rows = zero_indices[0][batch_indices]
            cols = zero_indices[1][batch_indices]
            new_values = values[batch_indices]
            
            # Update test matrix (only zeros)
            for r, c, v in zip(rows, cols, new_values):
                X_test[r, c] = v
            
            # Calculate new marker agreement score
            current_score = calculate_marker_agreement(X_test, marker_genes, adata)
            imputed_frac = (i + len(batch_indices)) / total_zeros
            
            # Store result
            imputation_curve.append({
                "iteration": (i // step_size) + 1,
                "imputed_entries": i + len(batch_indices),
                "imputed_fraction": imputed_frac,
                "score": current_score
            })
        
        # Find point of maximum score
        scores_array = np.array([point["score"] for point in imputation_curve])
        max_score_idx = np.argmax(scores_array)
        max_score = scores_array[max_score_idx]
        max_score_frac = imputation_curve[max_score_idx]["imputed_fraction"]
        
        # Create optimal matrix based on best score
        optimal_entries = imputation_curve[max_score_idx]["imputed_entries"]
        optimal_indices = sorted_indices[:optimal_entries]
        
        X_optimal = X_original.copy()
        if max_score_idx > 0:  # If we found an improvement
            opt_rows = zero_indices[0][optimal_indices]
            opt_cols = zero_indices[1][optimal_indices]
            opt_values = values[optimal_indices]
            
            for r, c, v in zip(opt_rows, opt_cols, opt_values):
                X_optimal[r, c] = v
        
        # Save results
        sample_result = {
            "disease": disease,
            "tissue": tissue,
            "imputation_curve": imputation_curve,
            "max_score_idx": max_score_idx,
            "max_score": max_score,
            "max_score_fraction": max_score_frac,
            "X_optimal": X_optimal
        }
        
        all_results.append(sample_result)
    
    return all_results

# Generate figures visualizing the results
def generate_figures(all_results, args):
    """Generate figures for the incremental imputation results."""
    log_message("Generating figures")
    
    for result in all_results:
        disease = result["disease"].strip().replace(" ", "_")
        tissue = result["tissue"].strip().replace(" ", "_")
        curve_data = result["imputation_curve"]
        max_score_idx = result["max_score_idx"]
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(curve_data)
        
        # 1. Score vs. Fraction curve
        plt.figure(figsize=(10, 6))
        plt.plot(df["imputed_fraction"], df["score"])
        plt.axvline(x=df.iloc[max_score_idx]["imputed_fraction"], color='r', linestyle='--')
        
        plt.title(f"Score vs. Imputation Fraction - {disease}, {tissue}")
        plt.xlabel("Fraction of Zeros Imputed")
        plt.ylabel("Marker Agreement Score")
        
        # Add annotation for the elbow point
        max_point = df.iloc[max_score_idx]
        plt.annotate(f"Optimal: {max_point['imputed_fraction']:.1%}\nScore: {max_point['score']:.4f}",
                     xy=(max_point["imputed_fraction"], max_point["score"]),
                     xytext=(max_point["imputed_fraction"] + 0.05, max_point["score"]),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
        
        plt.grid(True)
        plt.tight_layout()
        # if folder doesn't exist
        os.makedirs(args.figures_dir, exist_ok=True)
        figure_file = os.path.join(args.figures_dir, f"curve_{disease}_{tissue}.png")
        plt.savefig(figure_file, dpi=300)
        plt.close()
        log_message(f"  Saved curve figure to {figure_file}")
        
        if len(df) > 1:
            initial_score = df.iloc[0]["score"]
            df["improvement"] = df["score"] - initial_score
            
            plt.figure(figsize=(10, 6))
            plt.plot(df["imputed_fraction"], df["improvement"])
            plt.axvline(x=df.iloc[max_score_idx]["imputed_fraction"], color='r', linestyle='--')
            
            plt.title(f"Score Improvement vs. Imputation Fraction - {disease}, {tissue}")
            plt.xlabel("Fraction of Zeros Imputed")
            plt.ylabel("Score Improvement")
            
            plt.grid(True)
            plt.tight_layout()
            
            # Save figure
            figure_file = os.path.join(args.figures_dir, f"improvement_{disease}_{tissue}.png")
            plt.savefig(figure_file, dpi=300)
            plt.close()
            
            log_message(f"  Saved improvement figure to {figure_file}")
    
    # Generate summary figure if multiple samples
    if len(all_results) > 1:
        plt.figure(figsize=(12, 8))
        
        for result in all_results:
            disease = result["disease"]
            tissue = result["tissue"]
            curve_data = result["imputation_curve"]
            df = pd.DataFrame(curve_data)
            
            # Normalize scores to show relative improvement
            initial_score = df.iloc[0]["score"]
            df["normalized_score"] = df["score"] / initial_score
            
            plt.plot(df["imputed_fraction"], df["normalized_score"], label=f"{tissue} ({disease})")
        
        plt.title("Normalized Score vs. Imputation Fraction Across Samples")
        plt.xlabel("Fraction of Zeros Imputed")
        plt.ylabel("Normalized Score (score / initial_score)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save figure
        figure_file = os.path.join(args.figures_dir, "summary_curves.png")
        plt.savefig(figure_file, dpi=300)
        plt.close()
        
        log_message(f"  Saved summary figure to {figure_file}")
        
    # Generate histogram of optimal fractions
    optimal_fractions = [result["max_score_fraction"] for result in all_results]
    
    plt.figure(figsize=(10, 6))
    plt.hist(optimal_fractions, bins=10, edgecolor='black')
    plt.axvline(x=np.mean(optimal_fractions), color='r', linestyle='--', 
                label=f"Mean: {np.mean(optimal_fractions):.1%}")
    
    plt.title("Distribution of Optimal Imputation Fractions")
    plt.xlabel("Optimal Fraction of Zeros Imputed")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save figure
    figure_file = os.path.join(args.figures_dir, "optimal_fractions_histogram.png")
    plt.savefig(figure_file, dpi=300)
    plt.close()
    
    log_message(f"  Saved distribution figure to {figure_file}")
    
    # Generate combined curve with all samples
    combined_data = []
    for result in all_results:
        disease = result["disease"]
        tissue = result["tissue"]
        curve_data = result["imputation_curve"]
        
        for point in curve_data:
            point_with_sample = point.copy()
            point_with_sample["disease"] = disease
            point_with_sample["tissue"] = tissue
            combined_data.append(point_with_sample)
    
    combined_df = pd.DataFrame(combined_data)
    
    plt.figure(figsize=(12, 8))
    sns.lineplot(x="imputed_fraction", y="score", hue="tissue", style="disease", data=combined_df)
    
    plt.title("Score vs. Imputation Fraction Across All Samples")
    plt.xlabel("Fraction of Zeros Imputed")
    plt.ylabel("Marker Agreement Score")
    plt.grid(True)
    plt.tight_layout()
    
    # Save figure
    figure_file = os.path.join(args.figures_dir, "all_samples_curves.png")
    plt.savefig(figure_file, dpi=300)
    plt.close()
    
    log_message(f"  Saved combined curve figure to {figure_file}")

def generate_report(all_results, args):
    """Generate CSV summary of the incremental imputation experiments."""
    csv_path = os.path.join(args.output_dir, "minimal_imputation_results.csv")
    
    with open(csv_path, 'w') as f:
        # Write header
        f.write("Disease,Tissue,Total_Zeros,Optimal_Percentage,Initial_Score,Optimal_Score,Improvement,Improvement_Percentage\n")
        
        # Write data for each sample
        for result in all_results:
            disease = result["disease"]
            tissue = result["tissue"]
            curve_data = result["imputation_curve"]
            max_score_idx = result["max_score_idx"]
            
            # Calculate metrics
            total_zeros = curve_data[-1]["imputed_entries"] if curve_data[-1]["imputed_fraction"] == 1.0 else "unknown"
            optimal_percentage = curve_data[max_score_idx]["imputed_fraction"] * 100
            initial_score = curve_data[0]["score"]
            optimal_score = result["max_score"]
            improvement = optimal_score - initial_score
            improvement_percentage = (improvement / initial_score) * 100 if initial_score > 0 else "N/A"
            
            # Write row
            f.write(f"{disease},{tissue},{total_zeros},{optimal_percentage:.2f},{initial_score:.4f},{optimal_score:.4f},{improvement:.4f},{improvement_percentage}\n")
    
    log_message(f"CSV report generated at {csv_path}")
    
    txt_path = os.path.join(args.output_dir, "minimal_imputation_summary.txt")
    
    with open(txt_path, 'w') as f:
        f.write("Minimal Imputation Experiment Summary\n")
        f.write("===================================\n\n")
        
        # Overall statistics
        avg_percentage = np.mean([result["max_score_fraction"] * 100 for result in all_results])
        avg_improvement = np.mean([result["max_score"] - result["imputation_curve"][0]["score"] for result in all_results])
        
        f.write(f"Number of samples analyzed: {len(all_results)}\n")
        f.write(f"Average optimal imputation percentage: {avg_percentage:.2f}%\n")
        f.write(f"Average score improvement: {avg_improvement:.4f}\n\n")
        
        f.write("Sample-specific results:\n")
        f.write("=======================\n\n")
        
        for result in all_results:
            disease = result["disease"]
            tissue = result["tissue"]
            curve_data = result["imputation_curve"]
            max_score_idx = result["max_score_idx"]
            
            # Calculate metrics
            optimal_percentage = curve_data[max_score_idx]["imputed_fraction"] * 100
            initial_score = curve_data[0]["score"]
            optimal_score = result["max_score"]
            improvement = optimal_score - initial_score
            
            f.write(f"Sample: {tissue} ({disease})\n")
            f.write(f"  Optimal imputation percentage: {optimal_percentage:.2f}%\n")
            f.write(f"  Initial marker score: {initial_score:.4f}\n")
            f.write(f"  Optimal marker score: {optimal_score:.4f}\n")
            f.write(f"  Improvement: {improvement:.4f}\n")
            if initial_score > 0:
                f.write(f"  Relative improvement: {(improvement/initial_score)*100:.2f}%\n")
            f.write("\n")
            
            # Add method weights if available
            if "method_weights" in result and result["method_weights"]:
                f.write("  Method weights:\n")
                for method, weight in result["method_weights"].items():
                    f.write(f"    {method}: {weight:.4f}\n")
                f.write("\n")
    
    log_message(f"Text summary generated at {txt_path}")

# Supporting functions for MAGIC
def run_magic(y):
    """Run the MAGIC imputation method on input data."""
    # Preprocess data - filter rare genes and normalize
    y_filtered = scprep.filter.filter_rare_genes(y, min_cells=5)
    y_norm = scprep.transform.sqrt(
        scprep.normalize.library_size_normalize(y_filtered)
    )
    
    # Initialize and run MAGIC
    magic_op = magic.MAGIC(t=7, n_pca=20, n_jobs=-1, random_state=42)
    y_hat = magic_op.fit_transform(y_norm, genes='all_genes')
    
    return y_hat

def expand_magic_matrix(y, reduced_matrix):
    """Expand a reduced matrix from MAGIC back to original dimensions."""
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

def get_imputation_files(dataset_id, method, disease, tissue):
    """Get the path to the imputation file based on existing structure."""
    base_path = os.path.join("output", method, dataset_id, disease)
    file_path = os.path.join(base_path, f"{tissue}.npy")
    
    # Check if file exists
    if os.path.exists(file_path):
        return file_path
    
    # Try alternate filename format with spaces
    alt_path = os.path.join(base_path, f"{tissue} epithelium.npy")
    if os.path.exists(alt_path):
        return alt_path
        
    # Check other possible variations
    variations = [
        f"{tissue} epithelium.npy",
        f"lamina propria of mucosa of {tissue}.npy",
        f"left {tissue}.npy",
        f"right {tissue}.npy",
        f"sigmoid {tissue}.npy"
    ]
    
    for var in variations:
        var_path = os.path.join(base_path, var)
        if os.path.exists(var_path):
            return var_path
    
    return None

# Main function
def main():
    """Main execution function for the minimal imputation experiment."""
    parser = argparse.ArgumentParser(description="Run TruthfulHypothesis (TH) Score-based Minimal Imputation Experiment")
    
    # Required arguments
    parser.add_argument("--input_file", type=str, required=True, 
                        help="Path to input h5ad file")
    
    # Optional arguments
    parser.add_argument("--output_dir", type=str, default="./output", 
                        help="Directory to save output files")
    parser.add_argument("--figures_dir", type=str, default="./figures", 
                        help="Directory to save figures")
    parser.add_argument("--cache_dir", type=str, default="./cache", 
                        help="Directory to cache imputation results")
    parser.add_argument("--markers_file", type=str, default=None, 
                        help="Path to JSON file with marker genes")
    parser.add_argument("--weights_file", type=str, default=None, 
                        help="Path to JSON file with method reliability weights")
    parser.add_argument("--methods", type=str, default="SAUCIE,MAGIC", 
                        help="Comma-separated list of imputation methods to use")
    parser.add_argument("--batch_percent", type=float, default=0.5, 
                        help="Percentage of zeros to impute in each iteration")
    parser.add_argument("--early_stop", type=int, default=3, 
                        help="Number of iterations without improvement before early stopping (0 to disable)")
    parser.add_argument("--force_recompute", action="store_true", 
                        help="Force recomputation of imputation results even if cached")
    
    args = parser.parse_args()
    
    # Create log file
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "experiment_log.txt")
    with open(log_file, "w") as f:
        f.write(f"TruthfulHypothesis (TH) Score-based Minimal Imputation Experiment\n")
        f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Arguments: {vars(args)}\n\n")
    
    # Set up directories
    setup_directories(args)
    
    # Record start time
    start_time = datetime.now()
    log_message(f"Starting minimal imputation experiment at {start_time}", log_file)
    
    try:
        # Step 1: Load dataset and marker genes
        adata, marker_genes = load_data(args)
        
        # Step 2: Run imputation methods and cache results
        # method_names = run_imputation_methods(adata, args)
        method_names = args.methods.split(',')

        # Step 3: Calculate TH scores with marker-based weights
        th_results = calculate_th_scores(adata, method_names, marker_genes, args)
        
        # Step 4: Run incremental imputation to find minimal imputation
        all_results = incremental_imputation(adata, marker_genes, th_results, args)
        
        # Step 5: Generate figures with 1% increments
        plot_incremental_results(all_results, args)
        
        # Step 6: Generate report
        generate_report(all_results, args)
        
        # Record end time and duration
        end_time = datetime.now()
        duration = end_time - start_time
        log_message(f"Experiment completed at {end_time}", log_file)
        log_message(f"Total duration: {duration}", log_file)
        
        # Print final summary
        log_message("\nExperiment Summary:", log_file)
        log_message(f"Number of samples analyzed: {len(all_results)}", log_file)
        
        avg_optimal_fraction = np.mean([r["max_score_fraction"] for r in all_results])
        log_message(f"Average optimal imputation fraction: {avg_optimal_fraction:.1%}", log_file)
        
        avg_improvement = np.mean([r["max_score"] - r["imputation_curve"][0]["score"] for r in all_results])
        log_message(f"Average improvement in marker agreement: {avg_improvement:.4f}", log_file)
        
        log_message("\nResults saved to:", log_file)
        log_message(f"  Output data: {args.output_dir}", log_file)
        log_message(f"  Figures: {args.figures_dir}", log_file)
        log_message(f"  Report: {os.path.join(args.output_dir, 'minimal_imputation_report.html')}", log_file)
        
    except Exception as e:
        import traceback
        error_message = f"Error occurred: {str(e)}\n{traceback.format_exc()}"
        log_message(error_message, log_file)
        print(error_message)
        sys.exit(1)

def run_saucie(y):
    """Run the SAUCIE imputation method on input data."""
    if not IMPORTED_METHODS:
        print("SAUCIE not available, using dummy implementation")
        return y
        
    # Import TensorFlow in a safe way
    tf = importlib.import_module('tensorflow.compat.v1')
    tf.disable_v2_behavior()
    
    # Reset graph to prevent memory issues with repeated calls
    tf.reset_default_graph()
    
    # Create and train the SAUCIE model
    saucie_model = SAUCIE.SAUCIE(y.shape[1])
    saucie_model.train(SAUCIE.Loader(y, shuffle=True), steps=1000)
    
    # Get the reconstructed data
    rec_y = saucie_model.get_reconstruction(SAUCIE.Loader(y, shuffle=False))
    
    return rec_y

def run_scscope(y):
    """Run the scScope imputation method on input data."""
    if not IMPORTED_METHODS:
        print("scScope not available, using dummy implementation")
        return y
        
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
        num_gpus=1
    )
    
    _, rec_y, _ = scScope.predict(y, model, batch_effect=[])
    return rec_y

def run_deepimpute(y):
    """Run the DeepImpute method on input data."""
    if not IMPORTED_METHODS:
        print("DeepImpute not available, using dummy implementation")
        return y
        
    # Set TensorFlow logging level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Convert to pandas DataFrame
    y_pd = pd.DataFrame(y)
    
    # Initialize and train the model
    model = multinet.MultiNet()
    model.fit(y_pd, cell_subset=1, minVMR=0.5)
    
    # Predict and convert back to numpy
    imputed_data = model.predict(y_pd)
    return imputed_data.to_numpy()

def run_scvi(y):
    """Run the scVI imputation method on input data."""
    if not IMPORTED_METHODS:
        print("scVI not available, using dummy implementation")
        return y
        
    return run_scvi(y)

if __name__ == "__main__":
    main()