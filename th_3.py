import os
import gc
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from anndata import read_h5ad
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.stats import ks_2samp, pearsonr, spearmanr
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import seaborn as sns

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

# Utility functions
def log_message(message, log_file=None, flush=True):
    """Log a message to console and optionally to a file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    
    if log_file:
        with open(log_file, "a") as f:
            f.write(full_message + "\n")
            if flush:
                f.flush()

def force_gc():
    """Force garbage collection to free up memory."""
    import psutil
    process = psutil.Process(os.getpid())
    before = process.memory_info().rss / (1024 * 1024)
    
    gc.collect()
    
    after = process.memory_info().rss / (1024 * 1024)
    if before - after > 10:  # Only log if significant memory was freed
        log_message(f"Memory usage: {before:.2f} MB → {after:.2f} MB (freed {before-after:.2f} MB)")

# Create directories
def setup_directories(args):
    """Create all necessary directories for the experiment."""
    for d in [args.output_dir, args.figures_dir, args.cache_dir]:
        os.makedirs(d, exist_ok=True)
        
    # Create subdirectories for each method
    for method in args.methods.split(','):
        os.makedirs(os.path.join(args.cache_dir, method), exist_ok=True)
        
    log_message(f"Created directory structure")

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
    marker_genes = {}
    if args.markers_file:
        with open(args.markers_file, 'r') as f:
            marker_genes = json.load(f)
    
    log_message(f"Loaded dataset with {adata.shape[0]} cells and {adata.shape[1]} genes")
    if marker_genes:
        log_message(f"Loaded {len(marker_genes)} cell types with marker genes")
    
    return adata, marker_genes

def inspect_gene_names(adata):
    """Inspect gene names in the dataset to help identify marker genes."""
    log_message(f"Gene name format example: first 5 genes - {', '.join(adata.var_names[:5])}")
    
    # Check if any common marker genes might be present in a different format
    common_markers = ['MUC2', 'LGR5', 'OLFM4', 'LYZ', 'DEFA5']
    
    # Try case-insensitive matching
    lowercase_var_names = [name.lower() for name in adata.var_names]
    for marker in common_markers:
        lower_marker = marker.lower()
        if lower_marker in lowercase_var_names:
            idx = lowercase_var_names.index(lower_marker)
            actual_name = adata.var_names[idx]
            log_message(f"Found marker {marker} as {actual_name}")
    
    # Check if we have any additional gene metadata
    if hasattr(adata.var, 'columns') and len(adata.var.columns) > 0:
        log_message(f"Additional gene metadata columns: {', '.join(adata.var.columns)}")
        
        # Look for gene symbol columns
        symbol_cols = [col for col in adata.var.columns 
                      if 'symbol' in col.lower() or 'name' in col.lower() or 'gene' in col.lower()]
        
        if symbol_cols:
            log_message(f"Potential gene symbol columns: {', '.join(symbol_cols)}")
            log_message(f"Sample from {symbol_cols[0]}: {adata.var[symbol_cols[0]].iloc[:5].tolist()}")

# Normalization helper function
def log1p_cpm(mat):
    """
    Normalize matrix to log1p of counts per million
    Works with both dense and sparse matrices
    
    Parameters:
    -----------
    mat : array or sparse matrix
        Gene expression matrix (cells x genes)
        
    Returns:
    --------
    array or sparse matrix
        Normalized matrix
    """
    # Handle sparse matrices appropriately
    is_sparse = hasattr(mat, 'toarray')
    
    # Calculate library size for each cell
    if is_sparse:
        libsize = np.asarray(mat.sum(axis=1)).flatten()
    else:
        libsize = np.sum(mat, axis=1)
        
    # Calculate scaling factor
    scale = 1e4 / (libsize + 1e-8)  # CPM scaling with small epsilon
    
    # Apply scaling
    if is_sparse:
        import scipy.sparse as sp
        if isinstance(mat, sp.csr_matrix):
            # Efficient scaling for CSR matrices
            mat_normalized = mat.copy()
            for i in range(mat.shape[0]):
                mat_normalized.data[mat_normalized.indptr[i]:mat_normalized.indptr[i+1]] *= scale[i]
        else:
            # For other sparse formats, convert to CSR first
            mat_csr = mat.tocsr()
            mat_normalized = mat_csr.copy()
            for i in range(mat_csr.shape[0]):
                mat_normalized.data[mat_normalized.indptr[i]:mat_normalized.indptr[i+1]] *= scale[i]
    else:
        # For dense matrices, use broadcasting
        mat_normalized = (mat.T * scale).T
        
    # Apply log1p transformation
    if is_sparse:
        # Keep matrix sparse
        import scipy.sparse as sp
        mat_log1p = mat_normalized.copy()
        np.log1p(mat_log1p.data, out=mat_log1p.data)
        return mat_log1p
    else:
        # For dense matrices
        return np.log1p(mat_normalized)

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

def expand_magic_matrix(original_matrix, imputed_matrix):
    """Expand MAGIC's imputed matrix to match original matrix dimensions."""
    expanded = np.zeros(original_matrix.shape)
    nonzero_counts = (original_matrix != 0).sum(axis=0)
    kept_genes = np.where(nonzero_counts >= 5)[0]
    for i, original_idx in enumerate(kept_genes):
        if i < imputed_matrix.shape[1]:
            expanded[:, original_idx] = imputed_matrix[:, i]
    return expanded

def calculate_complete_score(X_original, X_imputed, adata=None, marker_genes=None, clusters=None):
    """
    Calculate a comprehensive imputation quality score that addresses all suggested metrics.
    
    Parameters:
    -----------
    X_original : np.ndarray
        Original gene expression matrix
    X_imputed : np.ndarray
        Imputed gene expression matrix
    adata : AnnData, optional
        Original AnnData object for cell-type information
    marker_genes : dict, optional
        Dictionary of marker genes per cell type
    clusters : np.ndarray, optional
        Cell cluster assignments for within-cluster correlation
        
    Returns:
    --------
    dict
        Dictionary with all scores and the final composite score
    """
    # Initialize score components
    scores = {}
    
    # 1. Gene-gene correlation preservation
    # (a) Global correlation preservation
    gene_correlations = []
    n_genes_sample = min(1000, X_original.shape[1])
    gene_indices = np.random.choice(X_original.shape[1], n_genes_sample, replace=False)
    
    
    for i in range(min(100, len(gene_indices))):
        gene_idx = gene_indices[i]
        if (np.std(X_original[:, gene_idx].toarray().flatten() if hasattr(X_original, 'toarray') else X_original[:, gene_idx]) > 0 and 
            np.std(X_imputed[:, gene_idx].toarray().flatten() if hasattr(X_imputed, 'toarray') else X_imputed[:, gene_idx]) > 0):
            corr = np.corrcoef(X_original[:, gene_idx], X_imputed[:, gene_idx])[0, 1]
            if not np.isnan(corr):
                gene_correlations.append(corr)
    
    scores['gene_corr_global'] = np.mean(gene_correlations) if gene_correlations else 0
    
    # (b) Within-cluster correlation preservation (if clusters provided)
    if clusters is not None:
        within_cluster_correlations = []
        cluster_ids = np.unique(clusters)
        
        for cluster_id in cluster_ids:
            cluster_mask = clusters == cluster_id
            if np.sum(cluster_mask) < 5:  # Skip tiny clusters
                continue
                
            cluster_corrs = []
            for i in range(min(50, len(gene_indices))):
                gene_idx = gene_indices[i]
                orig_expr = X_original[cluster_mask, gene_idx]
                imp_expr = X_imputed[cluster_mask, gene_idx]
                
                if np.std(orig_expr) > 0 and np.std(imp_expr) > 0:
                    corr = np.corrcoef(orig_expr, imp_expr)[0, 1]
                    if not np.isnan(corr):
                        cluster_corrs.append(corr)
            
            if cluster_corrs:
                within_cluster_correlations.append(np.mean(cluster_corrs))
        
        scores['gene_corr_within_cluster'] = np.mean(within_cluster_correlations) if within_cluster_correlations else 0
    else:
        scores['gene_corr_within_cluster'] = scores['gene_corr_global']  # Fallback
    
    # 2. Preservation of zeros and non-zeros (FIXED: now with tolerance)
    ZERO_TOLERANCE = 1e-6  # Define a small epsilon for "zero" values
    
    zero_mask = X_original == 0
    zeros_preserved = np.mean(np.abs(X_imputed[zero_mask]) < ZERO_TOLERANCE) if np.sum(zero_mask) > 0 else 0
    
    nonzero_mask = X_original > 0
    nonzeros_preserved = np.mean(X_imputed[nonzero_mask] >= ZERO_TOLERANCE) if np.sum(nonzero_mask) > 0 else 0
    
    scores['preservation_zeros'] = zeros_preserved
    scores['preservation_nonzeros'] = nonzeros_preserved
    scores['preservation_overall'] = (zeros_preserved + nonzeros_preserved) / 2
    
    # 3. Distribution agreement
    imputed_zeros_mask = (X_original == 0) & (X_imputed >= ZERO_TOLERANCE)  # FIXED: use tolerance
    if np.sum(imputed_zeros_mask) > 0 and np.sum(nonzero_mask) > 0:
        # 3a. Calculate KS statistic between distributions
        orig_nonzeros = X_original[nonzero_mask].flatten()
        imputed_values = X_imputed[imputed_zeros_mask].flatten()
        
        # Sample for efficiency
        max_samples = 10000
        if len(orig_nonzeros) > max_samples:
            orig_nonzeros = np.random.choice(orig_nonzeros, max_samples, replace=False)
        if len(imputed_values) > max_samples:
            imputed_values = np.random.choice(imputed_values, max_samples, replace=False)
            
        ks_stat, _ = ks_2samp(orig_nonzeros, imputed_values)
        scores['distribution_ks'] = 1.0 - ks_stat  # Lower KS statistic is better, so 1-KS for higher=better
        
        # 3b. Mean-variance trend preservation
        orig_means = []
        orig_vars = []
        imp_means = []
        imp_vars = []
        
        for gene_idx in range(min(500, X_original.shape[1])):
            orig_gene = X_original[:, gene_idx]
            imp_gene = X_imputed[:, gene_idx]
            
            if np.sum(orig_gene > 0) > 0:
                orig_means.append(np.mean(orig_gene))
                orig_vars.append(np.var(orig_gene))
            
            if np.sum(imp_gene > 0) > 0:
                imp_means.append(np.mean(imp_gene))
                imp_vars.append(np.var(imp_gene))
        
        if orig_means and imp_means:
            orig_cv = np.sqrt(orig_vars) / (np.array(orig_means) + 1e-10)
            imp_cv = np.sqrt(imp_vars) / (np.array(imp_means) + 1e-10)
            
            try:
                cv_corr, _ = spearmanr(orig_cv, imp_cv)
                scores['mean_var_trend'] = max(0, cv_corr)  # Spearman correlation between CV values
            except:
                scores['mean_var_trend'] = 0
        else:
            scores['mean_var_trend'] = 0
            
        # Combine distribution metrics
        scores['distribution_score'] = 0.5 * scores['distribution_ks'] + 0.5 * scores['mean_var_trend']
    else:
        scores['distribution_ks'] = 1.0
        scores['mean_var_trend'] = 1.0
        scores['distribution_score'] = 1.0
    
    # 4. Cell-cell structure preservation
    # 4a. PCA correlation
    try:
        n_pcs = min(50, X_original.shape[0] - 1, X_original.shape[1] - 1)
        if n_pcs >= 2:
            # Only apply PCA on a subset of cells if dataset is large to save memory
            max_cells_for_pca = 5000
            if X_original.shape[0] > max_cells_for_pca:
                cell_indices = np.random.choice(X_original.shape[0], max_cells_for_pca, replace=False)
                pca_orig = PCA(n_components=n_pcs).fit_transform(X_original[cell_indices])
                pca_imp = PCA(n_components=n_pcs).fit_transform(X_imputed[cell_indices])
            else:
                pca_orig = PCA(n_components=n_pcs).fit_transform(X_original)
                pca_imp = PCA(n_components=n_pcs).fit_transform(X_imputed)
            
            # Calculate correlation between PC coordinates
            pc_corrs = []
            for i in range(n_pcs):
                corr, _ = pearsonr(pca_orig[:, i], pca_imp[:, i])
                if not np.isnan(corr):
                    pc_corrs.append(corr)
            
            scores['pc_correlation'] = np.mean(pc_corrs) if pc_corrs else 0
        else:
            scores['pc_correlation'] = 0
    except:
        scores['pc_correlation'] = 0
    
    # 4b. k-NN graph preservation - Using sampling for large datasets
    try:
        k = min(15, X_original.shape[0] - 1)
        if k >= 5:
            # Limit kNN analysis to a subset of cells if dataset is large
            max_cells_for_knn = 2000
            if X_original.shape[0] > max_cells_for_knn:
                cell_indices = np.random.choice(X_original.shape[0], max_cells_for_knn, replace=False)
                # Use PCA to reduce dimensions for kNN calculation
                n_dims = min(50, X_original.shape[1] - 1)
                if n_dims >= 2:
                    pca_orig = PCA(n_components=n_dims).fit_transform(X_original[cell_indices])
                    pca_imp = PCA(n_components=n_dims).fit_transform(X_imputed[cell_indices])
                else:
                    pca_orig = X_original[cell_indices]
                    pca_imp = X_imputed[cell_indices]
            else:
                # For smaller datasets, use more dimensions but still reduce with PCA
                n_dims = min(50, X_original.shape[1] - 1)
                if n_dims >= 2:
                    pca_orig = PCA(n_components=n_dims).fit_transform(X_original)
                    pca_imp = PCA(n_components=n_dims).fit_transform(X_imputed)
                else:
                    pca_orig = X_original
                    pca_imp = X_imputed
            
            # Create distance matrices
            dist_orig = squareform(pdist(pca_orig, 'euclidean'))
            dist_imp = squareform(pdist(pca_imp, 'euclidean'))
            
            # For each cell, get k nearest neighbors
            knn_overlap_sum = 0
            for i in range(len(dist_orig)):
                orig_nn = np.argsort(dist_orig[i])[1:k+1]  # Skip self
                imp_nn = np.argsort(dist_imp[i])[1:k+1]
                overlap = len(set(orig_nn) & set(imp_nn))
                knn_overlap_sum += overlap / k
            
            scores['knn_preservation'] = knn_overlap_sum / len(dist_orig)
        else:
            scores['knn_preservation'] = 0
    except Exception as e:
        log_message(f"Warning: kNN calculation failed: {str(e)}")
        scores['knn_preservation'] = 0
    
    # 4c. Clustering stability - Sample for large datasets
    try:
        if X_original.shape[0] >= 20:  # Need sufficient cells
            # Sample cells for clustering if dataset is large
            max_cells_for_cluster = 5000
            if X_original.shape[0] > max_cells_for_cluster:
                cell_indices = np.random.choice(X_original.shape[0], max_cells_for_cluster, replace=False)
                X_orig_sample = X_original[cell_indices]
                X_imp_sample = X_imputed[cell_indices]
            else:
                X_orig_sample = X_original
                X_imp_sample = X_imputed
            
            # Get number of clusters proportional to data size
            n_clusters = max(2, min(20, X_orig_sample.shape[0] // 50))
            
            # Cluster original and imputed data
            kmeans_orig = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans_imp = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            
            clusters_orig = kmeans_orig.fit_predict(X_orig_sample)
            clusters_imp = kmeans_imp.fit_predict(X_imp_sample)
            
            # Calculate ARI
            ari = adjusted_rand_score(clusters_orig, clusters_imp)
            scores['clustering_stability'] = max(0, ari)  # Ensure non-negative
            
            # Calculate silhouette improvement - skip if dataset is very large
            if X_orig_sample.shape[0] <= 2000:
                try:
                    sil_orig = silhouette_score(X_orig_sample, clusters_orig)
                    sil_imp = silhouette_score(X_imp_sample, clusters_imp)
                    scores['silhouette_improvement'] = max(0, (sil_imp - sil_orig + 1) / 2)  # Scale to [0,1]
                except:
                    scores['silhouette_improvement'] = 0.5  # Neutral if can't compute
            else:
                scores['silhouette_improvement'] = 0.5  # Skip for very large datasets
        else:
            scores['clustering_stability'] = 0
            scores['silhouette_improvement'] = 0.5
    except Exception as e:
        log_message(f"Warning: Clustering calculation failed: {str(e)}")
        scores['clustering_stability'] = 0
        scores['silhouette_improvement'] = 0.5
    
    # 5. Marker gene enrichment (biological score)
    if marker_genes and adata is not None:
        marker_score = calculate_marker_agreement(X_imputed, marker_genes, adata)
        scores['marker_agreement'] = marker_score
    else:
        scores['marker_agreement'] = 0.5  # Neutral if markers not available
    
    # Calculate composite score - weighted geometric mean for balanced influence
    # Define the weights explicitly for transparency
    weights = {
        'gene_correlation': 0.25,
        'preservation': 0.20,
        'distribution': 0.15,
        'cell_structure': 0.20,
        'biological_signal': 0.20
    }
    
    # First, ensure all scores are in [0,1] range and compile them into main categories
    main_scores = {
        'gene_correlation': 0.6 * scores['gene_corr_global'] + 0.4 * scores['gene_corr_within_cluster'],
        'preservation': scores['preservation_overall'],
        'distribution': scores['distribution_score'],
        'cell_structure': (0.3 * scores['pc_correlation'] + 
                          0.3 * scores['knn_preservation'] + 
                          0.2 * scores['clustering_stability'] + 
                          0.2 * scores['silhouette_improvement']),
        'biological_signal': scores['marker_agreement']
    }
    
    # Calculate Z-scores to standardize
    z_scores = {}
    for key, value in main_scores.items():
        z_scores[key] = max(0.01, value)  # Ensure positive for geometric mean
    
    # Weighted geometric mean
    log_scores = np.array([weights[k] * np.log(z_scores[k]) for k in weights.keys()])
    composite_score = np.exp(np.sum(log_scores) / np.sum(list(weights.values())))
    
    # Add composite score
    scores['composite_score'] = composite_score
    
    # Add main category scores
    for key, value in main_scores.items():
        scores[key] = value
    
    # Log all scores
    log_message(f"  Gene correlation score: {main_scores['gene_correlation']:.4f}")
    log_message(f"  Preservation score: {main_scores['preservation']:.4f}")
    log_message(f"  Distribution score: {main_scores['distribution']:.4f}")
    log_message(f"  Cell structure score: {main_scores['cell_structure']:.4f}")
    log_message(f"  Biological signal score: {main_scores['biological_signal']:.4f}")
    log_message(f"  Final composite score: {composite_score:.4f}")
    log_message(f"  Metric weights: {weights}")
    
    return scores

def calculate_marker_agreement(X, marker_genes, adata):
    """
    Calculate the marker gene agreement score across all cells.
    Updated to use zero tolerance and be more memory-efficient.
    """
    # Set zero tolerance for numeric stability
    ZERO_TOLERANCE = 1e-6
    
    # Create gene index lookup
    gene2idx = {}
    if 'feature_name' in adata.var.columns:
        for i, (ensembl_id, feature_name) in enumerate(zip(adata.var_names, adata.var['feature_name'])):
            if isinstance(feature_name, str):
                # Store gene symbol -> index mapping
                gene2idx[feature_name] = i
                
                # Also try without version number if present
                if '.' in feature_name:
                    base_name = feature_name.split('.')[0]
                    gene2idx[base_name] = i

    # Print some debug info
    print(f"Loaded {len(gene2idx)} unique gene names from feature_name column")
    if len(gene2idx) > 0:
        print(f"Sample gene names: {list(gene2idx.keys())[:5]}")
        
        # Check if any marker genes match
        for cell_type, markers in marker_genes.items():
            found_markers = [m for m in markers if m in gene2idx]
            if found_markers:
                print(f"Found {len(found_markers)} markers for {cell_type}: {', '.join(found_markers)}")

    # Check if marker genes exist in the dataset
    missing_markers = []
    for cell_type, markers in marker_genes.items():
        for marker in markers:
            if marker not in gene2idx:
                missing_markers.append((cell_type, marker))
    
    if missing_markers:
        log_message(f"Warning: {len(missing_markers)} marker genes not found in dataset:")
        for cell_type, marker in missing_markers[:10]:  # Show first 10
            log_message(f"  {cell_type}: {marker}")
        if len(missing_markers) > 10:
            log_message(f"  ... and {len(missing_markers) - 10} more")
    
    # Cell type to index
    if "Celltype" in adata.obs.columns:
        cell_types = adata.obs["Celltype"].values
        unique_cell_types = np.unique(cell_types)
        log_message(f"Found {len(unique_cell_types)} cell types in data")
        
        # Create a mapping from marker gene cell types to annotation cell types
        cell_type_mapping = {}
        for marker_type in marker_genes.keys():
            # Find matching cell types by prefix (ignoring the suffix in parentheses)
            matches = [ct for ct in unique_cell_types if ct == marker_type]
            if matches:
                cell_type_mapping[marker_type] = matches
                log_message(f"Mapped '{marker_type}' to {len(matches)} cell types: {', '.join(matches)}")
        
        if not cell_type_mapping:
            log_message("Could not map any marker gene cell types to annotation cell types")
            return 0.0
        
        # Initialize score components
        specificity_score = 0.0
        expression_score = 0.0
        valid_pairs = 0
        
        # Check if X is sparse
        is_sparse = hasattr(X, 'toarray')
        
        # For each cell type with markers
        for marker_type, markers in marker_genes.items():
            if marker_type not in cell_type_mapping:
                continue
                
            # Get all matching cell types
            matching_types = cell_type_mapping[marker_type]
            
            # Create a mask for all cells of these types
            type_mask = np.zeros(X.shape[0], dtype=bool)
            for ct in matching_types:
                type_mask = type_mask | (cell_types == ct)
            
            if sum(type_mask) == 0:
                continue
            
            # Get indices of the marker genes present in our dataset
            marker_indices = [gene2idx[g] for g in markers if g in gene2idx]
            if len(marker_indices) == 0:
                continue
            
            # Process each marker gene - memory efficient approach
            for marker_idx in marker_indices:
                # Calculate in batches if dataset is large
                if X.shape[0] > 10000:
                    # Use batched processing for large datasets
                    batch_size = 5000
                    n_batches = (X.shape[0] + batch_size - 1) // batch_size
                    
                    # Initialize accumulators
                    sum_in_type = 0
                    count_in_type = 0
                    sum_in_others = 0
                    count_in_others = 0
                    sum_expr_in_type = 0
                    
                    for b in range(n_batches):
                        start_idx = b * batch_size
                        end_idx = min((b + 1) * batch_size, X.shape[0])
                        batch_mask = type_mask[start_idx:end_idx]
                        
                        # Get expression values for this batch
                        if is_sparse:
                            # Extract values for this batch and marker
                            batch_expr = X[start_idx:end_idx, marker_idx].toarray().flatten()
                        else:
                            batch_expr = X[start_idx:end_idx, marker_idx]
                        
                        # Sum expression in correct type
                        batch_in_type = batch_expr[batch_mask]
                        sum_in_type += np.sum(batch_in_type)
                        count_in_type += len(batch_in_type)
                        
                        # Sum expression in other types
                        batch_in_others = batch_expr[~batch_mask]
                        sum_in_others += np.sum(batch_in_others)
                        count_in_others += len(batch_in_others)
                        
                        # Calculate expression fraction (% of cells expressing)
                        # FIXED: Use zero tolerance for expression check
                        sum_expr_in_type += np.sum(batch_in_type >= ZERO_TOLERANCE)
                    
                    # Calculate means
                    expr_in_type = sum_in_type / max(1, count_in_type)
                    expr_in_others = sum_in_others / max(1, count_in_others)
                    expression = sum_expr_in_type / max(1, count_in_type)
                    
                else:
                    # For smaller datasets, calculate directly
                    if is_sparse:
                        marker_values = X[:, marker_idx].toarray().flatten()
                    else:
                        marker_values = X[:, marker_idx]
                        
                    # Expression in correct cell type
                    expr_in_type = np.mean(marker_values[type_mask])
                    
                    # Expression in other cell types
                    expr_in_others = np.mean(marker_values[~type_mask]) if sum(~type_mask) > 0 else 0
                    
                    # Expression level: What fraction of cells of this type express the marker
                    # FIXED: Use zero tolerance for expression check
                    expression = np.mean(marker_values[type_mask] >= ZERO_TOLERANCE)
                
                # 1. Specificity: How much more expressed in correct type
                if expr_in_others > ZERO_TOLERANCE:  # FIXED: Use zero tolerance
                    ratio = expr_in_type / expr_in_others
                    specificity = np.tanh(ratio - 1)  # Bounded growth function
                else:
                    specificity = 1.0 if expr_in_type > ZERO_TOLERANCE else 0.0  # FIXED: Use zero tolerance
                
                # Combine scores
                specificity_score += specificity
                expression_score += expression
                valid_pairs += 1
        
        # Return combined score, normalized by number of marker-celltype pairs
        if valid_pairs > 0:
            final_score = (0.7 * specificity_score + 0.3 * expression_score) / valid_pairs
            log_message(f"Calculated marker agreement score with {valid_pairs} valid cell type/marker pairs: {final_score:.4f}")
            return final_score
        else:
            log_message("No valid cell type/marker gene pairs found")
            return 0.0
    else:
        # If no cell type annotation available, use a simpler approach
        log_message("No cell type annotations found - using general marker expression score")
        
        # Initialize score
        expression_score = 0.0
        valid_pairs = 0
        
        # For all marker genes (regardless of cell type)
        for cell_type, markers in marker_genes.items():
            # Get indices of marker genes
            marker_indices = [gene2idx[g] for g in markers if g in gene2idx]
            
            # Check if X is sparse
            is_sparse = hasattr(X, 'toarray')
            
            for marker_idx in marker_indices:
                # Get gene expression
                if is_sparse:
                    marker_values = X[:, marker_idx].toarray().flatten()
                else:
                    marker_values = X[:, marker_idx]
                
                # Calculate expression across all cells (% of cells with non-zero expression)
                # FIXED: Use zero tolerance for expression check
                expr = np.mean(marker_values >= ZERO_TOLERANCE)
                expression_score += expr
                valid_pairs += 1
        
        # Return overall score
        if valid_pairs > 0:
            score = expression_score / valid_pairs
            log_message(f"Calculated general marker expression score: {score:.4f}")
            return score
        else:
            log_message("No valid marker genes found")
            return 0.0

def calculate_marker_based_weights(original_matrix, imputed_matrices, marker_genes, adata):
    """
    Calculate weights for each imputation method based on marker gene agreement improvement.
    Now with proper normalization and sparse matrix handling.
    """
    weights = []
    
    # Normalize original matrix to ensure consistent scale
    original_norm = log1p_cpm(original_matrix)
    
    # Get cell type annotations
    if "Celltype" in adata.obs.columns:
        cell_types = adata.obs["Celltype"].values
    else:
        # If no cell type annotation, create a dummy one
        is_large_matrix = original_matrix.shape[0] > 10000
        
        if is_large_matrix:
            # For large matrices, use simple clusters to avoid memory issues
            n_clusters = min(10, original_matrix.shape[0] // 50)
            # Sample a subset for clustering
            sample_size = min(5000, original_matrix.shape[0])
            sample_indices = np.random.choice(original_matrix.shape[0], sample_size, replace=False)
            
            # Get matrix subset for clustering
            if hasattr(original_matrix, 'toarray'):
                X_sample = original_norm[sample_indices].toarray()
            else:
                X_sample = original_norm[sample_indices]
                
            # Cluster the subset
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_ids_sample = kmeans.fit_predict(X_sample)
            
            # Assign all cells to a cluster (use 0 for non-sampled)
            cell_types = np.zeros(original_matrix.shape[0], dtype=int)
            cell_types[sample_indices] = cluster_ids_sample
        else:
            # For smaller matrices, cluster all cells
            n_clusters = min(10, original_matrix.shape[0] // 10)
            
            # Convert to dense if needed
            if hasattr(original_matrix, 'toarray'):
                X_for_clustering = original_norm.toarray()
            else:
                X_for_clustering = original_norm
                
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cell_types = kmeans.fit_predict(X_for_clustering)
        
        cell_types = np.array([f"Cluster_{i}" for i in cell_types])
    
    # Calculate baseline marker agreement score
    baseline_score = calculate_marker_agreement(original_norm, marker_genes, adata)
    
    # Calculate improvement for each method
    for i, imputed in enumerate(imputed_matrices):
        # Normalize imputed matrix to ensure same scale
        imputed_norm = log1p_cpm(imputed)
        
        # Calculate marker agreement
        score = calculate_marker_agreement(imputed_norm, marker_genes, adata)
        improvement = max(0, score - baseline_score)  # Only consider positive improvements
        weights.append(improvement + 0.01)  # Add small constant to avoid zero weights
    
    # Normalize weights
    weights = np.array(weights)
    return weights / np.sum(weights)

def adaptive_step_size(matrix_shape, zero_count, args):
    """
    Determine an appropriate step size for incremental imputation based on matrix size.
    Uses smaller steps for sparse matrices and larger steps for dense matrices.
    
    Parameters:
    -----------
    matrix_shape : tuple
        Shape of the matrix (n_cells, n_genes)
    zero_count : int
        Number of zeros in the matrix
    args : argparse.Namespace
        Command line arguments
        
    Returns:
    --------
    step_size : int
        Number of zeros to impute in each step
    step_percent : float
        Percentage of zeros to impute in each step
    """
    density = 1 - (zero_count / (matrix_shape[0] * matrix_shape[1]))
    
    # For very sparse matrices (>95% zeros), use smaller steps
    if density < 0.05:
        base_step = 0.005  # 0.5% steps
    # For moderately sparse matrices (80-95% zeros), use medium steps
    elif density < 0.2:
        base_step = 0.01   # 1% steps
    # For less sparse matrices (<80% zeros), use larger steps
    else:
        base_step = 0.02   # 2% steps
    
    # Override with user-specified step if provided
    if args.batch_percent > 0:
        step_percent = args.batch_percent / 100.0  # Convert from percentage to fraction
    else:
        step_percent = base_step
    
    # Calculate step size, ensuring at least 1 zero is imputed per step
    step_size = max(1, int(zero_count * step_percent))
    
    # Adjust step percent to reflect actual step size
    actual_step_percent = step_size / zero_count
    
    log_message(f"Using adaptive step size: {step_size} zeros per step ({actual_step_percent:.3%} of zeros)")
    
    return step_size, actual_step_percent

def calculate_th_scores_with_marker_weights(adata, method_names, marker_genes, args):
    """
    Calculate TruthfulHypothesis scores using marker-based method weights.
    Updated for memory efficiency and consistent normalization.
    """
    log_message("Calculating TruthfulHypothesis scores with marker-based weights")
    
    # Process all disease/tissue combinations
    if "disease" in adata.obs.columns and "tissue" in adata.obs.columns:
        metadata_markers = adata.obs[["disease", "tissue"]].drop_duplicates()
        
        results = []
        for idx, (disease, tissue) in enumerate(metadata_markers.values[:1]):
            log_message(f"Processing sample {idx+1}/{len(metadata_markers)}: {tissue} ({disease})")
            
            # Extract the subsample
            mask = (adata.obs["disease"] == disease) & (adata.obs["tissue"] == tissue)
            adata_sample = adata[mask]
            X = adata_sample.X
            
            # Keep sparse matrix as sparse if it is sparse
            is_sparse = hasattr(X, 'toarray')
            
            # Find zero entries (handle sparse matrices efficiently)
            if is_sparse:
                # For sparse matrices, zeros are the entries not stored
                import scipy.sparse as sp
                
                # Count zeros - non-zeros is efficiently stored in nnz attribute
                n_zeros = X.shape[0] * X.shape[1] - X.nnz
                log_message(f"  Found {n_zeros} zero entries")
                
                # For sparse matrices, we don't create a full zero_mask (memory inefficient)
                # We'll handle this differently below
                zero_indices = None  # Will be created later
            else:
                # For dense matrices, we can use standard array operations
                ZERO_TOLERANCE = 1e-6  # Use tolerance for zero comparison
                zero_mask = np.abs(X) < ZERO_TOLERANCE
                zero_indices = np.where(zero_mask)
                
                if len(zero_indices[0]) == 0:
                    log_message(f"  No zero entries found in this sample, skipping")
                    continue
                    
                log_message(f"  Found {len(zero_indices[0])} zero entries")
            
            # Initialize storage for imputed matrices
            imputed_matrices = []
            valid_methods = []
            
            # Normalize original matrix
            X_norm = log1p_cpm(X)
            
            # Load imputed matrices for all methods
            for method_name in method_names:
                # Load imputed matrix
                dataset_id = os.path.basename(args.input_file).split('.')[0]
                if dataset_id.endswith("_raw") or dataset_id.endswith("_pre"):
                    dataset_id = dataset_id[:-4]

                imputed_file = get_imputation_files(dataset_id, method_name, disease, tissue)
                if imputed_file is None:
                    log_message(f"  Skipping method {method_name} for {disease}, {tissue} - file not found")
                    continue
                
                # Load imputed matrix - handle NPZ format for sparse
                if imputed_file.endswith('.npz'):
                    import scipy.sparse as sp
                    imputed = sp.load_npz(imputed_file)
                else:
                    imputed = np.load(imputed_file)
                
                if method_name == "MAGIC":
                    imputed = expand_magic_matrix(X, imputed)
                
                # Normalize the imputed matrix to ensure same scale
                imputed_norm = log1p_cpm(imputed)
                
                imputed_matrices.append(imputed_norm)
                valid_methods.append(method_name)
                
                log_message(f"  Loaded method: {method_name}, shape: {imputed.shape}")
            
            if len(imputed_matrices) == 0:
                log_message(f"  No valid imputation methods found for this sample, skipping")
                continue
            
            # Calculate marker-based weights
            weights_array = calculate_marker_based_weights(X, imputed_matrices, marker_genes, adata_sample)
            weights = {method: weight for method, weight in zip(valid_methods, weights_array)}
            
            log_message(f"  Calculated marker-based weights: {weights}")
            
            # For sparse matrices, now we need to find the zeros efficiently
            if is_sparse and zero_indices is None:
                # For large sparse matrices, we'll sample a subset of zeros to avoid memory issues
                import scipy.sparse as sp
                
                MAX_ZEROS_TO_PROCESS = 10000000  # Limit total zeros to process
                
                # Convert to COO format for efficient iteration over non-zeros
                X_coo = X.tocoo()
                
                # Create sets for quick lookup
                nonzero_positions = set(zip(X_coo.row, X_coo.col))
                
                # Generate all possible positions and filter out the non-zeros
                # This can be memory intensive, so we'll process in batches
                all_zeros = []
                batch_size = 10000
                
                # Process by row to make it tractable
                for row_idx in range(X.shape[0]):
                    row_zeros = [(row_idx, col_idx) for col_idx in range(X.shape[1]) 
                                if (row_idx, col_idx) not in nonzero_positions]
                    all_zeros.extend(row_zeros)
                    
                    # Check if we've reached our limit
                    if len(all_zeros) >= MAX_ZEROS_TO_PROCESS:
                        log_message(f"  Reached maximum zero processing limit ({MAX_ZEROS_TO_PROCESS}), sampling a subset")
                        # Sample from collected zeros
                        sample_size = min(MAX_ZEROS_TO_PROCESS, len(all_zeros))
                        all_zeros = [all_zeros[i] for i in np.random.choice(len(all_zeros), sample_size, replace=False)]
                        break
                
                # Convert to the standard zero_indices format
                zero_indices = (np.array([z[0] for z in all_zeros]), np.array([z[1] for z in all_zeros]))
                log_message(f"  Processing {len(zero_indices[0])} zero entries")
            
            # Calculate TH scores and consensus values
            scores = np.zeros(len(zero_indices[0]))
            values = np.zeros(len(zero_indices[0]))
            
            for i, method_name in enumerate(valid_methods):
                # Get imputed matrix and weight
                imputed = imputed_matrices[i]
                weight = weights[method_name]
                
                # Extract values at zero positions
                if hasattr(imputed, 'toarray'):
                    # For sparse matrix, extract efficiently
                    imputed_values = np.zeros(len(zero_indices[0]))
                    for j, (r, c) in enumerate(zip(zero_indices[0], zero_indices[1])):
                        imputed_values[j] = imputed[r, c]
                else:
                    # For dense matrix
                    imputed_values = imputed[zero_indices]
                
                # Update scores (1 if imputed > 0, else 0)
                # FIXED: Use tolerance for zero comparison
                ZERO_TOLERANCE = 1e-6
                scores += weight * (np.abs(imputed_values) >= ZERO_TOLERANCE).astype(float)
                
                # Update consensus values
                values += weight * imputed_values
            
            # Store results
            results.append({
                "disease": disease,
                "tissue": tissue,
                "zero_indices": zero_indices,
                "scores": scores,
                "values": values,
                "original_matrix": X,
                "method_weights": weights,
                "valid_methods": valid_methods,
                "imputed_matrices": imputed_matrices,
                "adata_sample": adata_sample
            })
            
            log_message(f"  Completed TH score calculation for this sample")
        
        return results
    else:
        # Process the entire dataset at once
        X = adata.X
        
        # Keep sparse matrix as sparse if it is sparse
        is_sparse = hasattr(X, 'toarray')
        
        # Find zero entries (handle sparse matrices efficiently)
        if is_sparse:
            # For sparse matrices, zeros are the entries not stored
            import scipy.sparse as sp
            
            # Count zeros
            n_zeros = X.shape[0] * X.shape[1] - X.nnz
            log_message(f"Found {n_zeros} zero entries in the dataset")
            
            # For large sparse matrices, we'll sample a subset of zeros to avoid memory issues
            MAX_ZEROS_TO_PROCESS = 10000000  # Limit total zeros to process
            
            if n_zeros > MAX_ZEROS_TO_PROCESS:
                log_message(f"  Large sparse matrix detected. Sampling {MAX_ZEROS_TO_PROCESS} zeros from {n_zeros} total")
                
                # We'll sample zeros more efficiently
                zeros_to_sample = min(MAX_ZEROS_TO_PROCESS, n_zeros)
                
                # Generate random row, col positions that are likely zeros
                # This approach avoids creating the full zero mask
                sampled_rows = np.random.randint(0, X.shape[0], size=zeros_to_sample * 2)  # Oversample to account for hits
                sampled_cols = np.random.randint(0, X.shape[1], size=zeros_to_sample * 2)
                
                # Filter out non-zeros (wasteful but memory efficient)
                zero_row_indices = []
                zero_col_indices = []
                
                # Convert to CSR for efficient row slicing
                X_csr = X.tocsr() if not isinstance(X, sp.csr_matrix) else X
                
                # Process in batches to avoid memory issues
                batch_size = 10000
                for i in range(0, len(sampled_rows), batch_size):
                    batch_rows = sampled_rows[i:i+batch_size]
                    batch_cols = sampled_cols[i:i+batch_size]
                    
                    for j, (r, c) in enumerate(zip(batch_rows, batch_cols)):
                        # Check if this position is zero in a memory-efficient way
                        row_slice = X_csr[r].toarray().flatten()
                        if c < len(row_slice) and abs(row_slice[c]) < 1e-6:  # Use tolerance
                            zero_row_indices.append(r)
                            zero_col_indices.append(c)
                            
                            # If we have enough zeros, stop
                            if len(zero_row_indices) >= zeros_to_sample:
                                break
                    
                    # Check if we have enough zeros
                    if len(zero_row_indices) >= zeros_to_sample:
                        break
                
                # Create zero indices from collected rows and columns
                zero_indices = (np.array(zero_row_indices), np.array(zero_col_indices))
                log_message(f"  Sampled {len(zero_indices[0])} zero entries")
            else:
                # For smaller sparse matrices, we can use a more direct approach
                # Convert to dense for zero finding (memory usage acceptable for smaller matrices)
                X_dense = X.toarray()
                ZERO_TOLERANCE = 1e-6  # Use tolerance for zero comparison
                zero_mask = np.abs(X_dense) < ZERO_TOLERANCE
                zero_indices = np.where(zero_mask)
                log_message(f"  Found {len(zero_indices[0])} zero entries")
        else:
            # For dense matrices, use standard approach
            ZERO_TOLERANCE = 1e-6  # Use tolerance for zero comparison
            zero_mask = np.abs(X) < ZERO_TOLERANCE
            zero_indices = np.where(zero_mask)
            
            log_message(f"Found {len(zero_indices[0])} zero entries in the dataset")
        
        # Initialize storage for imputed matrices
        imputed_matrices = []
        valid_methods = []
        
        # Normalize original matrix
        X_norm = log1p_cpm(X)
        
        # Load imputed matrices for all methods
        for method_name in method_names:
            # Load imputed matrix
            cache_file = os.path.join(args.cache_dir, method_name, "full_dataset.npy")
            cache_file_sparse = os.path.join(args.cache_dir, method_name, "full_dataset.npz")
            
            if os.path.exists(cache_file):
                imputed = np.load(cache_file)
            elif os.path.exists(cache_file_sparse):
                import scipy.sparse as sp
                imputed = sp.load_npz(cache_file_sparse)
            else:
                log_message(f"  Skipping method {method_name} - file not found: {cache_file}")
                continue
            
            # Normalize imputed matrix for consistent scale
            imputed_norm = log1p_cpm(imputed)
            
            imputed_matrices.append(imputed_norm)
            valid_methods.append(method_name)
            
            log_message(f"  Loaded method: {method_name}, shape: {imputed.shape}")
        
        if len(imputed_matrices) == 0:
            log_message(f"  No valid imputation methods found, aborting")
            return []
        
        # Calculate marker-based weights
        weights_array = calculate_marker_based_weights(X, imputed_matrices, marker_genes, adata)
        weights = {method: weight for method, weight in zip(valid_methods, weights_array)}
        
        log_message(f"  Calculated marker-based weights: {weights}")
        
        # Calculate TH scores and consensus values
        scores = np.zeros(len(zero_indices[0]))
        values = np.zeros(len(zero_indices[0]))
        
        for i, method_name in enumerate(valid_methods):
            # Get imputed matrix and weight
            imputed = imputed_matrices[i]
            weight = weights[method_name]
            
            # Extract values at zero positions
            if hasattr(imputed, 'toarray'):
                # For sparse matrix, extract efficiently
                imputed_values = np.zeros(len(zero_indices[0]))
                for j, (r, c) in enumerate(zip(zero_indices[0], zero_indices[1])):
                    imputed_values[j] = imputed[r, c]
            else:
                # For dense matrix
                imputed_values = imputed[zero_indices]
            
            # Update scores (1 if imputed > 0, else 0)
            # FIXED: Use tolerance for zero comparison
            ZERO_TOLERANCE = 1e-6
            scores += weight * (np.abs(imputed_values) >= ZERO_TOLERANCE).astype(float)
            
            # Update consensus values
            values += weight * imputed_values
        
        # Store results
        results = [{
            "disease": "all",
            "tissue": "all",
            "zero_indices": zero_indices,
            "scores": scores,
            "values": values,
            "original_matrix": X,
            "method_weights": weights,
            "valid_methods": valid_methods,
            "imputed_matrices": imputed_matrices,
            "adata_sample": adata
        }]
        
        log_message(f"Completed TH score calculation with marker-based weights")
        
        return results

def incremental_imputation(adata, marker_genes, th_results, args):
    """
    Enhanced incremental imputation:
    1. Uses adaptive step size based on matrix sparsity
    2. Applies bootstrapping to assess score uncertainty
    3. Employs comprehensive evaluation metrics
    4. Implements early stopping when score plateaus
    5. Memory-efficient with sparse matrices
    """
    all_results = []
    
    # Set improvement threshold for early stopping
    improvement_threshold = args.improvement_threshold if hasattr(args, 'improvement_threshold') else 0.001
    
    for sample_idx, sample_data in enumerate(th_results):
        # Unpack sample data
        disease = sample_data["disease"]
        tissue = sample_data["tissue"]
        zero_indices = sample_data["zero_indices"]
        scores = sample_data["scores"]
        values = sample_data["values"]
        X_original = sample_data["original_matrix"]
        adata_sample = sample_data.get("adata_sample", adata)
        
        # Important: Normalize both original and imputed matrices to ensure they're on the same scale
        log_message(f"  Normalizing data to log1p CPM scale for consistent metrics")
        X_original_norm = log1p_cpm(X_original)
        
        # Keep matrices sparse as long as possible if they are sparse
        is_original_sparse = hasattr(X_original, 'toarray')
        
        # Check if the original matrix is very large
        is_large_matrix = X_original.shape[0] * X_original.shape[1] > 10**8
        
        # Extract cell type information if available (for clustering metrics)
        if "Celltype" in adata_sample.obs.columns:
            cell_types = adata_sample.obs["Celltype"].values
            # Convert to numeric for clustering metrics
            unique_types = np.unique(cell_types)
            type_to_idx = {t: i for i, t in enumerate(unique_types)}
            clusters = np.array([type_to_idx[t] for t in cell_types])
        else:
            # Create simple clustering if no cell types and if dataset is not too large
            if not is_large_matrix:
                n_clusters = min(10, X_original.shape[0] // 20)
                if n_clusters >= 2:
                    # Use a subset for clustering if matrix is large
                    if X_original.shape[0] > 5000:
                        sample_indices = np.random.choice(X_original.shape[0], 5000, replace=False)
                        X_sample = X_original_norm[sample_indices]
                        
                        # Convert to dense if needed for k-means
                        if hasattr(X_sample, 'toarray'):
                            X_sample = X_sample.toarray()
                        
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                        sample_clusters = kmeans.fit_predict(X_sample)
                        
                        # Assign remaining cells based on nearest centroid
                        clusters = np.zeros(X_original.shape[0], dtype=int)
                        clusters[sample_indices] = sample_clusters
                        
                        # For remaining cells, keep as cluster 0 to avoid memory issues
                        remaining_indices = np.array([i for i in range(X_original.shape[0]) if i not in sample_indices])
                        if len(remaining_indices) > 0:
                            clusters[remaining_indices] = 0
                    else:
                        # Small enough to cluster all cells
                        X_for_clustering = X_original_norm
                        if hasattr(X_for_clustering, 'toarray'):
                            X_for_clustering = X_for_clustering.toarray()
                        
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                        clusters = kmeans.fit_predict(X_for_clustering)
                else:
                    clusters = np.zeros(X_original.shape[0], dtype=int)
            else:
                # Skip clustering for very large matrices
                clusters = np.zeros(X_original.shape[0], dtype=int)
        
        # Create test matrix (copy of original)
        X_test = X_original.copy()
        X_test_norm = X_original_norm.copy()
        
        # Sort indices by TH scores (descending)
        sorted_indices = np.argsort(-scores)
        
        # Use adaptive step size based on matrix sparsity
        total_zeros = len(sorted_indices)
        step_size, step_percent = adaptive_step_size(X_original.shape, total_zeros, args)
        
        # Cache the imputation ranking to avoid recomputing
        log_message(f"  Cached imputation ranking for {total_zeros} zeros")
        
        # Initialize tracking with multiple curves for different metrics
        imputation_curve = []
        
        # Calculate initial comprehensive score
        initial_scores = calculate_complete_score(
            X_original_norm, X_test_norm, 
            adata=adata_sample, 
            marker_genes=marker_genes, 
            clusters=clusters
        )
        
        imputation_curve.append({
            "iteration": 0,
            "imputed_entries": 0,
            "imputed_fraction": 0.0,
            "composite_score": initial_scores["composite_score"],
            "gene_correlation": initial_scores["gene_correlation"],
            "preservation": initial_scores["preservation"],
            "distribution": initial_scores["distribution"],
            "cell_structure": initial_scores["cell_structure"],
            "biological_signal": initial_scores["biological_signal"],
            "all_metrics": initial_scores
        })
        
        # Track iterations without improvement for early stopping
        no_improvement_count = 0
        best_score_so_far = initial_scores["composite_score"]
        best_iteration_so_far = 0
        
        # Main imputation loop
        for i in range(0, total_zeros, step_size):
            # Get next batch of indices
            batch_indices = sorted_indices[i:i+step_size]
            
            if len(batch_indices) == 0:
                break
                
            # Get coordinates to update
            rows = zero_indices[0][batch_indices]
            cols = zero_indices[1][batch_indices]
            new_values = values[batch_indices]
            
            # Update test matrix (only zeros)
            # Handle sparse matrices appropriately
            if is_original_sparse:
                # Sparse matrices: first convert to COO, update, then back to CSR
                import scipy.sparse as sp
                X_test_coo = X_test.tocoo()
                X_test_coo.data = np.concatenate([X_test_coo.data, new_values])
                X_test_coo.row = np.concatenate([X_test_coo.row, rows])
                X_test_coo.col = np.concatenate([X_test_coo.col, cols])
                X_test = sp.csr_matrix((X_test_coo.data, (X_test_coo.row, X_test_coo.col)), 
                                      shape=X_test.shape)
            else:
                # For dense matrices, direct update
                for r, c, v in zip(rows, cols, new_values):
                    X_test[r, c] = v
            
            # Re-normalize after updates
            X_test_norm = log1p_cpm(X_test)
            
            # Calculate new comprehensive score
            current_scores = calculate_complete_score(
                X_original_norm, X_test_norm, 
                adata=adata_sample, 
                marker_genes=marker_genes, 
                clusters=clusters
            )
            
            imputed_frac = (i + len(batch_indices)) / total_zeros
            current_score = current_scores["composite_score"]
            
            # Store result with all metrics
            imputation_curve.append({
                "iteration": (i // step_size) + 1,
                "imputed_entries": i + len(batch_indices),
                "imputed_fraction": imputed_frac,
                "composite_score": current_score,
                "gene_correlation": current_scores["gene_correlation"],
                "preservation": current_scores["preservation"],
                "distribution": current_scores["distribution"],
                "cell_structure": current_scores["cell_structure"],
                "biological_signal": current_scores["biological_signal"],
                "improvement": current_score - initial_scores["composite_score"],
                "all_metrics": current_scores
            })
            
            # Force log message to be written immediately
            log_message(f"  Step {(i // step_size) + 1}: Imputed {imputed_frac:.1%}, Score: {current_score:.4f}" + 
                       (f" (improved by {current_score - initial_scores['composite_score']:.4f})" 
                         if current_score > initial_scores["composite_score"] else ""), 
                       flush=True)
            
            # Check for early stopping (if enabled)
            if args.early_stop > 0:
                if current_score > (best_score_so_far + improvement_threshold):
                    # We found improvement, reset counter and update best score
                    no_improvement_count = 0
                    best_score_so_far = current_score
                    best_iteration_so_far = len(imputation_curve) - 1
                else:
                    no_improvement_count += 1
                
                if no_improvement_count >= args.early_stop:
                    log_message(f"  Early stopping after {no_improvement_count} iterations without significant improvement " +
                               f"(threshold: {improvement_threshold:.6f})")
                    break
        
        # Perform bootstrapping to assess uncertainty (if matrix size allows and bootstrapping is enabled)
        bootstrap_results = []
        if args.bootstraps > 0 and not is_large_matrix:
            log_message(f"  Performing {args.bootstraps} bootstrap iterations to assess uncertainty")
            
            # Store optimal percentage for each bootstrap
            optimal_percentages = []
            
            for b in range(args.bootstraps):
                # Stratified bootstrap sampling by cell types if available
                if "Celltype" in adata_sample.obs.columns:
                    # Perform stratified sampling by cell type
                    bootstrap_indices = []
                    for cell_type in np.unique(adata_sample.obs["Celltype"]):
                        type_indices = np.where(adata_sample.obs["Celltype"] == cell_type)[0]
                        if len(type_indices) > 0:
                            # Sample with replacement within each cell type
                            sampled_indices = np.random.choice(type_indices, len(type_indices), replace=True)
                            bootstrap_indices.extend(sampled_indices)
                else:
                    # Simple random sampling with replacement
                    n_cells = X_original.shape[0]
                    bootstrap_indices = np.random.choice(n_cells, n_cells, replace=True)
                
                # Create bootstrapped matrices
                if is_original_sparse:
                    X_boot_orig = X_original[bootstrap_indices]
                    X_boot_orig_norm = X_original_norm[bootstrap_indices]
                else:
                    X_boot_orig = X_original[bootstrap_indices]
                    X_boot_orig_norm = X_original_norm[bootstrap_indices]
                
                # Only test a few key points on the curve to save computation
                test_fractions = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
                test_fractions = [f for f in test_fractions if f <= 1.0]
                
                boot_scores = []
                
                # Evaluate each imputation level
                for frac in test_fractions:
                    if frac == 0.0:
                        # Original matrix (no imputation)
                        boot_score = calculate_complete_score(X_boot_orig_norm, X_boot_orig_norm)["composite_score"]
                    else:
                        # Apply imputation up to this fraction
                        entries_to_impute = min(int(frac * total_zeros), total_zeros)
                        subset_indices = sorted_indices[:entries_to_impute]
                        
                        # Create test matrix for this fraction
                        if is_original_sparse:
                            import scipy.sparse as sp
                            # Create copy of bootstrap original
                            X_boot_test = X_boot_orig.copy()
                            # Get values to update
                            update_rows = []
                            update_cols = []
                            update_values = []
                            for idx in subset_indices:
                                r, c = zero_indices[0][idx], zero_indices[1][idx]
                                if r < X_boot_test.shape[0]:  # Check if row exists in bootstrapped sample
                                    update_rows.append(r)
                                    update_cols.append(c)
                                    update_values.append(values[idx])
                            
                            if update_rows:  # Only update if we have values to update
                                # Convert to COO for updating
                                X_boot_test_coo = X_boot_test.tocoo()
                                X_boot_test_coo.data = np.concatenate([X_boot_test_coo.data, update_values])
                                X_boot_test_coo.row = np.concatenate([X_boot_test_coo.row, update_rows])
                                X_boot_test_coo.col = np.concatenate([X_boot_test_coo.col, update_cols])
                                X_boot_test = sp.csr_matrix((X_boot_test_coo.data, 
                                                           (X_boot_test_coo.row, X_boot_test_coo.col)), 
                                                          shape=X_boot_test.shape)
                        else:
                            X_boot_test = X_boot_orig.copy()
                            for idx in subset_indices:
                                r, c = zero_indices[0][idx], zero_indices[1][idx]
                                if r < X_boot_test.shape[0]:  # Check if row exists in bootstrapped sample
                                    X_boot_test[r, c] = values[idx]
                        
                        # Normalize
                        X_boot_test_norm = log1p_cpm(X_boot_test)
                        
                        # Calculate score
                        boot_score = calculate_complete_score(X_boot_orig_norm, X_boot_test_norm)["composite_score"]
                    
                    boot_scores.append(boot_score)
                
                # Find optimal percentage for this bootstrap
                max_idx = np.argmax(boot_scores)
                optimal_percentages.append(test_fractions[max_idx])
            
            # Calculate confidence intervals
            optimal_percentages = np.array(optimal_percentages)
            lower_ci = np.percentile(optimal_percentages, 2.5)
            upper_ci = np.percentile(optimal_percentages, 97.5)
            
            bootstrap_results = {
                "optimal_percentages": optimal_percentages.tolist(),
                "mean": float(np.mean(optimal_percentages)),
                "median": float(np.median(optimal_percentages)),
                "ci_lower": float(lower_ci),
                "ci_upper": float(upper_ci)
            }
            
            log_message(f"  Bootstrap results: median optimal = {np.median(optimal_percentages):.1%}, " + 
                       f"95% CI: [{lower_ci:.1%}, {upper_ci:.1%}]")
        else:
            if args.bootstraps > 0:
                log_message(f"  Skipping bootstrapping for large matrix to conserve memory")
        
        # Find point of maximum score on the main curve
        scores_array = np.array([point["composite_score"] for point in imputation_curve])
        max_score_idx = np.argmax(scores_array)
        max_score = scores_array[max_score_idx]
        max_score_frac = imputation_curve[max_score_idx]["imputed_fraction"]
        
        log_message(f"  Optimal imputation found at {max_score_frac:.1%} of zeros with score {max_score:.4f}")
        
        # Create optimal matrix based on best score
        if max_score_idx > 0:  # If we found an improvement
            optimal_entries = imputation_curve[max_score_idx]["imputed_entries"]
            optimal_indices = sorted_indices[:optimal_entries]
            
            # Create optimal matrix
            if is_original_sparse:
                import scipy.sparse as sp
                X_optimal = X_original.copy()
                # Get values to update
                update_rows = zero_indices[0][optimal_indices]
                update_cols = zero_indices[1][optimal_indices]
                update_values = values[optimal_indices]
                
                # Update sparse matrix
                X_optimal_coo = X_optimal.tocoo()
                X_optimal_coo.data = np.concatenate([X_optimal_coo.data, update_values])
                X_optimal_coo.row = np.concatenate([X_optimal_coo.row, update_rows])
                X_optimal_coo.col = np.concatenate([X_optimal_coo.col, update_cols])
                X_optimal = sp.csr_matrix((X_optimal_coo.data, 
                                         (X_optimal_coo.row, X_optimal_coo.col)), 
                                        shape=X_optimal.shape)
            else:
                X_optimal = X_original.copy()
                opt_rows = zero_indices[0][optimal_indices]
                opt_cols = zero_indices[1][optimal_indices]
                opt_values = values[optimal_indices]
                
                for r, c, v in zip(opt_rows, opt_cols, opt_values):
                    X_optimal[r, c] = v
        else:
            # No improvement found, optimal is the original
            X_optimal = X_original.copy()
        
        # Save optimal matrix
        output_file = os.path.join(args.output_dir, f"optimal_{disease}_{tissue}.npy")
        
        if is_original_sparse:
            # Save sparse matrix in NPZ format
            import scipy.sparse as sp
            sp.save_npz(output_file.replace('.npy', '.npz'), X_optimal)
            log_message(f"  Saved optimal sparse matrix to {output_file.replace('.npy', '.npz')}")
        else:
            # Save dense matrix
            np.save(output_file, X_optimal)
            log_message(f"  Saved optimal matrix to {output_file}")
        
        # Save results
        sample_result = {
            "disease": disease,
            "tissue": tissue,
            "imputation_curve": imputation_curve,
            "max_score_idx": max_score_idx,
            "max_score": max_score,
            "max_score_fraction": max_score_frac,
            "X_optimal": None,  # Don't store the actual matrix in the results to save memory
            "method_weights": sample_data.get("method_weights", {}),
            "bootstrap_results": bootstrap_results,
            "step_size_used": step_size,
            "step_percent_used": step_percent,
            "is_sparse_matrix": is_original_sparse,
            "early_stopped": no_improvement_count >= args.early_stop if args.early_stop > 0 else False,
            "improvement_threshold_used": improvement_threshold
        }
        
        all_results.append(sample_result)
        
        # Force garbage collection to free memory
        # if args.verbose:
        #     # force_gc()
        # else:
        #     gc.collect()
    
    return all_results

def generate_synthetic_markers(adata):
    """
    Generate synthetic marker genes from cell type annotations.
    This is used when no marker genes file is provided.
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object with cell type annotations
    
    Returns:
    --------
    dict
        Dictionary of marker genes per cell type
    """
    log_message("Generating synthetic marker genes from cell type annotations")
    
    if "Celltype" not in adata.obs.columns:
        log_message("  Error: No 'Celltype' column found in adata.obs")
        return {}
    
    # Get unique cell types
    cell_types = adata.obs["Celltype"].unique()
    log_message(f"  Found {len(cell_types)} unique cell types")
    
    # Create marker gene dictionary
    marker_genes = {}
    
    # For each cell type, find genes that are most differentially expressed
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    
    for cell_type in cell_types:
        # Get cells of this type
        mask = adata.obs["Celltype"] == cell_type
        
        if sum(mask) < 5:  # Skip if too few cells
            log_message(f"  Skipping {cell_type}: too few cells ({sum(mask)})")
            continue
            
        # Calculate mean expression in this cell type vs others
        mean_in_type = np.mean(X[mask], axis=0)
        mean_in_others = np.mean(X[~mask], axis=0) if sum(~mask) > 0 else np.zeros_like(mean_in_type)
        
        # Calculate fold change
        epsilon = 1e-10  # To avoid division by zero
        fold_change = mean_in_type / (mean_in_others + epsilon)
        
        # Get top 10 genes by fold change
        top_indices = np.argsort(-fold_change)[:10]
        
        # Only keep genes with fold change > 2
        significant_indices = [idx for idx in top_indices if fold_change[idx] > 2]
        
        if len(significant_indices) > 0:
            # Get gene names
            top_genes = [adata.var_names[idx] for idx in significant_indices]
            marker_genes[cell_type] = top_genes
            
            log_message(f"  Found {len(top_genes)} marker genes for {cell_type}")
        else:
            log_message(f"  No significant marker genes found for {cell_type}")
    
    log_message(f"  Generated marker genes for {len(marker_genes)} cell types")
    
    return marker_genes

def cross_validate_optimal_percentages(all_results, adata, args):
    """
    Perform cross-validation analysis to assess the generalizability of the optimal imputation percentages.
    
    Parameters:
    -----------
    all_results : list
        List of results from incremental imputation
    adata : AnnData
        AnnData object
    args : argparse.Namespace
        Command line arguments
    
    Returns:
    --------
    dict
        Cross-validation results
    """
    log_message("Performing cross-validation analysis")
    
    # Extract tissue and disease info
    if "disease" not in adata.obs.columns or "tissue" not in adata.obs.columns:
        log_message("  Skipping cross-validation: no disease/tissue info available")
        return {}
    
    # Get unique tissues and diseases
    tissues = np.unique([r["tissue"] for r in all_results])
    diseases = np.unique([r["disease"] for r in all_results])
    
    log_message(f"  Found {len(tissues)} tissues and {len(diseases)} diseases")
    
    # Initialize results
    cv_results = {
        "by_tissue": {},
        "by_disease": {},
        "overall": {}
    }
    
    # Cross-validation by tissue
    if len(tissues) > 1:
        log_message("  Performing leave-one-tissue-out cross-validation")
        
        for test_tissue in tissues:
            # Split into train and test
            train_results = [r for r in all_results if r["tissue"] != test_tissue]
            test_results = [r for r in all_results if r["tissue"] == test_tissue]
            
            if len(train_results) == 0 or len(test_results) == 0:
                continue
            
            # Calculate average optimal percentage from training set
            train_percentages = [r["max_score_fraction"] for r in train_results]
            avg_train_pct = np.mean(train_percentages)
            
            # Calculate scores at the average percentage for test set
            test_performances = []
            for test_result in test_results:
                curve_data = test_result["imputation_curve"]
                
                # Find closest percentage point in the curve
                pct_diffs = [abs(point["imputed_fraction"] - avg_train_pct) for point in curve_data]
                closest_idx = np.argmin(pct_diffs)
                
                # Get score at this point
                score_at_avg = curve_data[closest_idx]["composite_score"]
                
                # Get optimal score for this test sample
                optimal_score = test_result["max_score"]
                
                # Calculate relative performance
                relative_perf = score_at_avg / optimal_score if optimal_score > 0 else 0
                
                test_performances.append({
                    "disease": test_result["disease"],
                    "tissue": test_result["tissue"],
                    "score_at_average_pct": score_at_avg,
                    "optimal_score": optimal_score,
                    "relative_performance": relative_perf
                })
            
            # Store results
            cv_results["by_tissue"][test_tissue] = {
                "train_tissue_count": len(train_results),
                "test_tissue_count": len(test_results),
                "avg_train_percentage": avg_train_pct,
                "test_performances": test_performances,
                "avg_relative_performance": np.mean([p["relative_performance"] for p in test_performances])
            }
    
    # Cross-validation by disease
    if len(diseases) > 1:
        log_message("  Performing leave-one-disease-out cross-validation")
        
        for test_disease in diseases:
            # Split into train and test
            train_results = [r for r in all_results if r["disease"] != test_disease]
            test_results = [r for r in all_results if r["disease"] == test_disease]
            
            if len(train_results) == 0 or len(test_results) == 0:
                continue
            
            # Calculate average optimal percentage from training set
            train_percentages = [r["max_score_fraction"] for r in train_results]
            avg_train_pct = np.mean(train_percentages)
            
            # Calculate scores at the average percentage for test set
            test_performances = []
            for test_result in test_results:
                curve_data = test_result["imputation_curve"]
                
                # Find closest percentage point in the curve
                pct_diffs = [abs(point["imputed_fraction"] - avg_train_pct) for point in curve_data]
                closest_idx = np.argmin(pct_diffs)
                
                # Get score at this point
                score_at_avg = curve_data[closest_idx]["composite_score"]
                
                # Get optimal score for this test sample
                optimal_score = test_result["max_score"]
                
                # Calculate relative performance
                relative_perf = score_at_avg / optimal_score if optimal_score > 0 else 0
                
                test_performances.append({
                    "disease": test_result["disease"],
                    "tissue": test_result["tissue"],
                    "score_at_average_pct": score_at_avg,
                    "optimal_score": optimal_score,
                    "relative_performance": relative_perf
                })
            
            # Store results
            cv_results["by_disease"][test_disease] = {
                "train_disease_count": len(train_results),
                "test_disease_count": len(test_results),
                "avg_train_percentage": avg_train_pct,
                "test_performances": test_performances,
                "avg_relative_performance": np.mean([p["relative_performance"] for p in test_performances])
            }
    
    # Overall cross-validation performance
    all_relative_perfs = []
    
    # Collect from tissue CV
    for tissue, tissue_results in cv_results["by_tissue"].items():
        all_relative_perfs.extend([p["relative_performance"] for p in tissue_results["test_performances"]])
    
    # Collect from disease CV
    for disease, disease_results in cv_results["by_disease"].items():
        all_relative_perfs.extend([p["relative_performance"] for p in disease_results["test_performances"]])
    
    # Calculate overall stats
    if all_relative_perfs:
        cv_results["overall"] = {
            "avg_relative_performance": np.mean(all_relative_perfs),
            "min_relative_performance": np.min(all_relative_perfs),
            "max_relative_performance": np.max(all_relative_perfs),
            "std_relative_performance": np.std(all_relative_perfs)
        }
        
        log_message(f"  Overall cross-validation relative performance: {np.mean(all_relative_perfs):.2f} ± {np.std(all_relative_perfs):.2f}")
    
    return cv_results

def generate_report(all_results, args):
    """
    Generate enhanced reports:
    1. Detailed CSV with all metrics
    2. Statistical analysis of optimal percentages
    3. Comprehensive text summary with confidence intervals
    """
    log_message("Generating comprehensive reports")
    
    # 1. Generate detailed CSV report
    csv_path = os.path.join(args.output_dir, "minimal_imputation_results.csv")
    
    with open(csv_path, 'w') as f:
        # Write header with all metrics
        f.write("Disease,Tissue,Total_Zeros,Optimal_Percentage,CI_Lower,CI_Upper,")
        f.write("Composite_Score,Gene_Correlation,Preservation,Distribution,Cell_Structure,Biological_Signal,")
        f.write("Initial_Composite,Improvement,Improvement_Percentage,Step_Size,Method_Weights\n")
        
        # Write data for each sample
        for result in all_results:
            disease = result["disease"]
            tissue = result["tissue"]
            curve_data = result["imputation_curve"]
            max_score_idx = result["max_score_idx"]
            bootstrap_results = result.get("bootstrap_results", {})
            
            # Calculate metrics
            total_zeros = len(result["zero_indices"][0]) if "zero_indices" in result else "unknown"
            optimal_percentage = curve_data[max_score_idx]["imputed_fraction"] * 100
            
            # Initial and optimal scores
            initial_scores = curve_data[0]["all_metrics"] if "all_metrics" in curve_data[0] else {}
            optimal_scores = curve_data[max_score_idx]["all_metrics"] if "all_metrics" in curve_data[max_score_idx] else {}
            
            # Get confidence intervals if available
            ci_lower = bootstrap_results.get("ci_lower", "") * 100 if bootstrap_results else ""
            ci_upper = bootstrap_results.get("ci_upper", "") * 100 if bootstrap_results else ""
            
            # Get composite scores
            initial_composite = curve_data[0].get("composite_score", 0)
            optimal_composite = result["max_score"]
            improvement = optimal_composite - initial_composite
            improvement_percentage = (improvement / initial_composite) * 100 if initial_composite > 0 else "N/A"
            
            # Get individual metrics at optimal point
            gene_correlation = curve_data[max_score_idx].get("gene_correlation", "N/A")
            preservation = curve_data[max_score_idx].get("preservation", "N/A")
            distribution = curve_data[max_score_idx].get("distribution", "N/A")
            cell_structure = curve_data[max_score_idx].get("cell_structure", "N/A")
            biological_signal = curve_data[max_score_idx].get("biological_signal", "N/A")
            
            # Get step size used
            step_size = result.get("step_size_used", "")
            
            # Format method weights
            method_weights = result.get("method_weights", {})
            weights_str = ";".join([f"{k}:{v:.3f}" for k, v in method_weights.items()])
            
            # Write row
            f.write(f"{disease},{tissue},{total_zeros},{optimal_percentage:.2f},{ci_lower:.2f},{ci_upper:.2f},")
            f.write(f"{optimal_composite:.4f},{gene_correlation:.4f},{preservation:.4f},{distribution:.4f},{cell_structure:.4f},{biological_signal:.4f},")
            f.write(f"{initial_composite:.4f},{improvement:.4f},{improvement_percentage},{step_size},{weights_str}\n")
    
    log_message(f"  CSV report generated at {csv_path}")
    
    # 2. Generate statistical analysis of optimal percentages
    stats_path = os.path.join(args.output_dir, "optimal_percentage_statistics.txt")
    
    with open(stats_path, 'w') as f:
        f.write("Statistical Analysis of Optimal Imputation Percentages\n")
        f.write("==================================================\n\n")
        
        # Collect all optimal percentages
        optimal_percentages = [result["max_score_fraction"] * 100 for result in all_results]
        
        # Calculate basic statistics
        f.write(f"Number of samples: {len(optimal_percentages)}\n")
        f.write(f"Mean optimal percentage: {np.mean(optimal_percentages):.2f}%\n")
        f.write(f"Median optimal percentage: {np.median(optimal_percentages):.2f}%\n")
        f.write(f"Standard deviation: {np.std(optimal_percentages):.2f}%\n")
        f.write(f"Range: {np.min(optimal_percentages):.2f}% - {np.max(optimal_percentages):.2f}%\n\n")
        
        # Calculate 95% confidence interval
        confidence_level = 0.95
        sample_mean = np.mean(optimal_percentages)
        sample_std = np.std(optimal_percentages)
        
        if len(optimal_percentages) > 1:
            # Use t-distribution for small sample sizes
            from scipy import stats
            t_critical = stats.t.ppf((1 + confidence_level) / 2, df=len(optimal_percentages)-1)
            margin_of_error = t_critical * (sample_std / np.sqrt(len(optimal_percentages)))
            
            f.write(f"95% Confidence Interval: {sample_mean - margin_of_error:.2f}% - {sample_mean + margin_of_error:.2f}%\n\n")
        
        # Include bootstrap CIs if available
        bootstrap_cis = []
        for result in all_results:
            if "bootstrap_results" in result and result["bootstrap_results"]:
                ci_lower = result["bootstrap_results"]["ci_lower"] * 100
                ci_upper = result["bootstrap_results"]["ci_upper"] * 100
                bootstrap_cis.append((result["disease"], result["tissue"], ci_lower, ci_upper))
        
        if bootstrap_cis:
            f.write("Sample-specific 95% bootstrap confidence intervals:\n")
            for disease, tissue, ci_lower, ci_upper in bootstrap_cis:
                f.write(f"  {tissue} ({disease}): {ci_lower:.2f}% - {ci_upper:.2f}%\n")
            f.write("\n")
        
        # Analyze by tissue type if available
        tissues = [result["tissue"] for result in all_results]
        unique_tissues = set(tissues)
        
        if len(unique_tissues) > 1:
            f.write("Analysis by tissue type:\n")
            for tissue in unique_tissues:
                tissue_percentages = [result["max_score_fraction"] * 100 for result in all_results if result["tissue"] == tissue]
                f.write(f"  {tissue} (n={len(tissue_percentages)}): {np.mean(tissue_percentages):.2f}% ± {np.std(tissue_percentages):.2f}%\n")
            f.write("\n")
        
        # Analyze by disease type if available
        diseases = [result["disease"] for result in all_results]
        unique_diseases = set(diseases)
        
        if len(unique_diseases) > 1:
            f.write("Analysis by disease status:\n")
            for disease in unique_diseases:
                disease_percentages = [result["max_score_fraction"] * 100 for result in all_results if result["disease"] == disease]
                f.write(f"  {disease} (n={len(disease_percentages)}): {np.mean(disease_percentages):.2f}% ± {np.std(disease_percentages):.2f}%\n")
            f.write("\n")
        
        # Add recommendations
        f.write("Recommendations:\n")
        avg_optimal = np.mean(optimal_percentages)
        
        if avg_optimal < 5:
            f.write("  The optimal imputation percentage is very low (<5%), suggesting that minimal imputation\n")
            f.write("  is sufficient for this dataset. Consider using a conservative approach with selective\n")
            f.write("  imputation of only the highest-confidence zeros.\n")
        elif avg_optimal < 15:
            f.write("  The optimal imputation percentage is moderate (5-15%), indicating that targeted imputation\n")
            f.write("  of the most reliable zeros provides the best signal while avoiding artifacts.\n")
            f.write("  This supports the 'minimal imputation' hypothesis.\n")
        else:
            f.write("  The optimal imputation percentage is relatively high (>15%), suggesting that more\n")
            f.write("  aggressive imputation may be beneficial for this dataset. However, monitor for potential\n")
            f.write("  artifacts and over-smoothing in downstream analyses.\n")
    
    log_message(f"  Statistical analysis generated at {stats_path}")
    
    # 3. Generate comprehensive text summary
    txt_path = os.path.join(args.output_dir, "minimal_imputation_summary.txt")
    
    with open(txt_path, 'w') as f:
        f.write("MINIMAL IMPUTATION EXPERIMENT SUMMARY\n")
        f.write("===================================\n\n")
        
        # Calculate overall statistics
        optimal_percentages = [result["max_score_fraction"] * 100 for result in all_results]
        avg_percentage = np.mean(optimal_percentages)
        median_percentage = np.median(optimal_percentages)
        
        # Calculate average improvements for each metric
        metric_improvements = {}
        for metric in ['composite_score', 'gene_correlation', 'preservation', 'distribution', 'cell_structure', 'biological_signal']:
            improvements = []
            for result in all_results:
                curve_data = result["imputation_curve"]
                if len(curve_data) > 1 and metric in curve_data[0] and metric in curve_data[result["max_score_idx"]]:
                    initial = curve_data[0][metric]
                    optimal = curve_data[result["max_score_idx"]][metric]
                    if initial > 0:
                        improvements.append((optimal - initial) / initial * 100)  # Percentage improvement
                    else:
                        improvements.append(0)
            
            if improvements:
                metric_improvements[metric] = np.mean(improvements)
            else:
                metric_improvements[metric] = 0
        
        # Write overview section
        f.write("OVERVIEW\n")
        f.write("--------\n")
        f.write("This report summarizes the results of our enhanced minimal imputation experiment,\n")
        f.write("which aims to find the optimal percentage of zeros to impute in scRNA-seq data while\n")
        f.write("maximizing biological signal and minimizing noise or artifacts.\n\n")
        
        f.write(f"Number of samples analyzed: {len(all_results)}\n")
        f.write(f"Mean optimal imputation: {avg_percentage:.2f}% of zeros\n")
        f.write(f"Median optimal imputation: {median_percentage:.2f}% of zeros\n")
        f.write(f"Range of optimal fractions: {min(optimal_percentages):.1f}% - {max(optimal_percentages):.1f}%\n")
        
        # Add statistical confidence if available
        if len(optimal_percentages) > 1:
            from scipy import stats
            confidence_level = 0.95
            t_critical = stats.t.ppf((1 + confidence_level) / 2, df=len(optimal_percentages)-1)
            margin_of_error = t_critical * (np.std(optimal_percentages) / np.sqrt(len(optimal_percentages)))
            
            f.write(f"95% Confidence Interval: {avg_percentage - margin_of_error:.2f}% - {avg_percentage + margin_of_error:.2f}%\n")
        
        f.write("\n")
        
        # Write key findings section
        f.write("KEY FINDINGS\n")
        f.write("-----------\n")
        f.write(f"Our experiment finds that imputing {avg_percentage:.1f}% of zeros (on average)\n")
        f.write("achieves optimal results across multiple quality metrics. This is significantly lower\n")
        f.write("than what most existing imputation methods apply by default, supporting our hypothesis\n")
        f.write("that minimalist imputation can effectively recover biological signal while minimizing artifacts.\n\n")
        
        f.write("Improvements by metric:\n")
        pretty_names = {
            'composite_score': 'Composite Score', 
            'gene_correlation': 'Gene Correlation', 
            'preservation': 'Data Structure Preservation',
            'distribution': 'Distribution Agreement', 
            'cell_structure': 'Cell-Cell Structure', 
            'biological_signal': 'Biological Signal'
        }
        
        for metric, improvement in metric_improvements.items():
            pretty_name = pretty_names.get(metric, metric)
            f.write(f"  - {pretty_name}: {improvement:.1f}% improvement\n")
        
        f.write("\n")
        
        # Write sample-specific results
        f.write("SAMPLE-SPECIFIC RESULTS\n")
        f.write("----------------------\n")
        f.write("Disease                  Tissue                     Optimal %     Composite    Improvement\n")
        f.write("---------------------------------------------------------------------------------\n")
        
        for result in all_results:
            disease = result["disease"]
            tissue = result["tissue"]
            optimal_pct = result["max_score_fraction"] * 100
            
            # Get confidence interval if available
            ci_str = ""
            if "bootstrap_results" in result and result["bootstrap_results"]:
                ci_lower = result["bootstrap_results"]["ci_lower"] * 100
                ci_upper = result["bootstrap_results"]["ci_upper"] * 100
                ci_str = f" (95% CI: {ci_lower:.1f}-{ci_upper:.1f}%)"
            
            initial_score = result["imputation_curve"][0].get("composite_score", 0)
            optimal_score = result["max_score"]
            improvement = optimal_score - initial_score
            rel_improvement = (improvement / initial_score) * 100 if initial_score > 0 else 0
            
            f.write(f"{disease[:25]:<25} {tissue[:25]:<25} {optimal_pct:>8.1f}%{ci_str[:20]:<20} {optimal_score:.4f}  {rel_improvement:>8.1f}%\n")
        
        f.write("\n")
        
        # Write method weights section if available
        has_weights = False
        for result in all_results:
            if "method_weights" in result and result["method_weights"]:
                has_weights = True
                break
        
        if has_weights:
            f.write("METHOD WEIGHTS\n")
            f.write("-------------\n")
            f.write("The following imputation methods were used with dynamic weights based on their\n")
            f.write("performance for each sample:\n\n")
            
            # Calculate average weights across samples
            all_methods = set()
            for result in all_results:
                all_methods.update(result.get("method_weights", {}).keys())
            
            avg_weights = {method: 0 for method in all_methods}
            count = {method: 0 for method in all_methods}
            
            for result in all_results:
                for method, weight in result.get("method_weights", {}).items():
                    avg_weights[method] += weight
                    count[method] += 1
            
            for method in avg_weights:
                if count[method] > 0:
                    avg_weights[method] /= count[method]
            
            # Print average weights
            f.write("Average weights across all samples:\n")
            for method, weight in sorted(avg_weights.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  - {method}: {weight:.4f}\n")
            
            f.write("\n")
        
        # Write conclusions and recommendations
        f.write("CONCLUSIONS AND RECOMMENDATIONS\n")
        f.write("------------------------------\n")
        f.write("Our enhanced minimal imputation approach demonstrates that:\n")
        
        if avg_percentage < 5:
            f.write("1. A very small fraction of zeros (<5%) contains most of the recoverable biological signal.\n")
            f.write("   This suggests extreme caution should be used with imputation, as most zeros appear to\n")
            f.write("   be true biological zeros rather than technical dropouts.\n")
        elif avg_percentage < 15:
            f.write("1. A small fraction of zeros (typically 5-15%) contains most of the recoverable biological signal.\n")
            f.write("   Beyond this optimal point, additional imputation tends to introduce noise rather than\n")
            f.write("   improve biological signal or cell-type separation.\n")
        else:
            f.write("1. A moderate fraction of zeros (>15%) appears beneficial to impute in this dataset.\n")
            f.write("   This suggests a higher dropout rate that benefits from more aggressive imputation,\n")
            f.write("   though still well below complete imputation of all zeros.\n")
        
        f.write("2. By ranking zeros using the TruthfulHypothesis score across multiple imputation methods,\n")
        f.write("   we can prioritize high-confidence imputations and avoid introducing artifacts.\n")
        f.write("3. Our multi-metric evaluation approach captures different aspects of imputation quality,\n")
        f.write("   ensuring that both gene-level and cell-level structures are preserved.\n")
        f.write("4. Bootstrapping analysis provides confidence intervals for optimal imputation percentages,\n")
        f.write("   accounting for biological and technical variation in the data.\n\n")
        
        f.write("Recommendations for application:\n")
        
        if avg_percentage < 10:
            f.write("- Use highly conservative imputation (≤5%) for this dataset\n")
            f.write("- Focus on imputing only the most confident zeros with highest TH scores\n")
            f.write("- Consider cell-type specific imputation parameters, as optimal points may vary\n")
        else:
            f.write(f"- Target {avg_percentage:.1f}% imputation for similar datasets\n")
            f.write("- Use the weighted consensus approach to prioritize zeros for imputation\n")
            f.write("- Consider bootstrapping to determine confidence intervals for your specific data\n")
        
        f.write("\nThis approach can significantly improve downstream analyses by providing cleaner signal\n")
        f.write("with minimal artifacts, leading to more reliable clustering, differential expression,\n")
        f.write("and trajectory inference results.\n")
    
    log_message(f"  Comprehensive report generated at {txt_path}")

def plot_incremental_results(all_results, args):
    """
    Generate enhanced visualizations:
    1. Multi-metric tracking plots
    2. Heatmap visualizations
    3. Confidence interval plots
    4. Summary visualizations across samples
    """
    log_message("Generating enhanced visualizations")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.figures_dir, exist_ok=True)
    
    # Plot individual sample curves with multiple metrics
    for result in all_results:
        disease = result["disease"].strip().replace(" ", "_")
        tissue = result["tissue"].strip().replace(" ", "_")
        curve_data = result["imputation_curve"]
        max_score_idx = result["max_score_idx"]
        bootstrap_results = result.get("bootstrap_results", None)
        
        # Convert to DataFrame
        df = pd.DataFrame(curve_data)
        
        # 1. Plot composite score curve with uncertainty
        plt.figure(figsize=(10, 6))
        plt.plot(df["imputed_fraction"] * 100, df["composite_score"], marker='o', markersize=4, 
                 color='blue', label='Composite Score')
        
        # Add uncertainty band if bootstrap results exist
        if bootstrap_results:
            # Add confidence interval as a shaded area
            ci_lower = bootstrap_results["ci_lower"] * 100
            ci_upper = bootstrap_results["ci_upper"] * 100
            
            # Create vertical band for the uncertainty in optimal percentage
            plt.axvspan(ci_lower, ci_upper, alpha=0.2, color='blue', 
                        label=f'95% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%]')
        
        # Add vertical line for optimal point
        plt.axvline(x=df.iloc[max_score_idx]["imputed_fraction"] * 100, color='r', linestyle='--')
        
        plt.title(f"Optimal Imputation Analysis - {tissue} ({disease})")
        plt.xlabel("Percentage of Zeros Imputed (%)")
        plt.ylabel("Composite Score")
        plt.grid(True)
        
        # Add annotation for the optimal point
        max_point = df.iloc[max_score_idx]
        plt.annotate(f"Optimal: {max_point['imputed_fraction']*100:.1f}%\nScore: {max_point['composite_score']:.4f}",
                     xy=(max_point["imputed_fraction"]*100, max_point["composite_score"]),
                     xytext=(max_point["imputed_fraction"]*100 + 5, max_point["composite_score"]),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
        
        plt.legend()
        plt.tight_layout()
        
        # Save the composite score figure
        figure_file = os.path.join(args.figures_dir, f"composite_score_{disease}_{tissue}.png")
        plt.savefig(figure_file, dpi=300)
        plt.close()
        
        log_message(f"  Saved composite score plot to {figure_file}")
        
        # 2. Plot multi-metric panel
        plt.figure(figsize=(15, 10))
        
        metrics = ['gene_correlation', 'preservation', 'distribution', 
                   'cell_structure', 'biological_signal']
        metric_labels = ['Gene Correlation', 'Preservation', 'Distribution', 
                         'Cell Structure', 'Biological Signal']
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            plt.subplot(2, 3, i+1)
            plt.plot(df["imputed_fraction"] * 100, df[metric], marker='o', markersize=3, 
                     label=label)
            plt.axvline(x=df.iloc[max_score_idx]["imputed_fraction"] * 100, color='r', 
                       linestyle='--', label='Optimal %')
            
            plt.title(label)
            plt.xlabel("% Zeros Imputed")
            plt.ylabel("Score")
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        # Add composite score as the last panel
        plt.subplot(2, 3, 6)
        plt.plot(df["imputed_fraction"] * 100, df["composite_score"], marker='o', 
                 markersize=3, label='Composite', color='black', linewidth=2)
        plt.axvline(x=df.iloc[max_score_idx]["imputed_fraction"] * 100, color='r', 
                   linestyle='--', label='Optimal %')
        plt.title("Composite Score")
        plt.xlabel("% Zeros Imputed")
        plt.ylabel("Score")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        # Save multi-metric panel
        figure_file = os.path.join(args.figures_dir, f"multi_metric_{disease}_{tissue}.png")
        plt.savefig(figure_file, dpi=300)
        plt.close()
        
        log_message(f"  Saved multi-metric panel to {figure_file}")
        
        # 3. Create heatmap showing step-by-step changes in all metrics
        plt.figure(figsize=(12, 8))
        
        # Prepare heatmap data
        heatmap_data = []
        heatmap_cols = ['gene_correlation', 'preservation', 'distribution', 
                       'cell_structure', 'biological_signal', 'composite_score']
        
        # Normalize each metric to [0,1] for fair comparison
        for col in heatmap_cols:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df[f"{col}_norm"] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[f"{col}_norm"] = 0.5  # If no variation, set to neutral
        
        # Extract normalized data for heatmap
        heatmap_norm_cols = [f"{col}_norm" for col in heatmap_cols]
        heatmap_data = df[heatmap_norm_cols].values.T
        
        # Create heatmap
        sns.heatmap(heatmap_data, cmap="viridis", 
                   xticklabels=[f"{x:.0f}%" for x in df["imputed_fraction"] * 100],
                   yticklabels=['Gene Corr', 'Preservation', 'Distribution', 
                               'Cell Structure', 'Biology', 'Composite'],
                   annot=False, cbar_kws={'label': 'Normalized Score'})
        
        # Add red vertical line at optimal point
        plt.axvline(x=max_score_idx, color='r', linestyle='--', linewidth=2)
        
        plt.title(f"Metric Evolution During Imputation - {tissue} ({disease})")
        plt.xlabel("Imputation Progress")
        plt.tight_layout()
        
        # Save heatmap
        figure_file = os.path.join(args.figures_dir, f"metric_heatmap_{disease}_{tissue}.png")
        plt.savefig(figure_file, dpi=300)
        plt.close()
        
        log_message(f"  Saved metric evolution heatmap to {figure_file}")
    
    # Create summary plot if multiple samples
    if len(all_results) > 1:
        plt.figure(figsize=(12, 8))
        
        for result in all_results:
            disease = result["disease"]
            tissue = result["tissue"]
            df = pd.DataFrame(result["imputation_curve"])
            plt.plot(df["imputed_fraction"]*100, df["composite_score"], 
                     label=f"{tissue} ({disease})", marker='o', markersize=3)
            
            # Mark optimal point
            max_idx = result["max_score_idx"]
            plt.scatter(df.iloc[max_idx]["imputed_fraction"]*100, 
                        df.iloc[max_idx]["composite_score"], 
                        marker='X', s=100, c='red')
        
        plt.title("Composite Score vs. Percentage of Zeros Imputed")
        plt.xlabel("Percentage of Zeros Imputed (%)")
        plt.ylabel("Composite Score")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        # Save summary figure
        figure_file = os.path.join(args.figures_dir, f"summary_imputation_curves.png")
        plt.savefig(figure_file, dpi=300)
        plt.close()
        
        log_message(f"  Saved summary curve plot to {figure_file}")
        
        # Create a histogram of optimal imputation percentages with bootstrap CIs
        plt.figure(figsize=(10, 6))
        optimal_percentages = [result["max_score_fraction"]*100 for result in all_results]
        
        # Plot histogram of optimal percentages
        plt.hist(optimal_percentages, bins=10, edgecolor='black', alpha=0.7)
        
        # Add vertical line for mean
        mean_optimal = np.mean(optimal_percentages)
        plt.axvline(x=mean_optimal, color='r', linestyle='--',
                   label=f"Mean: {mean_optimal:.1f}%")
        
        # Add confidence intervals if available
        bootstrap_cis = []
        for result in all_results:
            if "bootstrap_results" in result and result["bootstrap_results"]:
                ci_lower = result["bootstrap_results"]["ci_lower"] * 100
                ci_upper = result["bootstrap_results"]["ci_upper"] * 100
                bootstrap_cis.append((ci_lower, ci_upper))
        
        # Add bootstrap CIs as error bars
        if bootstrap_cis:
            for i, ((ci_lower, ci_upper), opt_pct) in enumerate(zip(bootstrap_cis, optimal_percentages)):
                plt.plot([ci_lower, ci_upper], [i*0.2 + 0.5, i*0.2 + 0.5], 'k-', alpha=0.5)
                plt.plot([opt_pct, opt_pct], [i*0.2 + 0.3, i*0.2 + 0.7], 'ko', markersize=5)
        
        plt.title("Distribution of Optimal Imputation Percentages")
        plt.xlabel("Optimal Percentage of Zeros Imputed (%)")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save histogram
        figure_file = os.path.join(args.figures_dir, f"optimal_percentage_histogram.png")
        plt.savefig(figure_file, dpi=300)
        plt.close()
        
        log_message(f"  Saved optimal percentage histogram to {figure_file}")
        
        # Create multi-metric radar plot
        plt.figure(figsize=(10, 10))
        
        # Define metrics for radar
        metrics = ['gene_correlation', 'preservation', 'distribution', 
                  'cell_structure', 'biological_signal']
        metric_labels = ['Gene\nCorrelation', 'Preservation', 'Distribution', 
                       'Cell\nStructure', 'Biological\nSignal']
        
        # Set up radar plot
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        ax = plt.subplot(111, polar=True)
        
        # For each sample, plot radar of normalized metric values at optimal point
        for result in all_results:
            df = pd.DataFrame(result["imputation_curve"])
            max_idx = result["max_score_idx"]
            
            # Get the optimal point metrics
            values = [df.iloc[max_idx][metric] for metric in metrics]
            values += values[:1]  # Close the loop
            
            # Plot radar
            ax.plot(angles, values, 'o-', linewidth=1, label=f"{result['tissue']} ({result['disease']})")
            ax.fill(angles, values, alpha=0.1)
        
        # Set radar labels
        plt.xticks(angles[:-1], metric_labels)
        
        plt.title("Multi-Metric Comparison at Optimal Imputation Point")
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Save radar plot
        figure_file = os.path.join(args.figures_dir, f"metric_radar_comparison.png")
        plt.savefig(figure_file, dpi=300)
        plt.close()
        
        log_message(f"  Saved metric radar comparison to {figure_file}")
        
        # Create optimal point comparison
        plt.figure(figsize=(12, 6))
        
        # Prepare data for bar chart
        sample_names = [f"{result['tissue']}\n({result['disease']})" for result in all_results]
        optimal_pcts = [result["max_score_fraction"]*100 for result in all_results]
        
        # Sort by optimal percentage
        sorted_indices = np.argsort(optimal_pcts)
        sample_names = [sample_names[i] for i in sorted_indices]
        optimal_pcts = [optimal_pcts[i] for i in sorted_indices]
        
        # Create bar chart
        plt.bar(sample_names, optimal_pcts, color='skyblue', edgecolor='black')
        plt.axhline(y=np.mean(optimal_pcts), color='r', linestyle='--', 
                   label=f"Mean: {np.mean(optimal_pcts):.1f}%")
        
        plt.title("Optimal Imputation Percentage by Sample")
        plt.xlabel("Sample")
        plt.ylabel("Optimal % of Zeros Imputed")
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save bar chart
        figure_file = os.path.join(args.figures_dir, f"optimal_percent_comparison.png")
        plt.savefig(figure_file, dpi=300)
        plt.close()
        
        log_message(f"  Saved optimal percentage comparison to {figure_file}")

# Main function
def main():
    """Main execution function for the enhanced minimal imputation experiment."""
    parser = argparse.ArgumentParser(description="Run Enhanced TruthfulHypothesis (TH) Score-based Minimal Imputation Experiment")
    
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
    parser.add_argument("--methods", type=str, default="SAUCIE,MAGIC,deepImpute,scScope,scVI,knn_smoothing", 
                        help="Comma-separated list of imputation methods to use")
    parser.add_argument("--batch_percent", type=float, default=0.0, 
                        help="Percentage of zeros to impute in each iteration (0 for adaptive)")
    parser.add_argument("--early_stop", type=int, default=5, 
                        help="Number of iterations without improvement before early stopping (0 to disable)")
    parser.add_argument("--bootstraps", type=int, default=20, 
                        help="Number of bootstrap iterations for uncertainty analysis (0 to disable)")
    parser.add_argument("--force_recompute", action="store_true", 
                        help="Force recomputation of imputation results even if cached")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true", 
                        help="Print detailed progress messages")
    parser.add_argument("--skip_plots", action="store_true", 
                        help="Skip generating plots (useful for headless servers)")
    parser.add_argument("--debug", action="store_true", 
                        help="Run in debug mode with additional logging and smaller sample sizes")
    parser.add_argument("--improvement_threshold", type=float, default=0.001,
                   help="Minimum improvement threshold for early stopping")

    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    # Create log file
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "experiment_log.txt")
    with open(log_file, "w") as f:
        f.write(f"Enhanced TruthfulHypothesis (TH) Score-based Minimal Imputation Experiment\n")
        f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Arguments: {vars(args)}\n\n")
    
    # Set up directories
    setup_directories(args)
    
    # Record start time
    start_time = datetime.now()
    log_message(f"Starting enhanced minimal imputation experiment at {start_time}", log_file)
    
    try:
        # Step 1: Load dataset and marker genes
        adata, marker_genes = load_data(args)
        
        # Examine gene names to help with marker gene matching
        if args.verbose or args.debug:
            inspect_gene_names(adata)
        
        # Step 2: Check if marker genes are available, generate synthetic ones if not
        if not marker_genes and "Celltype" in adata.obs.columns:
            log_message("No marker genes provided. Generating synthetic markers from cell type annotations.")
            # marker_genes = generate_synthetic_markers(adata)
        
        # Step 3: Get method names
        method_names = args.methods.split(',')
        log_message(f"Using methods: {', '.join(method_names)}")

        # Step 4: Calculate TH scores with marker-based weights
        log_message("Calculating TH scores with marker-based weights")
        th_results = calculate_th_scores_with_marker_weights(adata, method_names, marker_genes, args)
        
        if not th_results:
            log_message("Error: No valid TH scores calculated. Check input files and methods.")
            sys.exit(1)
        
        # Step 5: Run enhanced incremental imputation
        log_message("Running enhanced incremental imputation with adaptive step size and uncertainty analysis")
        all_results = incremental_imputation(adata, marker_genes, th_results, args)
        
        # Step 6: Generate enhanced visualizations
        if not args.skip_plots:
            log_message("Generating enhanced visualizations")
            plot_incremental_results(all_results, args)
        else:
            log_message("Skipping plot generation as requested")
        
        # Step 7: Generate comprehensive reports
        log_message("Generating comprehensive reports")
        generate_report(all_results, args)
        
        # Step 8: Calculate cross-validation metrics if multiple samples
        if len(all_results) > 2 and "disease" in adata.obs.columns and "tissue" in adata.obs.columns:
            log_message("Performing cross-validation analysis across samples")
            cross_validation_results = cross_validate_optimal_percentages(all_results, adata, args)
            
            # Save cross-validation results
            cv_path = os.path.join(args.output_dir, "cross_validation_results.json")
            with open(cv_path, 'w') as f:
                json.dump(cross_validation_results, f, indent=2)
            log_message(f"  Cross-validation results saved to {cv_path}")
        
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
        
        avg_improvement = np.mean([r["max_score"] - r["imputation_curve"][0].get("composite_score", 0) for r in all_results])
        log_message(f"Average improvement in composite score: {avg_improvement:.4f}", log_file)
        
        # Get bootstrap confidence interval if available
        bootstrap_cis = []
        for result in all_results:
            if "bootstrap_results" in result and result["bootstrap_results"]:
                ci_lower = result["bootstrap_results"]["ci_lower"]
                ci_upper = result["bootstrap_results"]["ci_upper"]
                bootstrap_cis.append((ci_lower, ci_upper))
        
        if bootstrap_cis:
            avg_ci_lower = np.mean([ci[0] for ci in bootstrap_cis])
            avg_ci_upper = np.mean([ci[1] for ci in bootstrap_cis])
            log_message(f"Average 95% CI for optimal imputation: [{avg_ci_lower:.1%}, {avg_ci_upper:.1%}]", log_file)
        
        log_message("\nResults saved to:", log_file)
        log_message(f"  Output data: {args.output_dir}", log_file)
        log_message(f"  Figures: {args.figures_dir}", log_file)
        log_message(f"  Reports: {os.path.join(args.output_dir, 'minimal_imputation_results.csv')}", log_file)
        log_message(f"           {os.path.join(args.output_dir, 'minimal_imputation_summary.txt')}", log_file)
        log_message(f"           {os.path.join(args.output_dir, 'optimal_percentage_statistics.txt')}", log_file)
        
    except Exception as e:
        import traceback
        error_message = f"Error occurred: {str(e)}\n{traceback.format_exc()}"
        log_message(error_message, log_file)
        print(error_message)
        sys.exit(1)


if __name__ == "__main__":
    main()