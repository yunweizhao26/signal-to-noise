import os
import numpy as np
import pandas as pd
import anndata
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import time

def load_data(dataset_id):
    """Load the full dataset without filtering by disease and tissue"""
    adata = anndata.read_h5ad(f"./datasets/{dataset_id}.h5ad")
    return adata

def load_marker_genes(path="./data/marker_genes.json"):
    with open(path, 'r') as f:
        marker_genes = json.load(f)
    return marker_genes

def expand_magic_matrix(original_matrix, imputed_matrix):
    expanded = np.zeros(original_matrix.shape)
    nonzero_counts = (original_matrix != 0).sum(axis=0)
    kept_genes = np.where(nonzero_counts >= 5)[0]
    for i, original_idx in enumerate(kept_genes):
        if i < imputed_matrix.shape[1]:
            expanded[:, original_idx] = imputed_matrix[:, i]
    return expanded

def load_imputed_matrices(methods, dataset_id, disease, tissue, adata_subset):
    imputed_matrices = []
    valid_methods = []
    
    for method in methods:
        try:
            file_path = f"./output/{method}/{dataset_id}/{disease}/{tissue}.npy"
            imputed = np.load(file_path)
            
            if method == "MAGIC":
                imputed = expand_magic_matrix(adata_subset.X, imputed)
                
            imputed_matrices.append(imputed)
            valid_methods.append(method)
            print(f"  Loaded {method} matrix: {imputed.shape}")
        except Exception as e:
            print(f"  Failed to load {method}: {e}")
    
    return imputed_matrices, valid_methods

def filter_matrices_to_common_size(imputed_matrices, adata):
    min_genes = min([matrix.shape[1] for matrix in imputed_matrices])
    
    filtered_matrices = []
    for matrix in imputed_matrices:
        filtered_matrices.append(matrix[:, :min_genes])
    
    adata_filtered = adata[:, :min_genes].copy()
    
    return filtered_matrices, adata_filtered

def calculate_th_scores(adata, imputed_matrices, methods):
    is_sparse = hasattr(adata.X, 'toarray')
    
    if is_sparse:
        X_dense = adata.X.toarray()
        zero_mask = X_dense == 0
    else:
        zero_mask = adata.X == 0
        
    zero_indices = np.where(zero_mask)
    weights = {method: 1.0/len(methods) for method in methods}
    
    n_zeros = len(zero_indices[0])
    scores = np.zeros(n_zeros)
    values = np.zeros(n_zeros)
    
    # Use batch processing for large datasets to avoid memory issues
    batch_size = 1000000  # Process 1 million zeros at a time
    for start_idx in range(0, n_zeros, batch_size):
        end_idx = min(start_idx + batch_size, n_zeros)
        batch_indices = (zero_indices[0][start_idx:end_idx], zero_indices[1][start_idx:end_idx])
        
        for i, method_name in enumerate(methods):
            imputed = imputed_matrices[i]
            weight = weights[method_name]
            
            imputed_values = np.zeros(end_idx - start_idx)
            for j, (r, c) in enumerate(zip(batch_indices[0], batch_indices[1])):
                imputed_values[j] = imputed[r, c]
            
            ZERO_TOLERANCE = 0
            scores[start_idx:end_idx] += weight * (np.abs(imputed_values) >= ZERO_TOLERANCE).astype(float)
            values[start_idx:end_idx] += weight * imputed_values
    
    return {
        "zero_indices": zero_indices,
        "scores": scores,
        "values": values,
        "method_weights": weights,
        "valid_methods": methods,
    }

def run_analysis_on_subdataset(methods, dataset_id, disease, tissue, base_output_dir, marker_genes):
    """Run the complete analysis pipeline on a single disease+tissue combination"""
    import gc
    
    output_dir = os.path.join(base_output_dir, f"{disease.replace(' ', '_')}_{tissue.replace(' ', '_')}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nAnalyzing subdataset: {disease} - {tissue}")
    
    # Load dataset - only this specific subset instead of full dataset
    try:
        # Load full dataset in read mode (without context manager)
        full_adata = anndata.read_h5ad(f"./datasets/{dataset_id}.h5ad", backed='r')
        
        # Get the indices for this subset
        mask = (full_adata.obs["disease"] == disease) & (full_adata.obs["tissue"] == tissue)
        subset_indices = np.where(mask)[0]
        
        if len(subset_indices) < 50:
            print(f"  Skipping subdataset with only {len(subset_indices)} cells")
            # Clean up
            if hasattr(full_adata, 'file') and full_adata.file is not None:
                full_adata.file.close()
            del full_adata
            return None
        
        # Load only this subset into memory
        # Instead of using .copy(), first select the subset and then load it to memory
        adata_subset = full_adata[subset_indices]
        adata = adata_subset.to_memory()
        
        # Clean up the full dataset once we have the subset
        if hasattr(full_adata, 'file') and full_adata.file is not None:
            full_adata.file.close()
        del full_adata
        del adata_subset
        
    except Exception as e:
        print(f"  Error loading dataset: {e}")
        return None
    
    print(f"  Loaded subdataset with {adata.shape[0]} cells and {adata.shape[1]} genes")
    
    # Load imputed matrices
    imputed_matrices, valid_methods = load_imputed_matrices(methods, dataset_id, disease, tissue, adata)
    
    # Check if we have enough methods
    if len(valid_methods) < 2:
        print(f"  Skipping subdataset with only {len(valid_methods)} valid imputation methods")
        del adata, imputed_matrices  # Free memory
        gc.collect()
        return None
    
    # Filter matrices to common size
    filtered_matrices, adata_filtered = filter_matrices_to_common_size(imputed_matrices, adata)
    print(f"  Filtered matrices to common dimensions: {adata_filtered.shape}")
    
    # Calculate TH scores
    print(f"  Calculating TH scores...")
    th_results = calculate_th_scores(adata_filtered, filtered_matrices, valid_methods)
    scores = th_results["scores"]
    values = th_results["values"]
    zero_indices = th_results["zero_indices"]
    print(f"  Calculated TH scores for {len(scores)} zeros")
    
    # Generate consensus files
    print(f"  Generating consensus CSV files...")
    generate_consensus_csv_files(adata_filtered, scores, values, zero_indices, marker_genes, output_dir)
    
    # Load the generated CSV files
    csv_files = [
        os.path.join(output_dir, "gene_ranking_th_threshold_scores.csv"),
        os.path.join(output_dir, "gene_ranking_th_scores.csv"),
        os.path.join(output_dir, "gene_ranking_consensus_threshold_scores.csv"),
        os.path.join(output_dir, "gene_ranking_consensus_scores.csv")
    ]
    
    dfs = {}
    for file in csv_files:
        if os.path.exists(file):
            try:
                dfs[os.path.basename(file)] = pd.read_csv(file)
                print(f"  Loaded {os.path.basename(file)} successfully")
            except Exception as e:
                print(f"  Error loading {os.path.basename(file)}: {e}")
                dfs[os.path.basename(file)] = None
        else:
            print(f"  File {os.path.basename(file)} not found")
            dfs[os.path.basename(file)] = None
    
    # Check if required files are available
    if (dfs.get("gene_ranking_th_threshold_scores.csv") is None or 
        dfs.get("gene_ranking_consensus_threshold_scores.csv") is None):
        print("  Error: Required CSV files are missing.")
        
        # Clean up
        del adata, adata_filtered, imputed_matrices, filtered_matrices, th_results
        gc.collect()
        
        return None
    
    # Merge TH and consensus scores
    comparison_df = merge_th_consensus_scores(
        dfs["gene_ranking_th_threshold_scores.csv"],
        dfs["gene_ranking_consensus_threshold_scores.csv"]
    )
    
    if comparison_df.empty:
        print("  No marker genes found for comparison")
        
        # Clean up
        del adata, adata_filtered, imputed_matrices, filtered_matrices, th_results
        gc.collect()
        
        return None
    
    print(f"  Created comparison dataframe with {len(comparison_df)} gene/cell type combinations")
    
    # Create visualizations only if we have significant improvements
    if len(comparison_df) >= 5 and (comparison_df['f1_improvement'] > 0).mean() > 0.5:
        print(f"  Generating key visualizations...")
        
        # Top genes comparison
        plot_f1_comparison(
            comparison_df, 
            top_n=min(10, len(comparison_df)), 
            save_path=os.path.join(output_dir, "top_genes_comparison.png")
        )
        
        # Cell type comparison if we have multiple cell types
        if len(comparison_df['cell_type'].unique()) > 1:
            plot_cell_type_comparison(
                comparison_df,
                save_path=os.path.join(output_dir, "cell_type_comparison.png")
            )
    
    # Save scores and values statistics
    score_stats = {
        'min': float(np.min(scores)),
        'max': float(np.max(scores)),
        'mean': float(np.mean(scores)),
        'median': float(np.median(scores)),
        'std': float(np.std(scores))
    }
    
    value_stats = {
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'std': float(np.std(values))
    }
    
    with open(os.path.join(output_dir, "th_score_statistics.json"), 'w') as f:
        json.dump(score_stats, f, indent=4)
    
    with open(os.path.join(output_dir, "consensus_value_statistics.json"), 'w') as f:
        json.dump(value_stats, f, indent=4)
    
    # Create network inference matrices with reduced memory usage
    thresholds = [0.0, 0.3, 0.7, 1.0]  # Reduced set of thresholds to save memory and storage
    create_network_inference_analysis(
        adata=adata_filtered, 
        scores=scores, 
        values=values, 
        zero_indices=zero_indices,
        output_dir=os.path.join(output_dir, "grn_analysis"),
        thresholds=thresholds
    )
    
    # Create result object with essential info
    result = {
        'disease': disease,
        'tissue': tissue,
        'cell_count': adata.shape[0],
        'gene_count': adata.shape[1],
        'comparison_df': comparison_df,
        'score_stats': score_stats,
        'value_stats': value_stats,
        'zero_count': len(scores)
    }
    
    print(f"  Analysis complete for {disease} - {tissue}")
    
    # Free memory before returning
    del adata, adata_filtered, imputed_matrices, filtered_matrices, th_results
    gc.collect()
    
    return result

def generate_consensus_csv_files(adata, scores, values, zero_indices, marker_genes, output_dir):
    """Generate the CSV files needed for consensus analysis directly from the TH scores and values"""
    # First, find marker genes in the dataset
    marker_gene_indices = {}
    
    for cell_type, genes in marker_genes.items():
        marker_gene_indices[cell_type] = []
        for gene in genes:
            if gene in adata.var_names:
                gene_idx = adata.var_names.get_loc(gene)
                marker_gene_indices[cell_type].append((gene, gene_idx))

    # Prepare to store results for all cell types and genes
    th_results = []
    consensus_results = []
    
    # For each cell type and marker gene
    for cell_type, markers in marker_gene_indices.items():
        # Skip if no markers found
        if not markers:
            continue
            
        # Find cells of this type
        if "Celltype" not in adata.obs.columns:
            continue
            
        cell_mask = adata.obs["Celltype"] == cell_type
        if np.sum(cell_mask) == 0:
            continue
            
        cell_indices = np.where(cell_mask)[0]
        
        # For each marker gene
        for gene, gene_idx in markers:
            # Find all zero entries for this gene
            col_mask = zero_indices[1] == gene_idx
            if np.sum(col_mask) == 0:
                continue
                
            # Get positions in the scores and values arrays
            gene_positions = np.where(col_mask)[0]
            cell_row_indices = zero_indices[0][gene_positions]
            
            # Create binary labels for these cells (1 for target cell type, 0 for others)
            target_mask = np.isin(cell_row_indices, cell_indices)
            
            # Skip if no positive examples
            if np.sum(target_mask) == 0:
                continue
                
            # Get scores and values for these positions
            gene_scores = scores[gene_positions]
            gene_values = values[gene_positions]
            
            # Try different thresholds for TH scores and consensus values
            best_th_threshold = find_best_threshold(gene_scores, target_mask)
            best_consensus_threshold = find_best_threshold(gene_values, target_mask)
            
            # Store results
            th_results.append({
                'cell_type': cell_type,
                'gene': gene,
                'f1': best_th_threshold['f1'],
                'precision': best_th_threshold['precision'],
                'recall': best_th_threshold['recall'],
                'cutoff': best_th_threshold['cutoff'],
                'rank': 1  # We're only keeping the best threshold
            })
            
            consensus_results.append({
                'cell_type': cell_type,
                'gene': gene,
                'f1': best_consensus_threshold['f1'],
                'precision': best_consensus_threshold['precision'],
                'recall': best_consensus_threshold['recall'],
                'cutoff': best_consensus_threshold['cutoff'],
                'rank': 1  # We're only keeping the best threshold
            })
    
    # Convert to DataFrames and save
    if th_results:
        th_df = pd.DataFrame(th_results)
        th_df.to_csv(os.path.join(output_dir, "gene_ranking_th_threshold_scores.csv"), index=False)
        
        # Also create the aggregated file
        th_agg = pd.DataFrame({
            'cell_type': th_df['cell_type'],
            'gene': th_df['gene'],
            'consensus_cutoff': th_df['cutoff'],
            'f1': th_df['f1'],
            'precision': th_df['precision'],
            'recall': th_df['recall'],
            'rank': th_df['rank']
        })
        th_agg.to_csv(os.path.join(output_dir, "gene_ranking_th_scores.csv"), index=False)
    
    if consensus_results:
        consensus_df = pd.DataFrame(consensus_results)
        consensus_df.to_csv(os.path.join(output_dir, "gene_ranking_consensus_threshold_scores.csv"), index=False)
        
        # Also create the aggregated file
        consensus_agg = pd.DataFrame({
            'cell_type': consensus_df['cell_type'],
            'gene': consensus_df['gene'],
            'consensus_cutoff': consensus_df['cutoff'],
            'f1': consensus_df['f1'],
            'precision': consensus_df['precision'],
            'recall': consensus_df['recall'],
            'rank': consensus_df['rank']
        })
        consensus_agg.to_csv(os.path.join(output_dir, "gene_ranking_consensus_scores.csv"), index=False)

def find_best_threshold(values, labels):
    """Find the best threshold value based on F1 score"""
    best_f1 = 0
    best_cutoff = None
    best_precision = 0
    best_recall = 0
    
    # Try different percentiles to find the best cutoff
    for percentile in np.arange(50, 99.5, 2.5):
        cutoff = np.percentile(values, percentile)
        predictions = (values >= cutoff).astype(int)
        
        # Calculate metrics
        tp = np.sum(predictions & labels)
        fp = np.sum(predictions & ~labels)
        fn = np.sum(~predictions & labels)
        
        if tp + fp > 0 and tp + fn > 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_cutoff = cutoff
                    best_precision = precision
                    best_recall = recall
    
    # If no good cutoff found, use a default
    if best_cutoff is None:
        best_cutoff = np.median(values)
        predictions = (values >= best_cutoff).astype(int)
        
        tp = np.sum(predictions & labels)
        fp = np.sum(predictions & ~labels)
        fn = np.sum(~predictions & labels)
        
        if tp + fp > 0 and tp + fn > 0:
            best_precision = tp / (tp + fp)
            best_recall = tp / (tp + fn)
            best_f1 = 2 * best_precision * best_recall / (best_precision + best_recall) if (best_precision + best_recall) > 0 else 0
        else:
            best_precision = 0
            best_recall = 0
            best_f1 = 0
    
    return {
        'cutoff': float(best_cutoff),
        'f1': float(best_f1),
        'precision': float(best_precision),
        'recall': float(best_recall)
    }

def plot_cell_type_comparison(comparison_df, save_path=None):
    """Create a comparison plot showing F1 scores by cell type"""
    if comparison_df.empty:
        print("  No data to plot")
        return
    
    # Group by cell type and calculate mean F1 scores
    cell_type_avg = comparison_df.groupby('cell_type').agg({
        'th_f1': 'mean',
        'cons_f1': 'mean',
        'gene': 'count'
    }).rename(columns={'gene': 'num_genes'})
    
    # Calculate improvement
    cell_type_avg['improvement'] = cell_type_avg['cons_f1'] - cell_type_avg['th_f1']
    
    # Sort by improvement
    cell_type_avg = cell_type_avg.sort_values('improvement', ascending=False)
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    # Convert to long format for seaborn
    plot_df = pd.DataFrame({
        'cell_type': cell_type_avg.index,
        'TH Score F1': cell_type_avg['th_f1'],
        'Consensus F1': cell_type_avg['cons_f1']
    })
    plot_df = pd.melt(plot_df, id_vars=['cell_type'], var_name='Method', value_name='Average F1 Score')
    
    # Create plot
    custom_palette = {"TH Score F1": "#6baed6", "Consensus F1": "#fd8d3c"}
    ax = sns.barplot(x='cell_type', y='Average F1 Score', hue='Method', data=plot_df, palette=custom_palette)
    
    # Add labels
    plt.title('Average F1 Score by Cell Type', fontsize=14)
    plt.xlabel('Cell Type', fontsize=12)
    plt.ylabel('Average F1 Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add counts to labels
    labels = [f"{ct} (n={cell_type_avg.loc[ct, 'num_genes']})" for ct in cell_type_avg.index]
    ax.set_xticklabels(labels)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_network_inference_analysis(adata, scores, values, zero_indices, output_dir, thresholds=[0.0, 0.3, 0.7, 1.0]):
    """Create analysis files for network inference"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a simple summary file
    summary = {
        'num_cells': adata.shape[0],
        'num_genes': adata.shape[1],
        'num_zeros': len(scores),
        'score_thresholds': thresholds,
        'score_percentiles': [np.percentile(scores, p) for p in range(0, 101, 10)],
        'value_percentiles': [float(np.percentile(values, p)) for p in range(0, 101, 10)]
    }
    
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Save data for a sample of genes for visualization (to avoid massive files)
    if adata.shape[1] > 100:
        sample_genes = np.random.choice(adata.shape[1], 100, replace=False)
    else:
        sample_genes = np.arange(adata.shape[1])
    
    # For each threshold, create a sample network
    for threshold in thresholds:
        # Create a sample adjacency matrix
        sample_matrix = np.zeros((len(sample_genes), len(sample_genes)))
        
        for i, gene1 in enumerate(sample_genes):
            for j, gene2 in enumerate(sample_genes):
                if i != j:
                    # Find zero entries for this gene pair
                    mask1 = (zero_indices[1] == gene1) & (zero_indices[0] == gene2)
                    mask2 = (zero_indices[1] == gene2) & (zero_indices[0] == gene1)
                    
                    if np.sum(mask1) > 0 or np.sum(mask2) > 0:
                        # Average scores for the gene pair
                        avg_score = 0
                        count = 0
                        
                        if np.sum(mask1) > 0:
                            avg_score += np.mean(scores[mask1])
                            count += 1
                        
                        if np.sum(mask2) > 0:
                            avg_score += np.mean(scores[mask2])
                            count += 1
                        
                        if count > 0:
                            avg_score /= count
                            
                        # Only include if score exceeds threshold
                        if avg_score >= threshold:
                            sample_matrix[i, j] = avg_score
        
        # Save the sample matrix
        np.save(os.path.join(output_dir, f"sample_network_th{threshold}.npy"), sample_matrix)
        
        # Create a simple edge list for visualization
        edges = []
        for i, gene1 in enumerate(sample_genes):
            for j, gene2 in enumerate(sample_genes):
                if i < j and sample_matrix[i, j] > 0:  # Only upper triangle to avoid duplicates
                    edges.append({
                        'source': adata.var_names[gene1],
                        'target': adata.var_names[gene2],
                        'weight': float(sample_matrix[i, j])
                    })
        
        # Save edge list if we have any edges
        if edges:
            with open(os.path.join(output_dir, f"sample_edges_th{threshold}.json"), 'w') as f:
                json.dump(edges, f, indent=4)


def plot_f1_comparison(comparison_df, top_n=10, save_path=None):
    """Plot comparison of TH and consensus F1 scores for top genes"""
    if comparison_df.empty:
        print("  No data to plot")
        return
    
    # Sort by improvement and get top N
    top_genes = comparison_df.sort_values('f1_improvement', ascending=False).head(top_n)
    
    plt.figure(figsize=(12, 8))
    
    labels = [f"{row['gene']} ({row['cell_type']})" for _, row in top_genes.iterrows()]
    x = np.arange(len(labels))
    width = 0.35
    
    ax = plt.subplot(111)
    th_bars = ax.bar(x - width/2, top_genes['th_f1'], width, label='TH Score F1', color='#6baed6')
    cons_bars = ax.bar(x + width/2, top_genes['cons_f1'], width, label='Consensus F1', color='#fd8d3c')
    
    # Add improvement text
    for i, (_, row) in enumerate(top_genes.iterrows()):
        plt.text(i, max(row['th_f1'], row['cons_f1']) + 0.02, 
                 f"+{row['f1_improvement']:.3f}", 
                 ha='center', va='bottom', 
                 fontsize=9)
    
    plt.xlabel('Gene (Cell Type)', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Top Genes by F1 Score Improvement', fontsize=14)
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def merge_th_consensus_scores(th_df, consensus_df):
    """Merge and compare TH and consensus scores"""
    if th_df is None or consensus_df is None or th_df.empty or consensus_df.empty:
        return pd.DataFrame()
    
    # Merge the dataframes on gene and cell_type
    merged_df = pd.merge(th_df, consensus_df, 
                          on=['gene', 'cell_type'], 
                          suffixes=('_th', '_cons'))
    
    # Calculate improvement
    merged_df['f1_improvement'] = merged_df['f1_cons'] - merged_df['f1_th']
    
    # Create a simplified dataframe with just the key metrics
    result_df = pd.DataFrame({
        'cell_type': merged_df['cell_type'],
        'gene': merged_df['gene'],
        'th_f1': merged_df['f1_th'],
        'cons_f1': merged_df['f1_cons'],
        'f1_improvement': merged_df['f1_improvement'],
        'th_precision': merged_df['precision_th'],
        'cons_precision': merged_df['precision_cons'],
        'th_recall': merged_df['recall_th'],
        'cons_recall': merged_df['recall_cons']
    })
    
    return result_df


def aggregate_results(all_results, output_dir):
    """Aggregate results across all subdatasets"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all comparison DataFrames
    all_comparisons = []
    subdataset_info = []
    
    for result in all_results:
        if result is None or result.get('comparison_df') is None:
            continue
            
        comparison_df = result['comparison_df']
        comparison_df['disease'] = result['disease']
        comparison_df['tissue'] = result['tissue']
        all_comparisons.append(comparison_df)
        
        subdataset_info.append({
            'disease': result['disease'],
            'tissue': result['tissue'],
            'cell_count': result['cell_count'],
            'gene_count': result['gene_count'],
            'zero_count': result['zero_count'],
            'marker_gene_count': len(comparison_df),
            'improved_percentage': (comparison_df['f1_improvement'] > 0).mean() * 100,
            'average_improvement': comparison_df['f1_improvement'].mean(),
            'max_improvement': comparison_df['f1_improvement'].max()
        })
    
    # Create a single DataFrame with all comparisons
    if all_comparisons:
        combined_df = pd.concat(all_comparisons, ignore_index=True)
        combined_df.to_csv(os.path.join(output_dir, "all_subdatasets_comparison.csv"), index=False)
        
        # Create a summary DataFrame
        subdataset_summary = pd.DataFrame(subdataset_info)
        subdataset_summary.to_csv(os.path.join(output_dir, "subdataset_summary.csv"), index=False)
        
        # Calculate overall statistics
        overall_stats = {
            'total_subdatasets': len(subdataset_info),
            'total_markers': len(combined_df),
            'improved_markers': int((combined_df['f1_improvement'] > 0).sum()),
            'improved_percentage': float((combined_df['f1_improvement'] > 0).mean() * 100),
            'average_improvement': float(combined_df['f1_improvement'].mean()),
            'median_improvement': float(combined_df['f1_improvement'].median()),
            'max_improvement': float(combined_df['f1_improvement'].max()),
            'min_improvement': float(combined_df['f1_improvement'].min())
        }
        
        with open(os.path.join(output_dir, "overall_statistics.json"), 'w') as f:
            json.dump(overall_stats, f, indent=4)
        
        # Create visualizations
        create_aggregated_visualizations(combined_df, subdataset_summary, output_dir)
        
        return combined_df, subdataset_summary, overall_stats
    else:
        print("No valid results to aggregate")
        return None, None, None

def create_aggregated_visualizations(combined_df, subdataset_summary, output_dir):
    """Create visualizations for the aggregated results"""
    # 1. Overall improvement distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(combined_df['f1_improvement'], kde=True, color='#fd8d3c')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    improved_pct = (combined_df['f1_improvement'] > 0).mean() * 100
    plt.title(f'Distribution of F1 Score Improvement Across All Subdatasets\n{improved_pct:.1f}% of genes improved with consensus approach', fontsize=16)
    plt.xlabel('F1 Score Improvement (Consensus - TH)', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "overall_improvement_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Scatter plot comparing TH and Consensus F1 scores
    plt.figure(figsize=(10, 10))
    plt.scatter(
        combined_df['th_f1'], 
        combined_df['cons_f1'],
        alpha=0.6, 
        c=combined_df['f1_improvement'], 
        cmap='viridis',
        s=50
    )
    max_val = max(combined_df['th_f1'].max(), combined_df['cons_f1'].max()) * 1.1
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
    cbar = plt.colorbar()
    cbar.set_label('F1 Improvement', fontsize=12)
    plt.title('TH Score F1 vs Consensus F1 Across All Subdatasets', fontsize=16)
    plt.xlabel('TH Score F1', fontsize=14)
    plt.ylabel('Consensus F1', fontsize=14)
    plt.grid(alpha=0.3)
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.savefig(os.path.join(output_dir, "overall_scatter_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Top cell types across all subdatasets
    cell_type_avg = combined_df.groupby('cell_type').agg({
        'th_f1': 'mean',
        'cons_f1': 'mean',
        'gene': 'count'
    }).rename(columns={'gene': 'num_genes'})
    cell_type_avg['improvement'] = cell_type_avg['cons_f1'] - cell_type_avg['th_f1']
    cell_type_avg = cell_type_avg.sort_values('improvement', ascending=False)
    
    # Filter to cell types with at least 3 genes
    cell_type_filtered = cell_type_avg[cell_type_avg['num_genes'] >= 3].head(15)
    
    if not cell_type_filtered.empty:
        plot_df = pd.DataFrame({
            'cell_type': cell_type_filtered.index,
            'TH Score F1': cell_type_filtered['th_f1'],
            'Consensus F1': cell_type_filtered['cons_f1']
        })
        plot_df = pd.melt(plot_df, id_vars=['cell_type'], var_name='Method', value_name='Average F1 Score')
        plt.figure(figsize=(14, 8))
        custom_palette = {"TH Score F1": "#6baed6", "Consensus F1": "#fd8d3c"}
        ax = sns.barplot(x='cell_type', y='Average F1 Score', hue='Method', data=plot_df, palette=custom_palette)
        plt.title('Average F1 Score by Cell Type Across All Subdatasets', fontsize=16)
        plt.xlabel('Cell Type', fontsize=14)
        plt.ylabel('Average F1 Score', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        labels = [f"{ct} (n={cell_type_filtered.loc[ct, 'num_genes']})" for ct in cell_type_filtered.index]
        ax.set_xticklabels(labels)
        plt.tight_layout()
        for i, bar in enumerate(ax.patches):
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.01,
                    f'{height:.3f}',
                    ha='center', va='bottom',
                    fontsize=9
                )
        plt.savefig(os.path.join(output_dir, "overall_cell_type_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Top genes across all subdatasets
    top_genes = combined_df.sort_values('f1_improvement', ascending=False).head(20)
    top_genes_table = top_genes[['cell_type', 'gene', 'disease', 'tissue', 'th_f1', 'cons_f1', 'f1_improvement']]
    top_genes_table.to_csv(os.path.join(output_dir, "top_improved_genes.csv"), index=False)
    
    # 5. Plot comparison by subdataset size
    plt.figure(figsize=(12, 6))
    plt.scatter(
        subdataset_summary['cell_count'],
        subdataset_summary['improved_percentage'],
        s=subdataset_summary['marker_gene_count'] * 5,  # Size proportional to number of marker genes
        alpha=0.7,
        c=subdataset_summary['average_improvement'],
        cmap='viridis'
    )
    plt.colorbar(label='Average F1 Improvement')
    plt.xlabel('Number of Cells in Subdataset', fontsize=14)
    plt.ylabel('Percentage of Improved Marker Genes (%)', fontsize=14)
    plt.title('Consensus Improvement by Subdataset Size', fontsize=16)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "improvement_by_subdataset_size.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Heatmap of top subdatasets by improvement
    top_subdatasets = subdataset_summary.sort_values('average_improvement', ascending=False).head(15)
    if len(top_subdatasets) > 1:
        plt.figure(figsize=(12, 8))
        heatmap_data = top_subdatasets.set_index(['disease', 'tissue'])[['improved_percentage', 'average_improvement', 'max_improvement']]
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='viridis')
        plt.title('Top Subdatasets by Average F1 Improvement', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "top_subdatasets_heatmap.png"), dpi=300, bbox_inches='tight')
        plt.close()

def run_all_subdatasets(dataset_id, methods, base_output_dir):
    """Run analysis on all available disease+tissue combinations"""
    import gc
    
    print(f"Starting analysis on all subdatasets for dataset {dataset_id}")
    
    # Step 1: First just collect metadata without loading full dataset
    print("Collecting metadata and planning analysis...")
    subdataset_plan = []
    
    # Load just the metadata to identify subdatasets
    adata_file = f"./datasets/{dataset_id}.h5ad"
    print(f"Reading metadata from {adata_file}")
    adata = anndata.read_h5ad(adata_file, backed='r')
    metadata_df = pd.DataFrame({
        'disease': adata.obs['disease'],
        'tissue': adata.obs['tissue']
    })

    # Get all disease+tissue combinations
    metadata_markers = metadata_df.drop_duplicates(["disease", "tissue"])
    print(f"Found {len(metadata_markers)} disease+tissue combinations")

    # Prepare analysis plan
    for _, (disease, tissue) in metadata_markers.iterrows():
        # Check if imputation files exist
        has_imputation = False
        for method in methods:
            file_path = f"./output/{method}/{dataset_id}/{disease}/{tissue}.npy"
            if os.path.exists(file_path):
                has_imputation = True
                break
        if not has_imputation:
                continue
        # Count cells
        mask = (metadata_df["disease"] == disease) & (metadata_df["tissue"] == tissue)
        cell_count = mask.sum()
        
        if cell_count >= 50:
            subdataset_plan.append({
                'disease': disease,
                'tissue': tissue,
                'cell_count': cell_count
            })
        else:
            print(f"Skipping {disease} - {tissue} with only {cell_count} cells")
    
    # Clean up and force garbage collection
    del metadata_df
    gc.collect()

    # Load marker genes (small enough to keep in memory)
    marker_genes = load_marker_genes()
    print(f"Loaded {sum(len(genes) for genes in marker_genes.values())} marker genes for {len(marker_genes)} cell types")
    
    # Create base output directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Step 2: Process each subdataset individually to control memory usage
    print(f"Starting analysis of {len(subdataset_plan)} valid subdatasets...")
    all_results = []
    
    for i, subdataset in enumerate(tqdm(subdataset_plan, desc="Processing subdatasets")):
        disease = subdataset['disease']
        tissue = subdataset['tissue']
        
        print(f"\n[{i+1}/{len(subdataset_plan)}] Processing: {disease} - {tissue}")
        
        # Run analysis on this combination
        try:
            result = run_analysis_on_subdataset(methods, dataset_id, disease, tissue, base_output_dir, marker_genes)
            if result is not None:
                # Only keep essential info, not the full data
                essential_result = {
                    'disease': disease,
                    'tissue': tissue,
                    'cell_count': subdataset['cell_count'],
                    'gene_count': result.get('gene_count', 0),
                    'zero_count': result.get('zero_count', 0),
                    'comparison_df': result.get('comparison_df')
                }
                all_results.append(essential_result)
            
            # Force cleanup after each subdataset
            gc.collect()
            
            # Optional: periodically save intermediate results
            if (i+1) % 5 == 0:
                print(f"Saving intermediate results after {i+1} subdatasets")
                tmp_output_dir = os.path.join(base_output_dir, "intermediate")
                os.makedirs(tmp_output_dir, exist_ok=True)
                
                # Save current results to avoid losing everything if process fails later
                with open(os.path.join(tmp_output_dir, f"results_after_{i+1}_subdatasets.json"), 'w') as f:
                    # Save only what can be serialized
                    summary_results = []
                    for res in all_results:
                        if res.get('comparison_df') is not None:
                            # Convert DataFrames to lists for JSON serialization
                            df_records = res['comparison_df'].to_dict('records')
                            res_copy = res.copy()
                            res_copy['comparison_df'] = df_records
                            summary_results.append(res_copy)
                    
                    json.dump(summary_results, f)
                
        except Exception as e:
            print(f"Error processing {disease} - {tissue}: {e}")
            # Continue with next subdataset even if this one fails
            continue
    
    # Step 3: Aggregate results
    print(f"\nAggregating results from {len(all_results)} subdatasets")
    combined_df, subdataset_summary, overall_stats = aggregate_results(all_results, os.path.join(base_output_dir, "aggregate"))
    
    if overall_stats:
        print("\nOverall Statistics:")
        for key, value in overall_stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    
    # Final cleanup
    gc.collect()
    
    print(f"\nAnalysis complete. Results saved to {base_output_dir}")

def main():
    output_dir = "./all_subdatasets_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting comprehensive scRNA-seq imputation consensus analysis across all subdatasets...")
    
    methods = ["SAUCIE", "MAGIC", "deepImpute", "scScope", "scVI", "knn_smoothing"]
    dataset_id = "63ff2c52-cb63-44f0-bac3-d0b33373e312"
    
    # Run the full analysis
    run_all_subdatasets(dataset_id, methods, output_dir)

if __name__ == "__main__":
    # Track execution time
    start_time = time.time()
    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")