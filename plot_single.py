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


def load_data(dataset_id, disease, tissue):
    adata = anndata.read_h5ad(f"./datasets/{dataset_id}.h5ad")
    mask = (adata.obs["disease"] == disease) & (adata.obs["tissue"] == tissue)
    adata = adata[mask]
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


def load_imputed_matrices(methods, dataset_id, disease, tissue, adata):
    imputed_matrices = []
    valid_methods = []
    
    for method in methods:
        try:
            file_path = f"./output/{method}/{dataset_id}/{disease}/{tissue}.npy"
            imputed = np.load(file_path)
            
            if method == "MAGIC":
                imputed = expand_magic_matrix(adata.X, imputed)
                
            imputed_matrices.append(imputed)
            valid_methods.append(method)
            print(f"Loaded {method} matrix: {imputed.shape}")
        except:
            print(f"Failed to load {method}")
    
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
    
    for i, method_name in enumerate(methods):
        imputed = imputed_matrices[i]
        weight = weights[method_name]
        
        imputed_values = np.zeros(n_zeros)
        for j, (r, c) in enumerate(zip(zero_indices[0], zero_indices[1])):
            imputed_values[j] = imputed[r, c]
        
        ZERO_TOLERANCE = 0
        scores += weight * (np.abs(imputed_values) >= ZERO_TOLERANCE).astype(float)
        values += weight * imputed_values
    
    return {
        "zero_indices": zero_indices,
        "scores": scores,
        "values": values,
        "method_weights": weights,
        "valid_methods": methods,
    }


def load_data_files(csv_files_list):
    data_dict = {}
    for file in csv_files_list:
        path = Path(file)
        if path.exists():
            try:
                data_dict[file] = pd.read_csv(file)
                print(f"Loaded {file} successfully")
            except Exception as e:
                print(f"Error loading {file}: {e}")
                data_dict[file] = None
        else:
            print(f"File {file} not found")
            data_dict[file] = None
    return data_dict


def merge_th_consensus_scores(th_df, consensus_df):
    th_best = th_df[th_df['rank'] == 1].copy()
    cons_best = consensus_df[consensus_df['rank'] == 1].copy()
    
    th_best = th_best.rename(columns={
        'f1': 'th_f1',
        'precision': 'th_precision',
        'recall': 'th_recall',
        'cutoff': 'th_cutoff'
    })
    
    cons_best = cons_best.rename(columns={
        'f1': 'cons_f1',
        'precision': 'cons_precision',
        'recall': 'cons_recall',
        'cutoff': 'cons_cutoff'
    })
    
    merged = pd.merge(
        th_best[['cell_type', 'gene', 'th_f1', 'th_precision', 'th_recall', 'th_cutoff']], 
        cons_best[['cell_type', 'gene', 'cons_f1', 'cons_precision', 'cons_recall', 'cons_cutoff']],
        on=['cell_type', 'gene'],
        how='inner'
    )
    
    merged['f1_improvement'] = merged['cons_f1'] - merged['th_f1']
    merged['relative_improvement'] = merged['f1_improvement'] / (merged['th_f1'] + 1e-10)
    
    return merged


def plot_f1_comparison(comparison_df, top_n=10, save_path=None):
    top_genes = comparison_df.sort_values('f1_improvement', ascending=False).head(top_n)
    
    plot_df = pd.DataFrame({
        'gene_cell': [f"{row['gene']} ({row['cell_type']})" for _, row in top_genes.iterrows()],
        'TH Score F1': top_genes['th_f1'],
        'Consensus F1': top_genes['cons_f1']
    })
    
    plot_df = pd.melt(plot_df, id_vars=['gene_cell'], var_name='Method', value_name='F1 Score')
    
    plt.figure(figsize=(12, 8))
    custom_palette = {"TH Score F1": "#6baed6", "Consensus F1": "#fd8d3c"}
    
    ax = sns.barplot(x='gene_cell', y='F1 Score', hue='Method', data=plot_df, palette=custom_palette)
    
    plt.title(f'Top {top_n} Genes with Highest F1 Improvement: Consensus vs TH Score', fontsize=16)
    plt.xlabel('Gene (Cell Type)', fontsize=14)
    plt.ylabel('F1 Score', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
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
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.close()


def plot_cell_type_comparison(comparison_df, save_path=None):
    cell_type_avg = comparison_df.groupby('cell_type').agg({
        'th_f1': 'mean',
        'cons_f1': 'mean',
        'gene': 'count'
    }).rename(columns={'gene': 'num_genes'})
    
    cell_type_avg['improvement'] = cell_type_avg['cons_f1'] - cell_type_avg['th_f1']
    cell_type_avg = cell_type_avg.sort_values('improvement', ascending=False)
    
    plot_df = pd.DataFrame({
        'cell_type': cell_type_avg.index,
        'TH Score F1': cell_type_avg['th_f1'],
        'Consensus F1': cell_type_avg['cons_f1']
    })
    
    plot_df = pd.melt(plot_df, id_vars=['cell_type'], var_name='Method', value_name='Average F1 Score')
    
    plt.figure(figsize=(14, 8))
    custom_palette = {"TH Score F1": "#6baed6", "Consensus F1": "#fd8d3c"}
    
    ax = sns.barplot(x='cell_type', y='Average F1 Score', hue='Method', data=plot_df, palette=custom_palette)
    
    plt.title('Average F1 Score by Cell Type: Consensus vs TH Score', fontsize=16)
    plt.xlabel('Cell Type', fontsize=14)
    plt.ylabel('Average F1 Score', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
    labels = [f"{ct} (n={cell_type_avg.loc[ct, 'num_genes']})" for ct in cell_type_avg.index]
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
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.close()


def plot_improvement_distribution(comparison_df, save_path=None):
    plt.figure(figsize=(10, 6))
    
    sns.histplot(comparison_df['f1_improvement'], kde=True, color='#fd8d3c')
    
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    improved_pct = (comparison_df['f1_improvement'] > 0).mean() * 100
    
    plt.title(f'Distribution of F1 Score Improvement\n{improved_pct:.1f}% of genes improved with consensus approach', fontsize=16)
    plt.xlabel('F1 Score Improvement (Consensus - TH)', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.close()


def plot_scatter_comparison(comparison_df, save_path=None):
    plt.figure(figsize=(10, 10))
    
    plt.scatter(
        comparison_df['th_f1'], 
        comparison_df['cons_f1'],
        alpha=0.6, 
        c=comparison_df['f1_improvement'], 
        cmap='viridis',
        s=50
    )
    
    max_val = max(comparison_df['th_f1'].max(), comparison_df['cons_f1'].max()) * 1.1
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
    
    cbar = plt.colorbar()
    cbar.set_label('F1 Improvement', fontsize=12)
    
    plt.title('TH Score F1 vs Consensus F1', fontsize=16)
    plt.xlabel('TH Score F1', fontsize=14)
    plt.ylabel('Consensus F1', fontsize=14)
    plt.grid(alpha=0.3)
    
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.close()


def generate_summary_tables(comparison_df, output_path=None):
    top_genes = comparison_df.sort_values('f1_improvement', ascending=False).head(20)
    top_genes_table = top_genes[['cell_type', 'gene', 'th_f1', 'cons_f1', 'f1_improvement', 'relative_improvement']]
    
    cell_type_summary = comparison_df.groupby('cell_type').agg({
        'th_f1': 'mean',
        'cons_f1': 'mean',
        'f1_improvement': 'mean',
        'relative_improvement': 'mean',
        'gene': 'count'
    }).rename(columns={'gene': 'num_genes'}).sort_values('f1_improvement', ascending=False)
    
    overall_stats = {
        'total_genes': int(len(comparison_df)),  # Convert to int
        'improved_genes': int((comparison_df['f1_improvement'] > 0).sum()),  # Convert to int
        'improved_percentage': float((comparison_df['f1_improvement'] > 0).mean() * 100),
        'average_improvement': comparison_df['f1_improvement'].mean(),
        'median_improvement': comparison_df['f1_improvement'].median(),
        'max_improvement': comparison_df['f1_improvement'].max(),
        'gene_with_max_improvement': comparison_df.loc[comparison_df['f1_improvement'].idxmax(), 'gene'],
        'cell_type_with_max_improvement': comparison_df.loc[comparison_df['f1_improvement'].idxmax(), 'cell_type']
    }
    
    print("Top 20 Genes with Highest F1 Improvement:")
    print(top_genes_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nCell Type Summary:")
    print(cell_type_summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nOverall Statistics:")
    for key, value in overall_stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        
        top_genes_table.to_csv(os.path.join(output_path, "top_improved_genes.csv"), index=False)
        cell_type_summary.to_csv(os.path.join(output_path, "cell_type_summary.csv"))
        
        with open(os.path.join(output_path, "overall_statistics.json"), 'w') as f:
            json.dump(overall_stats, f, indent=4)
        
        print(f"Summary tables saved to {output_path}")


def run_consensus_analysis(output_dir="./figures"):
    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = [
        "gene_ranking_th_threshold_scores.csv",
        "gene_ranking_th_scores.csv",
        "gene_ranking_consensus_threshold_scores.csv",
        "gene_ranking_consensus_scores.csv"
    ]
    
    print("Loading data files...")
    dfs = load_data_files(csv_files)
    
    if (dfs["gene_ranking_th_threshold_scores.csv"] is None or 
        dfs["gene_ranking_consensus_threshold_scores.csv"] is None):
        print("Error: Required CSV files are missing.")
        return
    
    print("Merging TH and consensus scores...")
    comparison_df = merge_th_consensus_scores(
        dfs["gene_ranking_th_threshold_scores.csv"],
        dfs["gene_ranking_consensus_threshold_scores.csv"]
    )
    
    print(f"Created comparison dataframe with {len(comparison_df)} gene/cell type combinations")
    
    print("\nGenerating summary tables...")
    generate_summary_tables(comparison_df, os.path.join(output_dir, "tables"))
    
    print("\nCreating visualizations...")
    
    print("Plotting top genes comparison...")
    plot_f1_comparison(
        comparison_df, 
        top_n=5, 
        save_path=os.path.join(output_dir, "top_genes_comparison.png")
    )
    
    print("Plotting cell type comparison...")
    plot_cell_type_comparison(
        comparison_df,
        save_path=os.path.join(output_dir, "cell_type_comparison.png")
    )
    
    print("Plotting improvement distribution...")
    plot_improvement_distribution(
        comparison_df,
        save_path=os.path.join(output_dir, "improvement_distribution.png")
    )
    
    print("Plotting scatter comparison...")
    plot_scatter_comparison(
        comparison_df,
        save_path=os.path.join(output_dir, "scatter_comparison.png")
    )
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")
    
    return comparison_df


def analyze_marker_gene(adata, scores, values, zero_indices, marker_gene, cell_type, save_path=None):
    if marker_gene not in adata.var_names:
        print(f"Gene {marker_gene} not found in dataset")
        return
    
    gene_idx = adata.var_names.get_loc(marker_gene)
    
    col_mask = zero_indices[1] == gene_idx
    if np.sum(col_mask) == 0:
        print(f"No zero entries found for gene {marker_gene}")
        return
    
    gene_positions = np.where(col_mask)[0]
    cell_indices = zero_indices[0][gene_positions]
    
    if "Celltype" not in adata.obs.columns:
        print("Cell type information not available")
        return
    
    cell_types = adata.obs["Celltype"].iloc[cell_indices].values
    target_mask = cell_types == cell_type
    
    th_scores = scores[gene_positions]
    cons_values = values[gene_positions]
    
    plot_df_th = pd.DataFrame({
        'TH Score': th_scores,
        'Cell Type': ['Target' if m else 'Other' for m in target_mask]
    })
    
    plot_df_cons = pd.DataFrame({
        'Consensus Value': cons_values,
        'Cell Type': ['Target' if m else 'Other' for m in target_mask]
    })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.boxplot(x='Cell Type', y='TH Score', data=plot_df_th, ax=ax1, palette={'Target': '#fd8d3c', 'Other': '#6baed6'})
    sns.stripplot(x='Cell Type', y='TH Score', data=plot_df_th, ax=ax1, size=4, alpha=0.3, jitter=True, palette={'Target': '#fd8d3c', 'Other': '#6baed6'})
    ax1.set_title(f'TH Score Distribution for {marker_gene}', fontsize=14)
    
    sns.boxplot(x='Cell Type', y='Consensus Value', data=plot_df_cons, ax=ax2, palette={'Target': '#fd8d3c', 'Other': '#6baed6'})
    sns.stripplot(x='Cell Type', y='Consensus Value', data=plot_df_cons, ax=ax2, size=4, alpha=0.3, jitter=True, palette={'Target': '#fd8d3c', 'Other': '#6baed6'})
    ax2.set_title(f'Consensus Value Distribution for {marker_gene}', fontsize=14)
    
    if np.sum(target_mask) > 0 and np.sum(~target_mask) > 0:
        mean_th_target = np.mean(th_scores[target_mask])
        mean_th_other = np.mean(th_scores[~target_mask])
        
        mean_cons_target = np.mean(cons_values[target_mask])
        mean_cons_other = np.mean(cons_values[~target_mask])
        
        ax1.text(0.05, 0.95, f'Mean (Target): {mean_th_target:.4f}\nMean (Other): {mean_th_other:.4f}\nRatio: {mean_th_target/mean_th_other:.2f}x', 
                transform=ax1.transAxes, fontsize=10, va='top')
        
        ax2.text(0.05, 0.95, f'Mean (Target): {mean_cons_target:.4f}\nMean (Other): {mean_cons_other:.4f}\nRatio: {mean_cons_target/mean_cons_other:.2f}x', 
                transform=ax2.transAxes, fontsize=10, va='top')
    
    plt.suptitle(f'Marker Gene Analysis: {marker_gene} in {cell_type}', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.close()
    
    return {
        'gene': marker_gene,
        'cell_type': cell_type,
        'target_cells': np.sum(target_mask),
        'other_cells': np.sum(~target_mask),
        'th_score_ratio': mean_th_target / mean_th_other if np.sum(~target_mask) > 0 else np.nan,
        'consensus_value_ratio': mean_cons_target / mean_cons_other if np.sum(~target_mask) > 0 else np.nan
    }


def plot_method_comparison_heatmap(adata, filtered_matrices, valid_methods, cell_type, top_n_genes=10, save_path=None):
    from sklearn.preprocessing import StandardScaler
    
    if 'Celltype' not in adata.obs.columns:
        print("Cell type information not available")
        return
    
    cell_mask = adata.obs['Celltype'] == cell_type
    if np.sum(cell_mask) == 0:
        print(f"No cells found of type {cell_type}")
        return
    
    cell_indices = np.where(cell_mask)[0]
    
    if len(filtered_matrices) < 2:
        print("Need at least 2 methods for comparison")
        return
    
    n_genes = filtered_matrices[0].shape[1]
    n_methods = len(filtered_matrices)
    
    gene_method_avg = np.zeros((n_genes, n_methods))
    for i, matrix in enumerate(filtered_matrices):
        gene_method_avg[:, i] = np.mean(matrix[cell_indices, :], axis=0)
    
    gene_cv = np.zeros(n_genes)
    for g in range(n_genes):
        values = gene_method_avg[g, :]
        gene_cv[g] = np.std(values) / (np.mean(values) + 1e-10)
    
    top_gene_indices = np.argsort(gene_cv)[-top_n_genes:][::-1]
    gene_names = [adata.var_names[i] for i in top_gene_indices]
    
    heatmap_data = gene_method_avg[top_gene_indices, :]
    
    scaler = StandardScaler()
    heatmap_data_scaled = scaler.fit_transform(heatmap_data)
    
    plt.figure(figsize=(12, 10))
    
    heatmap_df = pd.DataFrame(
        heatmap_data_scaled,
        index=gene_names,
        columns=valid_methods
    )
    
    ax = sns.heatmap(
        heatmap_df, 
        cmap='viridis', 
        center=0,
        annot=False, 
        cbar_kws={'label': 'Z-Score'}
    )
    
    plt.title(f'Top {top_n_genes} Variable Genes Across Methods for {cell_type}', fontsize=16)
    plt.ylabel('Gene', fontsize=14)
    plt.xlabel('Imputation Method', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.close()
    
    plt.figure(figsize=(14, 8))
    
    for i, gene_idx in enumerate(top_gene_indices[:5]):
        gene_name = adata.var_names[gene_idx]
        plt.subplot(1, 5, i+1)
        
        gene_values = [matrix[cell_indices[0], gene_idx] for matrix in filtered_matrices]
        plt.bar(valid_methods, gene_values)
        plt.title(gene_name, fontsize=12)
        plt.xticks(rotation=90, fontsize=8)
        
        if i == 0:
            plt.ylabel('Imputed Value', fontsize=12)
    
    plt.suptitle(f'Imputed Values for Top Genes in {cell_type}', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        raw_save_path = save_path.replace('.png', '_raw_values.png')
        plt.savefig(raw_save_path, dpi=300, bbox_inches='tight')
        print(f"Saved raw values figure to {raw_save_path}")
    
    plt.close()
    
    return heatmap_df


def create_network_inference_analysis(adata, scores, values, zero_indices, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    thresholds = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
    imputed_matrices = []
    
    is_sparse = hasattr(adata.X, 'toarray')
    if is_sparse:
        original_matrix = adata.X.toarray()
    else:
        original_matrix = adata.X
    
    print(f"Creating matrices with different TH score thresholds...")
    
    for threshold in thresholds:
        imputed_matrix = np.copy(original_matrix)
        
        if threshold == 0.0:
            pass
        elif threshold == 1.0:
            imputed_matrix[zero_indices] = values
        else:
            high_th_indices = scores >= threshold
            rows = zero_indices[0][high_th_indices]
            cols = zero_indices[1][high_th_indices]
            imputed_matrix[rows, cols] = values[high_th_indices]
        
        imputed_matrices.append(imputed_matrix)
    
    print("Saving imputed matrices for GRN inference...")
    
    threshold_names = ["original"] + [f"th_{t}" for t in thresholds[1:-1]] + ["all_imputed"]
    
    for i, (matrix, name) in enumerate(zip(imputed_matrices, threshold_names)):
        np.save(os.path.join(output_dir, f"{name}_matrix.npy"), matrix)
        
        n_zeros = np.sum(matrix == 0)
        pct_zeros = n_zeros / matrix.size * 100
        
        print(f"Matrix '{name}': {n_zeros} zeros ({pct_zeros:.2f}%), shape: {matrix.shape}")
    
    print("Creating zero distribution visualization...")
    
    plt.figure(figsize=(12, 6))
    
    zero_counts = [np.sum(m == 0) for m in imputed_matrices]
    zero_pcts = [count / imputed_matrices[0].size * 100 for count in zero_counts]
    
    plt.bar(threshold_names, zero_pcts, color='#6baed6')
    
    plt.title('Percentage of Zeros in Matrices with Different TH Score Thresholds', fontsize=14)
    plt.xlabel('Imputation Threshold', fontsize=12)
    plt.ylabel('Percentage of Zeros (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    for i, pct in enumerate(zero_pcts):
        plt.text(i, pct + 1, f"{pct:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "zero_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Network inference preparation complete. Files saved to {output_dir}")


def main():
    output_dir = "./imputation_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting scRNA-seq imputation consensus analysis...")
    
    methods = ["SAUCIE", "MAGIC", "deepImpute", "scScope", "scVI", "knn_smoothing"]
    dataset_id = "63ff2c52-cb63-44f0-bac3-d0b33373e312"
    disease = "Crohn disease"
    tissue = "lamina propria of mucosa of colon"

    adata = load_data(dataset_id, disease, tissue)
    print(f"Loaded dataset with {adata.shape[0]} cells and {adata.shape[1]} genes")
    
    marker_genes = load_marker_genes()
    print(f"Loaded {sum(len(genes) for genes in marker_genes.values())} marker genes for {len(marker_genes)} cell types")
    
    imputed_matrices, valid_methods = load_imputed_matrices(methods, dataset_id, disease, tissue, adata)
    
    filtered_matrices, adata_filtered = filter_matrices_to_common_size(imputed_matrices, adata)
    print(f"Filtered matrices to common dimensions: {adata_filtered.shape}")
    
    print("Calculating TH scores...")
    th_results = calculate_th_scores(adata_filtered, filtered_matrices, valid_methods)
    scores = th_results["scores"]
    values = th_results["values"]
    zero_indices = th_results["zero_indices"]
    print(f"Calculated TH scores for {len(scores)} zeros")
    
    print("Running consensus analysis...")
    comparison_df = run_consensus_analysis(output_dir)
    
    print("Generating visualizations...")
    if comparison_df is not None:
        top_markers = comparison_df.sort_values('f1_improvement', ascending=False).head(5)
        
        for _, row in top_markers.iterrows():
            gene = row['gene']
            cell_type = row['cell_type']
            
            if gene in adata.var_names:
                print(f"Analyzing marker gene {gene} for cell type {cell_type}...")
                marker_results = analyze_marker_gene(
                    adata, 
                    scores, 
                    values, 
                    zero_indices, 
                    marker_gene=gene, 
                    cell_type=cell_type,
                    save_path=os.path.join(output_dir, f"{gene}_{cell_type.replace(' ', '_')}_marker_analysis.png")
                )
    
    plt.figure(figsize=(10, 6))
    sns.histplot(scores, bins=50, color='#6baed6', kde=True)
    plt.title('Distribution of TH Scores', fontsize=16)
    plt.xlabel('TH Score', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "th_scores_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(values, bins=50, color='#fd8d3c', kde=True)
    plt.title('Distribution of Consensus Values', fontsize=16)
    plt.xlabel('Consensus Value', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "consensus_values_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
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
    
    for ct in adata.obs['Celltype'].unique():
        if np.sum(adata.obs['Celltype'] == ct) >= 10:
            plot_method_comparison_heatmap(
                adata_filtered, 
                filtered_matrices, 
                valid_methods, 
                cell_type=ct,
                save_path=os.path.join(output_dir, f"{ct.replace(' ', '_')}_method_comparison.png")
            )
    
    create_network_inference_analysis(
        adata=adata_filtered, 
        scores=scores, 
        values=values, 
        zero_indices=zero_indices,
        output_dir=os.path.join(output_dir, "grn_analysis")
    )
        

    print("\nAnalysis complete. Results saved to", output_dir)


if __name__ == "__main__":
    main()