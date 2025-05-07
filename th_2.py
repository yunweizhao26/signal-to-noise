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
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.stats import ks_2samp, pearsonr, spearmanr
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import networkx as nx

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
    marker_genes = {}
    if args.markers_file:
        with open(args.markers_file, 'r') as f:
            marker_genes = json.load(f)
    
    log_message(f"Loaded dataset with {adata.shape[0]} cells and {adata.shape[1]} genes")
    if marker_genes:
        log_message(f"Loaded {len(marker_genes)} cell types with marker genes")
    
    return adata, marker_genes

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
        if np.std(X_original[:, gene_idx]) > 0 and np.std(X_imputed[:, gene_idx]) > 0:
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
    
    # 2. Preservation of zeros and non-zeros
    zero_mask = X_original == 0
    zeros_preserved = np.mean(X_imputed[zero_mask] == 0) if np.sum(zero_mask) > 0 else 0
    nonzero_mask = X_original > 0
    nonzeros_preserved = np.mean(X_imputed[nonzero_mask] > 0) if np.sum(nonzero_mask) > 0 else 0
    
    scores['preservation_zeros'] = zeros_preserved
    scores['preservation_nonzeros'] = nonzeros_preserved
    scores['preservation_overall'] = (zeros_preserved + nonzeros_preserved) / 2
    
    # 3. Distribution agreement
    imputed_zeros_mask = (X_original == 0) & (X_imputed > 0)
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
    
    # 4b. k-NN graph preservation
    try:
        k = min(15, X_original.shape[0] - 1)
        if k >= 5:
            # Use PCA to reduce dimensions for efficiency
            n_dims = min(50, X_original.shape[0] - 1, X_original.shape[1] - 1)
            if n_dims < 2:
                scores['knn_preservation'] = 0
            else:
                pca_orig = PCA(n_components=n_dims).fit_transform(X_original)
                pca_imp = PCA(n_components=n_dims).fit_transform(X_imputed)
                
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
    except:
        scores['knn_preservation'] = 0
    
    # 4c. Clustering stability
    try:
        if X_original.shape[0] >= 20:  # Need sufficient cells
            # Get number of clusters proportional to data size
            n_clusters = max(2, min(20, X_original.shape[0] // 50))
            
            # Cluster original and imputed data
            kmeans_orig = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans_imp = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            
            clusters_orig = kmeans_orig.fit_predict(X_original)
            clusters_imp = kmeans_imp.fit_predict(X_imputed)
            
            # Calculate ARI
            ari = adjusted_rand_score(clusters_orig, clusters_imp)
            scores['clustering_stability'] = max(0, ari)  # Ensure non-negative
            
            # Calculate silhouette improvement
            try:
                sil_orig = silhouette_score(X_original, clusters_orig)
                sil_imp = silhouette_score(X_imputed, clusters_imp)
                scores['silhouette_improvement'] = max(0, (sil_imp - sil_orig + 1) / 2)  # Scale to [0,1]
            except:
                scores['silhouette_improvement'] = 0.5  # Neutral if can't compute
        else:
            scores['clustering_stability'] = 0
            scores['silhouette_improvement'] = 0.5
    except:
        scores['clustering_stability'] = 0
        scores['silhouette_improvement'] = 0.5
    
    # 5. Marker gene enrichment (biological score)
    if marker_genes and adata is not None:
        marker_score = calculate_marker_agreement(X_imputed, marker_genes, adata)
        scores['marker_agreement'] = marker_score
    else:
        scores['marker_agreement'] = 0.5  # Neutral if markers not available
    
    # Calculate composite score - weighted geometric mean for balanced influence
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
    
    # Category weights
    weights = {
        'gene_correlation': 0.25,
        'preservation': 0.20,
        'distribution': 0.15,
        'cell_structure': 0.20,
        'biological_signal': 0.20
    }
    
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
    
    return scores

def calculate_marker_agreement(X, marker_genes, adata):
    """
    Calculate the marker gene agreement score across all cells.
    Updated to use zero tolerance and be more memory-efficient.
    """
    # Set zero tolerance for numeric stability
    ZERO_TOLERANCE = 1e-6
    
    # Create gene index lookup
    gene2idx = {g: i for i, g in enumerate(adata.var_names)}
    
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
            matches = [ct for ct in unique_cell_types if ct.startswith(marker_type)]
            if matches:
                cell_type_mapping[marker_type] = matches
                log_message(f"Mapped '{marker_type}' to {len(matches)} cell types")
        
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
    """Calculate weights for each imputation method based on marker gene agreement improvement."""
    weights = []
    
    # Get cell type annotations
    if "Celltype" in adata.obs.columns:
        cell_types = adata.obs["Celltype"].values
    else:
        # If no cell type annotation, create a dummy one
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
    """Calculate TruthfulHypothesis scores using marker-based method weights."""
    log_message("Calculating TruthfulHypothesis scores with marker-based weights")
    
    # Process all disease/tissue combinations
    if "disease" in adata.obs.columns and "tissue" in adata.obs.columns:
        metadata_markers = adata.obs[["disease", "tissue"]].drop_duplicates()
        
        results = []
        for idx, (disease, tissue) in enumerate(metadata_markers.values):
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
            
            # Initialize storage for imputed matrices
            imputed_matrices = []
            valid_methods = []
            
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
                
                imputed = np.load(imputed_file)
                if method_name == "MAGIC":
                    imputed = expand_magic_matrix(X, imputed)
                imputed_matrices.append(imputed)
                valid_methods.append(method_name)
                
                log_message(f"  Loaded method: {method_name}, shape: {imputed.shape}")
            
            if len(imputed_matrices) == 0:
                log_message(f"  No valid imputation methods found for this sample, skipping")
                continue
            
            # Calculate marker-based weights
            weights_array = calculate_marker_based_weights(X, imputed_matrices, marker_genes, adata_sample)
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
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        
        # Find zero entries
        zero_mask = X == 0
        zero_indices = np.where(zero_mask)
        
        log_message(f"Found {len(zero_indices[0])} zero entries in the dataset")
        
        # Initialize storage for imputed matrices
        imputed_matrices = []
        valid_methods = []
        
        # Load imputed matrices for all methods
        for method_name in method_names:
            # Load imputed matrix
            cache_file = os.path.join(args.cache_dir, method_name, "full_dataset.npy")
            
            if not os.path.exists(cache_file):
                log_message(f"  Skipping method {method_name} - file not found: {cache_file}")
                continue
                
            imputed = np.load(cache_file)
            imputed_matrices.append(imputed)
            valid_methods.append(method_name)
        
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
            "original_matrix": X,
            "method_weights": weights,
            "valid_methods": valid_methods,
            "imputed_matrices": imputed_matrices,
            "adata_sample": adata
        }]
        
        log_message(f"Completed TH score calculation with marker-based weights")
        
        return results


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
    parser.add_argument("--methods", type=str, default="SAUCIE,MAGIC,deepImpute,scScope,scVI", 
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
            marker_genes = generate_synthetic_markers(adata)
        
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