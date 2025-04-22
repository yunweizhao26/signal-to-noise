import os
import numpy as np
import matplotlib.pyplot as plt
from anndata import read_h5ad
from pathlib import Path

###############################################
# 0) Helper function: expand_magic_matrix
###############################################
def expand_magic_matrix(original_matrix, magic_matrix):
    """
    MAGIC sometimes filters out genes with very low expression (e.g., those
    expressed in fewer than 5 cells). This function 'expands' the smaller matrix
    back to the original shape by placing the MAGIC-imputed columns in the
    appropriate positions (kept columns) and retaining the original values in
    the removed columns (dropped by MAGIC).
    
    Parameters
    ----------
    original_matrix : np.ndarray
        The original NxM matrix (N cells, M genes).
    magic_matrix : np.ndarray
        The MAGIC-imputed matrix, which may have fewer columns than 'original_matrix'.
    
    Returns
    -------
    expanded : np.ndarray
        An NxM matrix aligned with 'original_matrix' shape.
    """
    # original_matrix shape: (num_cells, num_genes)
    num_cells, num_genes = original_matrix.shape

    # This logic matches the approach in your pipeline:
    # we identify which columns were dropped (genes not expressed in >=5 cells).
    # Then we place the MAGIC-imputed columns back in the original shape.
    
    # Count how many cells express each gene
    nonzero_counts = (original_matrix != 0).sum(axis=0)

    # Genes that have >=5 non-zero expressions
    kept_columns = np.where(nonzero_counts >= 5)[0]
    # Genes that didn't meet the threshold
    removed_columns = np.where(nonzero_counts < 5)[0]

    # Build an empty array to fill
    expanded = np.zeros((num_cells, num_genes), dtype=magic_matrix.dtype)

    # Fill the kept columns with the new (imputed) values
    for j_reduced, j_original in enumerate(kept_columns):
        expanded[:, j_original] = magic_matrix[:, j_reduced]

    # For removed columns, revert to original values
    for j_original in removed_columns:
        expanded[:, j_original] = original_matrix[:, j_original]

    return expanded

######################################
# 1) Utility to plot a single histogram
######################################
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

def plot_kde_line(
    values,
    output_path,
    title="KDE Distribution",
    xlabel="Value",
    ylabel="Density",
    bw_method=None,      # Bandwidth method (float or str). If None, defaults to scipy's auto
    grid_points=200,
    show_grid=True
):
    """
    Plots and saves a kernel density estimation (KDE) line chart of `values` using Matplotlib.
    This replicates the sort of KDE plot you'd get from seaborn's kdeplot, but without Seaborn.

    Parameters
    ----------
    values : array-like
        1D array of numeric values.
    output_path : str
        File path to save the resulting plot (e.g. 'myplot.png').
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    bw_method : float or str, optional
        Bandwidth for gaussian_kde. E.g., 0.3 or 'silverman'. If None, it uses the default.
    grid_points : int
        Number of points in the range at which to evaluate the KDE for the line plot.
    show_grid : bool
        Whether to show a grid on the plot. Defaults to True.
    """
    # Convert input values to a numpy array (important if `values` is a list)
    values = np.asarray(values)
    
    # Filter out any NaNs or infinite values, if present
    values = values[np.isfinite(values)]
    
    # Skip plotting if no valid data
    if len(values) == 0:
        print("No valid values to plot in KDE.")
        return
    
    # Create the KDE
    kde = gaussian_kde(values, bw_method=bw_method)

    # Create an x-axis range that covers the min to max of your data
    x_min, x_max = values.min(), values.max()
    # Add a small padding in case all values are identical
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    x_range = np.linspace(x_min, x_max, grid_points)
    
    # Evaluate the KDE on the x-range
    y_density = kde(x_range)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(x_range, y_density)  # a simple line chart
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if show_grid:
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_histogram(
    values,
    output_path,
    title="Distribution",
    xlabel="Expression Value",
    ylabel="Count",
    bins="auto",    # or an integer, e.g. 50 or 100
    log_scale=False,
    alpha=1,      # Transparency
    show_grid=True,
    color="teal"  # Change color from the default blue
):
    """
    Plots and saves a histogram of the given values using matplotlib,
    with some additional stylistic tweaks (grid, minor ticks, etc.).

    Parameters
    ----------
    values : array-like
        1D array of numeric values to plot in the histogram.
    output_path : str
        Where to save the resulting .png figure.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    bins : int or str
        Number of bins (int) or strategy (e.g. 'auto'). Defaults to 'auto'.
    log_scale : bool
        Whether to set y-axis to log scale. Defaults to False.
    alpha : float
        Transparency level for the histogram. Defaults to 0.7.
    show_grid : bool
        Whether to show grid lines. Defaults to True.
    color : str
        Histogram color (e.g. "orange", "green", "red"). Defaults to "orange".
    """
    plt.figure(figsize=(8, 6))

    # Plot the histogram with optional alpha and custom color
    plt.hist(values, bins=bins, histtype='stepfilled', alpha=alpha, color=color)

    # Titles and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Optional grid
    if show_grid:
        plt.grid(alpha=0.3)

    # Optionally log-scale the y-axis
    if log_scale:
        plt.yscale('log')

    # Turn on minor ticks for both axes
    plt.minorticks_on()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

##########################################
# 2) Main function to plot distributions
#    - includes checks for MAGIC shape
#    - includes check for originally non-zero changes
##########################################
def plot_original_and_imputed_distributions(
    h5ad_path,
    methods,
    dataset_outdir="plots_distributions"
):
    """
    For each (disease, tissue) subset within a given .h5ad, this function will:
      - Read the file (h5ad_path) into an AnnData object
      - Filter out adata.obs["is_primary_data"] == True
      - Group by (disease, tissue)
      - For each subset:
         A) Plot distribution of all original values
         B) Plot distribution of original non-zero values
         C) For each method, plot distribution of originally-zero positions that
            become > 0 in the imputed data
         D) For each method, check how non-zero are changed by the imputed method
            and plot distribution of (imputed - original) for originally non-zero
    `methods` is a dict mapping:
        method_name -> folder containing imputed .npy files
        (organized similarly to your existing pipeline).
    """
    # Read in the dataset
    adata = read_h5ad(h5ad_path)
    dataset_id = os.path.splitext(os.path.basename(h5ad_path))[0]
    
    # Filter out cells that are not primary_data
    if "is_primary_data" in adata.obs:
        adata = adata[adata.obs["is_primary_data"] == True]
    
    # Create an overall output directory
    os.makedirs(dataset_outdir, exist_ok=True)
    
    # Group by (disease, tissue)
    metadata_markers = adata.obs[["disease", "tissue"]].drop_duplicates()
    
    for idx, (disease, tissue) in enumerate(metadata_markers.values, start=1):
        print(f"Subset {idx}/{len(metadata_markers)}: disease={disease}, tissue={tissue}")
        
        # Mask for just that subset
        mask = (adata.obs["disease"] == disease) & (adata.obs["tissue"] == tissue)
        adata_sample = adata[mask]
        
        # Original expression matrix
        X = adata_sample.X.toarray()  # ensure it's dense

        # Create a subset-specific output folder
        subset_outdir = os.path.join(
            dataset_outdir,
            dataset_id,
            f"{disease}_{tissue}".replace(" ", "_").replace(",", "")
        )
        os.makedirs(subset_outdir, exist_ok=True)
        
        # A) Distribution of ALL original values
        all_values = X.flatten()
        plot_histogram(
            all_values,
            os.path.join(subset_outdir, "original_all_values_hist.png"),
            title=f"{disease}, {tissue}: All Original Expression Values",
            xlabel="Expression Value",
            ylabel="Count",
            bins=100
        )
        
        # B) Distribution of original NON-ZERO values only
        nonzero_values = all_values[all_values > 0]
        if len(nonzero_values) > 0:
            plot_histogram(
                nonzero_values,
                os.path.join(subset_outdir, "original_nonzero_values_hist.png"),
                title=f"{disease}, {tissue}: Original Non-Zero Values",
                xlabel="Expression Value",
                ylabel="Count",
                bins=100
            )
        else:
            print(" - No non-zero values found in this subset.")
        
        # Track which entries were originally zero or non-zero
        originally_zero_mask = (X == 0)
        originally_nonzero_mask = (X > 0)
        
        ##################################################
        # C) For each method, load the imputed matrix
        #    then plot distributions for originally-zero
        #    and originally-non-zero differences
        ##################################################
        for method_name, method_folder in methods.items():
            # Path to the method’s .npy for the disease/tissue
            imputed_path = os.path.join(method_folder, dataset_id, disease, tissue + ".npy")
            if not os.path.exists(imputed_path):
                print(f"   {method_name}: no imputed file at {imputed_path}")
                continue
            
            imputed_matrix = np.load(imputed_path)
            
            # If the method is MAGIC, fix shape by expanding columns
            # so that imputed_matrix has the same shape as X.
            if method_name.upper() == "MAGIC":
                if imputed_matrix.shape != X.shape:
                    imputed_matrix = expand_magic_matrix(X, imputed_matrix)
            
            # Check again if shape matches after expansion
            if imputed_matrix.shape != X.shape:
                print(f"   [WARNING] shape mismatch even after expansion: original={X.shape}, imputed={imputed_matrix.shape}")
                print("   Skipping distribution plots for this subset/method.")
                continue
            
            ########################
            # C.1) originally zero => non-zero
            ########################
            newly_imputed_values = imputed_matrix[originally_zero_mask]
            newly_imputed_values = newly_imputed_values[newly_imputed_values > 0]
            
            if len(newly_imputed_values) == 0:
                print(f"   {method_name}: no originally-zero positions became non-zero.")
            else:
                plot_histogram(
                    newly_imputed_values,
                    os.path.join(subset_outdir, f"{method_name}_original_zero_became_nonzero.png"),
                    title=(f"{disease}, {tissue}: Zeros Imputed with Non-zeros by {method_name}"),
                    xlabel="Imputed Value",
                    ylabel="Count",
                    bins=100
                )
            
            ##################################################
            # D) Check how originally non-zero are changed
            #    Plot distribution of (imputed - original)
            ##################################################
            orig_nonzero_vals = X[originally_nonzero_mask]
            imputed_nonzero_vals = imputed_matrix[originally_nonzero_mask]
            diff = imputed_nonzero_vals - orig_nonzero_vals
            
            if len(diff) > 0:
                # (D.1) distribution of differences
                plot_histogram(
                    diff,
                    os.path.join(subset_outdir, f"{method_name}_original_nonzero_diff.png"),
                    title=(f"{disease}, {tissue}: Diff(imputed, original) for original non-zeros ({method_name})"),
                    xlabel="Difference (Imputed - Original)",
                    ylabel="Count",
                    bins=100
                )
                
                # (D.2) check how many originally non-zero become EXACTLY zero
                newly_zero_count = np.sum(imputed_nonzero_vals == 0)
                if newly_zero_count > 0:
                    print(f"   {method_name}: {newly_zero_count} original non-zeros changed to zeros.")
            
            print("")  # line break for logs
        
        print("")  # spacer for clarity in logs


##############################################
# Example usage in __main__
##############################################
if __name__ == "__main__":
    # Where your .h5ad files are located
    files = list(Path("./datasets").glob("*.h5ad"))
    
    # Dictionary to where you have the .npy files for each imputation method
    # *Note: the shape mismatch typically happens with MAGIC, so
    #        we apply expand_magic_matrix above if the method is "MAGIC".
    imputed_methods = {
        "SAUCIE": "output/SAUCIE",
        "MAGIC":  "output/MAGIC",
    }

    for f in files:
        print(f"Processing dataset {f.stem}")
        
        # Call the function with the file path
        plot_original_and_imputed_distributions(
            h5ad_path=str(f),
            methods=imputed_methods,
            dataset_outdir="plots_distributions"
        )
