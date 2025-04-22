import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path


def run_preprocessing(adata):
    adata.layers["counts"] = adata.X.copy()
    
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="Sample")
    
    return adata[:, adata.var["highly_variable"]]


def run_pipeline_on_sets():
    h5ad_files = list(Path("./datasets").glob("*.h5ad"))
    sc.settings.verbosity = 2 # verbosity: errors (0), warnings (1), info (2), hints (3)
    
    for f in h5ad_files:
        adata = sc.read_h5ad(f)
        # filter out cells that actually originate from other datasets
        adata = adata[adata.obs["is_primary_data"] == True]
        
        adata = run_preprocessing(adata)
        
        #run_imputation_methods(adata)



############## OTHER STUFF


# https://scanpy.readthedocs.io/en/stable/tutorials/experimental/pearson_residuals.html
def run_preprocessing_pearson(adata):
    sc.experimental.pp.highly_variable_genes(
        adata, flavor="pearson_residuals", n_top_genes=2000
    )
    
    adata = adata[:, adata.var["highly_variable"]]
    
    adata.layers["raw"] = adata.X.copy()
    adata.layers["sqrt_norm"] = np.sqrt(
        sc.pp.normalize_total(adata, inplace=False)["X"]
    )
    
    sc.experimental.pp.normalize_pearson_residuals(adata)
    
    sc.pp.pca(adata, n_comps=50)
    
    return adata


def run_magic(adata):
    import scanpy.external as sce
    sce.pp.magic(adata, name_list='all_genes', knn=5)


def run_qc_metrics(adata, species, plot_path):
    import re
    
    ########## QC METRICS
    print("QC metrics...")
    
    
    mt_ = "MT-" if species == "Homo sapiens" else "Mt-"
    
    # this is based on the format utilized by cellxgene census
    adata.var["mt"] = [feat.startswith(mt_) for feat in adata.var["feature_name"]]
    adata.var["ribo"] = [feat.startswith(("RPS", "RPL")) for feat in adata.var["feature_name"]]
    adata.var["hb"] = [bool(re.match("^HB[^(P)]", feat)) for feat in adata.var["feature_name"]]
    
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
    )
    
    sc.pl.violin(
        adata,
        ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
        jitter=0.4,
        multi_panel=True,
        show=False,
    )
    plt.savefig(plot_path / "qc_metrics_violin.png")
    
    sc.pl.scatter(
        adata,
        "total_counts", 
        "n_genes_by_counts", 
        color="pct_counts_mt",
        show=False
    )
    plt.savefig(plot_path / "qc_metrics_scatter.png")
    
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)
    
    
    print("QC metrics done")


from matplotlib import pyplot as plt
def run_clustering(adata, plot_path):
    from matplotlib import pyplot as plt
    ########## NEAREST NEIGHBOR & VISUALIZATION
    print("Nearest neighbor & UMAP...")
    
    
    sc.pp.neighbors(adata)
    
    sc.tl.umap(adata)
    
    sc.pl.umap(
        adata,
        #color="disease",
        # # Setting a smaller point size to get prevent overlap
        # size=2,
        show=False
    )
    plt.savefig(plot_path / "umap.png")
    
    
    print("Finished UMAP reduction")
    
    
    ########## CLUSTERING
    print("Clustering...")
    
    
    # Using the igraph implementation and a fixed number of iterations can be significantly faster, especially for larger datasets
    sc.tl.leiden(adata, flavor="igraph", n_iterations=2)
    
    sc.pl.umap(adata, color=["leiden"], show=False)
    plt.savefig(plot_path / "umap_leiden.png")
    
    
    print("Finished clustering")
    

def run_qc_again(adata, plot_path):
    ########## RE-ASSESS QC
    print("QC 2.0")
    
    
    sc.pl.umap(
        adata,
        color=["leiden", "predicted_doublet", "doublet_score"],
        # increase horizontal space between panels
        wspace=0.5,
        size=3,
        show=False
    )
    plt.savefig(plot_path / "qc_metrics_umap_1.png")
    
    sc.pl.umap(
        adata,
        color=["leiden", "log1p_total_counts", "pct_counts_mt", "log1p_n_genes_by_counts"],
        wspace=0.5,
        ncols=2,
        show=False
    )
    plt.savefig(plot_path / "qc_metrics_umap_2.png")