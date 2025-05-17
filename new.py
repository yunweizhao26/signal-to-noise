# %%
import os
import re
import sys
import csv
import json
import torch
import queue
import math
import time
import umap.umap_ as umap
import numpy as np
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import multiprocessing
import torch.optim as optim
from itertools import product
import matplotlib.pyplot as plt
import torch.nn.functional as F
import datetime
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    auc,
    f1_score,
    confusion_matrix,
    matthews_corrcoef
)
from typing import Dict, List, Tuple
from sklearn.decomposition import FactorAnalysis, PCA, NMF
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from lightning_lite.utilities.seed import seed_everything
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls import VariationalELBO
from scipy.special import expit
from pathlib import Path
from multiprocessing import Queue

_GLOBAL_CUSTOM: List[Dict] = []

# enable our method on beelines
# test on non-specific dataset
# enable gnnlink to run on the datasets and get performance
# enable genelink to run on the datasets and get performance
# enable beeline to run on the datasets and get the performance
# enable gnnlink to run on our simulated datasets and get performance
# enable genelink to run on our simulated datasets and get performance
# enable beeline to run on our datasets and get performance

DATASET_INFO_MAPPING = {
    # STRING (14 combos)
    1501: ("STRING", "hESC 500"),
    1502: ("STRING", "hESC 1000"),
    1503: ("STRING", "hHEP 500"),
    1504: ("STRING", "hHEP 1000"),
    1505: ("STRING", "mDC 500"),
    1506: ("STRING", "mDC 1000"),
    1507: ("STRING", "mESC 500"),
    1508: ("STRING", "mESC 1000"),
    1509: ("STRING", "mHSC-E 500"),
    1510: ("STRING", "mHSC-E 1000"),
    1511: ("STRING", "mHSC-GM 500"),
    1512: ("STRING", "mHSC-GM 1000"),
    1513: ("STRING", "mHSC-L 500"),
    1514: ("STRING", "mHSC-L 1000"),

    # Non-Specific (14 combos)
    1601: ("Non-Specific", "hESC 500"),
    1602: ("Non-Specific", "hESC 1000"),
    1603: ("Non-Specific", "hHEP 500"),
    1604: ("Non-Specific", "hHEP 1000"),
    1605: ("Non-Specific", "mDC 500"),
    1606: ("Non-Specific", "mDC 1000"),
    1607: ("Non-Specific", "mESC 500"),
    1608: ("Non-Specific", "mESC 1000"),
    1609: ("Non-Specific", "mHSC-E 500"),
    1610: ("Non-Specific", "mHSC-E 1000"),
    1611: ("Non-Specific", "mHSC-GM 500"),
    1612: ("Non-Specific", "mHSC-GM 1000"),
    1613: ("Non-Specific", "mHSC-L 500"),
    1614: ("Non-Specific", "mHSC-L 1000"),

    # Specific (14 combos)
    1701: ("Specific", "hESC 500"),
    1702: ("Specific", "hESC 1000"),
    1703: ("Specific", "hHEP 500"),
    1704: ("Specific", "hHEP 1000"),
    1705: ("Specific", "mDC 500"),
    1706: ("Specific", "mDC 1000"),
    1707: ("Specific", "mESC 500"),
    1708: ("Specific", "mESC 1000"),
    1709: ("Specific", "mHSC-E 500"),
    1710: ("Specific", "mHSC-E 1000"),
    1711: ("Specific", "mHSC-GM 500"),
    1712: ("Specific", "mHSC-GM 1000"),
    1713: ("Specific", "mHSC-L 500"),
    1714: ("Specific", "mHSC-L 1000"),

    # Lofgof (2 combos)
    1801: ("Lofgof", "mESC 500"),
    1802: ("Lofgof", "mESC 1000"),
}

# %%
def parse_dataset_name(folder_name):
    pattern1 = r'De-noised_(\d+)G_(\d+)T_(\d+)cPerT_dynamics_(\d+)_DS(\d+)'
    pattern2 = r'De-noised_(\d+)G_(\d+)T_(\d+)cPerT_(\d+)_DS(\d+)'
    match_p1 = re.match(pattern1, folder_name)
    match_p2 = re.match(pattern2, folder_name)
    if match_p1:
        return {
            'number_genes': int(match_p1.group(1)),
            'number_bins': int(match_p1.group(2)),
            'cells_per_type': int(match_p1.group(3)),
            'dynamics': int(match_p1.group(4)),
            'dataset_id': int(match_p1.group(5)),
            'folder_name': folder_name
        }
    if match_p2:
        return {
            'number_genes': int(match_p2.group(1)),
            'number_bins': int(match_p2.group(2)),
            'cells_per_type': int(match_p2.group(3)),
            'dynamics': int(match_p2.group(4)),
            'dataset_id': int(match_p2.group(5)),
            'folder_name': folder_name
        }
    return

def get_datasets():
    datasets = []
    data_sets_dir = './SERGIO/data_sets'
    for folder_name in os.listdir(data_sets_dir):
        dataset_info = parse_dataset_name(folder_name)
        if dataset_info:
            datasets.append(dataset_info)
    new_datasets = [
        {
            'dataset_id': 1000,
            'dataset_name': 'mDC',
            'expression_file': 'data/raws/mDC-ExpressionData.csv',
            'network_file': 'data/raws/mDC-network.csv',
        },
        {
            'dataset_id': 1001,
            'dataset_name': 'mESC',
            'expression_file': 'data/raws/mESC-ExpressionData.csv',
            'network_file': 'data/raws/mESC-network.csv',
        },
        {
            'dataset_id': 1002,
            'dataset_name': 'mHSC-E',
            'expression_file': 'data/raws/mHSC-E-ExpressionData.csv',
            'network_file': 'data/raws/mHSC-E-network.csv',
        },
        {
            'dataset_id': 1003,
            'dataset_name': 'mHSC-GM',
            'expression_file': 'data/raws/mHSC-GM-ExpressionData.csv',
            'network_file': 'data/raws/mHSC-GM-network.csv',
        },
        {
            'dataset_id': 1004,
            'dataset_name': 'mHSC-L',
            'expression_file': 'data/raws/mHSC-L-ExpressionData.csv',
            'network_file': 'data/raws/mHSC-L-network.csv',
        },
        {
            'dataset_id': 1005,
            'dataset_name': 'hESC',
            'expression_file': 'data/raws/hESC-ExpressionData.csv',
            'network_file': 'data/raws/hESC-network.csv',
        },
        {
            'dataset_id': 1006,
            'dataset_name': 'hHep',
            'expression_file': 'data/raws/hHep-ExpressionData.csv',
            'network_file': 'data/raws/hHep-network.csv',
        },
        {
            'dataset_id': 1007,
            'dataset_name': 'mouse',
            'expression_file': 'data/raws/mouse-ExpressionData.csv',
            'network_file': 'data/raws/mouse-network.csv',
        },
        {
            'dataset_id': 1008,
            'dataset_name': 'human',
            'expression_file': 'data/raws/human-ExpressionData.csv',
            'network_file': 'data/raws/human-network.csv',
        },
    ]
    datasets.extend(new_datasets)
    genelink_datasets = []
    for ds_id, (net_type, cell_type_str) in DATASET_INFO_MAPPING.items():
        # net_type => "Non-Specific"
        # cell_type_str => "hESC 500"
        base_dir = f"../bni/GENELink/processed/{net_type}/{cell_type_str}"
        expression_file = os.path.join(base_dir, "BL--ExpressionData.csv")
        network_file = os.path.join(base_dir, "BL--network.csv")

        dataset_name = f"{net_type}_{cell_type_str.replace(' ', '_')}"  
        # e.g. "Non-Specific_hESC_500"

        genelink_datasets.append({
            'dataset_id': ds_id,
            'dataset_name': dataset_name,
            'expression_file': expression_file,
            'network_file': network_file,
        })
    datasets.extend(genelink_datasets)
    datasets.extend(_GLOBAL_CUSTOM)
    return datasets

def load_network_data(file_path, gene_list):
    # Map gene names to indices
    gene_to_index = {gene: idx for idx, gene in enumerate(gene_list)}
    num_genes = len(gene_list)
    H = np.zeros((num_genes, num_genes))
    import csv
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for line in reader:
            source_gene, target_gene = line
            if source_gene in gene_to_index and target_gene in gene_to_index:
                source_idx = gene_to_index[source_gene]
                target_idx = gene_to_index[target_gene]
                H[source_idx, target_idx] = 1
    return H

def load_data(dataset_info):
    dataset_id = dataset_info['dataset_id']
    if dataset_id < 1000:
        # Existing datasets
        data_dir = f'../SERGIO/imputation_data_2/DS{dataset_id}/'
        ds_clean_path = os.path.join(data_dir, 'DS6_clean.npy')
        ds_noisy_path = os.path.join(data_dir, 'DS6_45_iter_0.npy')
        if not os.path.exists(ds_clean_path) or not os.path.exists(ds_noisy_path):
            print(f"Data files not found for Dataset {dataset_id}. Skipping.")
            return None, None, None
        ds_clean = np.load(ds_clean_path).astype(np.float32)
        ds_noisy = np.load(ds_noisy_path).astype(np.float32)
        gene_names = None
        cell_names = None
    elif 'h5ad_file' in dataset_info:
        import anndata, scipy.sparse as sp
        adata = anndata.read_h5ad(dataset_info['h5ad_file'])

        mask = (
            (adata.obs["disease"] == dataset_info['disease']) &
            (adata.obs["tissue"]  == dataset_info['tissue'])
        )
        adata_sample = adata[mask]

        X = adata_sample.X
        if sp.issparse(X):
            X = X.toarray()
        ds_noisy = X.T.astype(np.float32)            # genes × cells
        ds_clean = None                              # not available
        gene_names = adata_sample.var_names.tolist() # genes
        cell_names = adata_sample.obs_names.tolist() # cells
        # number of genes and cells
        print(f"Number of genes: {len(gene_names)}, Number of cells: {len(cell_names)}")
    else:
        expression_file = dataset_info['expression_file']
        if not os.path.exists(expression_file):
            print(f"Expression file not found for Dataset {dataset_id}. Skipping.")
            return None, None, None, None
        import pandas as pd
        df = pd.read_csv(expression_file, index_col=0)
        ds_noisy = df.values.astype(np.float32)
        # Not available for new datasets, so set ds_noisy as raw data
        ds_clean = None  
        gene_names = df.index.tolist()
        cell_names = df.columns.tolist()
        # print("gene_names: ", gene_names)
    return ds_clean, ds_noisy, gene_names, cell_names

def load_interactions_info(num_genes, interactions_file):
    gt = np.zeros((num_genes, num_genes))
    with open(interactions_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line_list = line.strip().split(',')
        target_index = int(float(line_list[0]))
        num_regs = int(float(line_list[1]))
        for i in range(num_regs):
            try:
                reg_index = int(float(line_list[i + 2]))
                gt[reg_index, target_index] = 1
            except:
                continue
    return gt

def load_transformer_data_for_contrastive(exp_file, train_file, test_file, val_file,
                                          embedding_type='fa', n_components=64):
    """
    Load transformer dataset format and convert to format needed for contrastive learning
    """
    # Load expression data
    data_input = pd.read_csv(exp_file, index_col=0)
    gene_names = data_input.index
    features = data_input.values
    num_genes = features.shape[0]

    # Load split data
    train_data = pd.read_csv(train_file, index_col=0).values
    valid_data = pd.read_csv(val_file,   index_col=0).values
    test_data  = pd.read_csv(test_file,  index_col=0).values

    # Sort by label => separate positives from negatives
    train_data = train_data[np.lexsort(-train_data.T)]
    valid_data = valid_data[np.lexsort(-valid_data.T)]
    test_data  = test_data[np.lexsort(-test_data.T)]

    # Count positives in each split
    train_pos_idx = np.sum(train_data[:, 2])
    val_pos_idx   = np.sum(valid_data[:, 2])
    test_pos_idx  = np.sum(test_data[:, 2])

    # Build adjacency matrices
    G_train = create_adjacency_matrix(train_data, train_pos_idx, num_genes)
    G_valid = create_adjacency_matrix(valid_data, val_pos_idx, num_genes)
    G_test  = create_adjacency_matrix(test_data,  test_pos_idx, num_genes)

    # Normalize features using standard scaling
    ds_scaled = StandardScaler().fit_transform(features)
    embeddings = generate_embeddings(data_input=ds_scaled,
                                    embedding_type=embedding_type,
                                    n_components=n_components)

    return {
        'features': embeddings,
        'gene_names': gene_names,
        'G_train': G_train,
        'G_valid': G_valid,
        'G_test': G_test,
        'metrics': {
            'train_pos': train_pos_idx,
            'train_total': len(train_data),
            'valid_pos': val_pos_idx,
            'valid_total': len(valid_data),
            'test_pos': test_pos_idx,
            'test_total': len(test_data)
        }
    }

def create_adjacency_matrix(data, pos_idx, num_genes):
    """
    Convert edge list format to adjacency matrix with -1 for unobserved edges
    """
    adj = -np.ones((num_genes, num_genes), dtype=np.int8)
    # Set positive edges
    for i in range(pos_idx):
        source, target = int(data[i,0]), int(data[i,1])
        adj[source, target] = 1
    # Set negative edges
    for i in range(pos_idx, len(data)):
        source, target = int(data[i,0]), int(data[i,1])
        adj[source, target] = 0
    return adj

def prepare_data_for_training(features, adj_matrix, batch_size, negative_ratio):
    """
    Prepare data for contrastive learning training
    """
    positive_edges = np.argwhere(adj_matrix == 1)
    negative_edges = np.argwhere(adj_matrix == 0)
    num_pos = len(positive_edges)
    num_batches = num_pos // batch_size
    return positive_edges, negative_edges, num_batches

def get_metrics_info(data_dict):
    """
    Get dataset statistics for logging
    """
    metrics = data_dict['metrics']
    return {
        'train_ratio': metrics['train_pos'] / metrics['train_total'],
        'valid_ratio': metrics['valid_pos'] / metrics['valid_total'],
        'test_ratio': metrics['test_pos'] / metrics['test_total'],
        'num_genes': data_dict['features'].shape[0],
        'num_cells': data_dict['features'].shape[1]
    }

def generate_embeddings(data, embedding_type="PCA", n_components=64, scale=False, visualize=False):
    """
    Apply dimensionality reduction on single-cell gene expression data using various methods.
    
    Parameters:
        data (array-like or AnnData): Input data matrix (cells x genes). Can be a NumPy array, pandas DataFrame,
                                      or AnnData (for methods like PAGA that use scanpy).
        method (str): One of {'PHATE', 'DiffusionMaps', 'PAGA', 'NMF', 'PCA', 'FA'}.
        n_components (int): Number of dimensions for the embedding.
        scale (bool): Whether to apply standard scaling to data before reduction.
        visualize (bool): If True, plot the resulting 2D embedding (if n_components >= 2).
    
    Returns:
        np.ndarray: Embedding array of shape (n_samples, n_components).
    """
    X = data
    # If input is AnnData and not PAGA method, extract the .X matrix
    try:
        import scanpy as sc
        if isinstance(X, sc.AnnData):
            # For non-PAGA methods, use the expression matrix; for PAGA, we'll use AnnData directly.
            if embedding_type != "PAGA":
                X = X.X  # get numpy matrix from AnnData
    except ImportError:
        # scanpy not available, assume data is array-like
        pass
    
    # Convert to numpy array if not already (for sklearn compatibility)
    X = np.array(X) if not isinstance(X, np.ndarray) else X
    
    # Optional scaling
    if scale:
        X = StandardScaler().fit_transform(X)
    
    embedding = None
    embedding_type = embedding_type.lower()  # make case-insensitive
    if embedding_type == "pca":
        # PCA transformation
        model = PCA(n_components=n_components, random_state=0)
        embedding = model.fit_transform(X)
    elif embedding_type == "fa" or embedding_type == "factoranalysis":
        # Factor Analysis
        model = FactorAnalysis(n_components=n_components, random_state=0)
        embedding = model.fit_transform(X)
    elif embedding_type == "nmf":
        model = NMF(n_components=n_components, init='nndsvda', random_state=0)
        # Ensure non-negativity for NMF (if X has negatives, one might need to normalize or shift)
        if np.min(X) < 0:
            # If data has negative values (e.g., after scaling), clip or shift to non-negative
            X_nmf = X - np.min(X)
        else:
            X_nmf = X
        embedding = model.fit_transform(X_nmf)
    elif embedding_type == "phate":
        try:
            import phate
        except ImportError:
            raise ImportError("phate library is not installed. Install via `pip install phate` to use this method.")
        # PHATE can handle high-dimensional data (including sparse) and returns an embedding.
        phate_op = phate.PHATE(n_components=n_components, random_state=0)
        embedding = phate_op.fit_transform(X)
    elif embedding_type in ["diffusionmaps", "diffusionmap", "diffusion_maps"]:
        try:
            from pydiffmap import diffusion_map as dm
        except ImportError:
            raise ImportError("pydiffmap library is not installed. Install via `pip install pydiffmap` to use Diffusion Maps.")
        # Construct diffusion map. Choose parameters (epsilon, alpha, k) or use defaults.
        # Using from_sklearn to utilize sklearn's neighbor finding
        dmap = dm.DiffusionMap.from_sklearn(n_evecs=n_components, alpha=0.5)  # you can tune epsilon and k as needed
        embedding = dmap.fit_transform(X)
        # dmap.fit_transform returns the diffusion eigenvectors for each sample (excluding the first trivial constant eigenvector by default).
    elif embedding_type == "paga":
        try:
            import scanpy as sc
        except ImportError:
            raise ImportError("scanpy is required for PAGA. Install via `pip install scanpy` to use this method.")
        if not isinstance(data, sc.AnnData):
            # convert data matrix to AnnData for scanpy processing
            adata = sc.AnnData(X)
        else:
            adata = data.copy()  # work on a copy to avoid modifying original
        # PAGA requires a neighborhood graph and clustering
        sc.pp.neighbors(adata, n_neighbors=15, use_rep='X')         # build KNN graph from data
        sc.tl.leiden(adata, resolution=1.0, key_added="paga_groups")  # cluster cells (you can adjust resolution)
        sc.tl.paga(adata, groups="paga_groups")
        # Compute a layout for the abstracted graph (coarse-grained graph of cluster connectivity)
        sc.pl.paga(adata, layout="fr", random_state=0, show=False)  # Fruchterman-Reingold layout; don't show plot
        # The coordinates of cluster nodes are stored in adata.uns['paga']['pos']
        cluster_positions = adata.uns['paga']['pos']
        # Map each cell to the coordinates of its cluster:
        labels = adata.obs["paga_groups"].astype(int).values
        embedding = cluster_positions[labels]
    else:
        raise ValueError(f"Unknown method '{method}'. Supported methods are: PHATE, DiffusionMaps, PAGA, NMF, PCA, FA.")
    
    # Ensure output is a numpy array (in case some library returns a list or other structure)
    embedding = np.array(embedding)
    
    # Optional visualization (simple 2D scatter if applicable)
    if visualize:
        import matplotlib.pyplot as plt
        if embedding.shape[1] >= 2:
            plt.figure(figsize=(6,5))
            plt.scatter(embedding[:, 0], embedding[:, 1], s=10, alpha=0.8)
            plt.title(f"{method.upper()} embedding")
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.show()
        else:
            print("Visualization: Embedding has <2 dimensions, cannot plot scatter.")
    
    return embedding

# %%
def load_ground_truth_grn(file_path, num_genes=None, gene_names=None):
    if gene_names is None:
        # Existing datasets: indices
        if num_genes is None:
            with open(file_path, 'r') as f:
                indices = []
                for line in f:
                    source, target = map(int, line.strip().split(','))
                    indices.extend([source, target])
            num_genes = max(indices) + 1
        H = np.zeros((num_genes, num_genes))
        with open(file_path, 'r') as f:
            for line in f:
                source, target = map(int, line.strip().split(','))
                H[source, target] = 1
    else:
        # New datasets: gene names
        num_genes = len(gene_names)
        gene_to_idx = {gene.lower(): idx for idx, gene in enumerate(gene_names)}
        H = np.zeros((num_genes, num_genes))
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            for line in reader:
                # print(line)
                source_gene, target_gene = line[0].lower(), line[1].lower()
                if source_gene in gene_to_idx and target_gene in gene_to_idx:
                    source_idx = gene_to_idx[source_gene]
                    target_idx = gene_to_idx[target_gene]
                    H[source_idx, target_idx] = 1
    return H

def sample_partial_grn(H, sample_ratio=8/10):  # 8:2 ratio
    positive_edges = np.argwhere(H == 1)
    negative_edges = np.argwhere(H == 0)
    total_positives = len(positive_edges)
    total_negatives = len(negative_edges)
    num_pos_sample = int(total_positives * sample_ratio)
    num_neg_sample = int(total_negatives * sample_ratio)
    np.random.shuffle(positive_edges)
    np.random.shuffle(negative_edges)
    sampled_pos = positive_edges[:num_pos_sample]
    sampled_neg = negative_edges[:num_neg_sample]
    G = -np.ones_like(H)
    G[tuple(zip(*sampled_pos))] = 1
    G[tuple(zip(*sampled_neg))] = 0
    return G

def split_train_valid(G, train_ratio=7/8):  # 7:1 ratio
    positive_edges = np.argwhere(G == 1)
    negative_edges = np.argwhere(G == 0)  # Only sampled negative edges
    num_pos_train = int(len(positive_edges) * train_ratio)
    num_neg_train = int(len(negative_edges) * train_ratio)
    
    np.random.shuffle(positive_edges)
    np.random.shuffle(negative_edges)

    train_pos = positive_edges[:num_pos_train]
    valid_pos = positive_edges[num_pos_train:]
    train_neg = negative_edges[:num_neg_train]
    valid_neg = negative_edges[num_neg_train:]
    
    G_train = -np.ones_like(G)
    G_valid = -np.ones_like(G)
    G_train[tuple(zip(*train_pos))] = 1
    G_train[tuple(zip(*train_neg))] = 0
    G_valid[tuple(zip(*valid_pos))] = 1
    G_valid[tuple(zip(*valid_neg))] = 0
    return G_train, G_valid

def get_test_set(H, G):
    remaining_positives = np.argwhere((H == 1) & (G == -1))
    remaining_negatives = np.argwhere((H == 0) & (G == -1))
    G_test = -np.ones_like(H)
    G_test[tuple(zip(*remaining_positives))] = 1
    G_test[tuple(zip(*remaining_negatives))] = 0
    print(f"Test set - Positive edges: {len(remaining_positives)}, Negative edges: {len(remaining_negatives)}")  
    return G_test
# %%
def generate_balanced_batches(embeddings, adjacency_matrix, batch_size, num_batches):
    num_nodes = embeddings.shape[0]
    positive_edges = np.argwhere(adjacency_matrix == 1)
    negative_edges = np.argwhere(adjacency_matrix == 0)
    
    np.random.shuffle(positive_edges)
    np.random.shuffle(negative_edges)
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(positive_edges))
        
        batch_positive = positive_edges[start_idx:end_idx]
        batch_negative = negative_edges[np.random.choice(len(negative_edges), size=len(batch_positive), replace=False)]

        batch_edges = np.concatenate([batch_positive, batch_negative])
        batch_labels = np.concatenate([np.ones(len(batch_positive)), np.zeros(len(batch_negative))])
        
        # Shuffle the batch
        shuffle_idx = np.random.permutation(len(batch_edges))
        batch_edges = batch_edges[shuffle_idx]
        batch_labels = batch_labels[shuffle_idx]
        
        x1 = torch.tensor(embeddings[batch_edges[:, 0]], dtype=torch.float32).to(device)
        x2 = torch.tensor(embeddings[batch_edges[:, 1]], dtype=torch.float32).to(device)
        labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)
        yield x1, x2, labels

def generate_batches(embeddings, adjacency_matrix, batch_size, num_batches, negative_ratio=10):
    positive_edges = np.argwhere(adjacency_matrix == 1)
    negative_edges = np.argwhere(adjacency_matrix == 0)
    np.random.shuffle(positive_edges)
    np.random.shuffle(negative_edges)
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(positive_edges))
        batch_positive = positive_edges[start_idx:end_idx]
        num_negatives = len(batch_positive) * negative_ratio
        if num_negatives > len(negative_edges):
            batch_negative = negative_edges[np.random.choice(len(negative_edges), size=num_negatives, replace=True)]
        else:
            batch_negative = negative_edges[np.random.choice(len(negative_edges), size=num_negatives, replace=False)]
        batch_edges = np.concatenate([batch_positive, batch_negative])
        batch_labels = np.concatenate([np.ones(len(batch_positive)), np.zeros(len(batch_negative))])
        # print(batch_positive.shape, batch_labels)
        # Shuffle the batch
        shuffle_idx = np.random.permutation(len(batch_edges))
        batch_edges = batch_edges[shuffle_idx]
        batch_labels = batch_labels[shuffle_idx]
        x1 = torch.tensor(embeddings[batch_edges[:, 0]], dtype=torch.float32)
        x2 = torch.tensor(embeddings[batch_edges[:, 1]], dtype=torch.float32)
        labels = torch.tensor(batch_labels, dtype=torch.float32)
        yield x1, x2, labels

class DirectionalContrastiveModel(nn.Module):
    def __init__(self, input_dim, proj_dim):
        super().__init__()
        self.projection_source = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim)
        )
        self.projection_target = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim)
        )
    
    def forward(self, x1, x2):
        return self.projection_source(x1), self.projection_target(x2)

    def get_embeddings(self, x, combine_mode="avg"):
        # returns a single embedding
        with torch.no_grad():
            src_emb = self.projection_source(x)
            tgt_emb = self.projection_target(x)
            if combine_mode == "avg":
                return 0.5*(src_emb + tgt_emb)
            elif combine_mode == "cat":
                return torch.cat([src_emb, tgt_emb], dim=-1)
            else:
                return src_emb

# %%
class ContrastiveModel(nn.Module):
    def __init__(self, input_dim, projection_dim):
        super(ContrastiveModel, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )
    
    def forward(self, x1, x2):
        proj1 = self.projection(x1)
        proj2 = self.projection(x2)
        return proj1, proj2

class SoftNearestNeighborLoss(nn.Module):
    def __init__(self, temperature=10., cos_distance=True):
        super(SoftNearestNeighborLoss, self).__init__()
        self.temperature = temperature
        self.cos_distance = cos_distance

    def pairwise_cos_distance(self, A, B):
        query_embeddings = F.normalize(A, dim=1)
        key_embeddings = F.normalize(B, dim=1)
        distances = 1 - torch.matmul(query_embeddings, key_embeddings.T)
        return distances

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: Batched embeddings to compute the SNNL.
            labels: Labels of embeddings.
        """
        batch_size = embeddings.shape[0]
        eps = 1e-9

        if self.cos_distance:
            pairwise_dist = self.pairwise_cos_distance(embeddings, embeddings)
        else:
            pairwise_dist = torch.cdist(embeddings, embeddings, p=2)

        pairwise_dist = pairwise_dist / self.temperature
        negexpd = torch.exp(-pairwise_dist)

        # Creating mask to sample same class neighborhood
        pairs_y = torch.broadcast_to(labels, (batch_size, batch_size))
        mask = pairs_y == torch.transpose(pairs_y, 0, 1)
        mask = mask.float()

        # creating mask to exclude diagonal elements
        ones = torch.ones([batch_size, batch_size], dtype=torch.float32).cuda()
        dmask = ones - torch.eye(batch_size, dtype=torch.float32).cuda()

        # all class neighborhood
        alcn = torch.sum(torch.multiply(negexpd, dmask), dim=1)
        # same class neighborhood
        sacn = torch.sum(torch.multiply(negexpd, mask), dim=1)

        # Adding eps for numerical stability
        loss = -torch.log((sacn+eps)/alcn).mean()
        return loss

def train_snn_directional(embeddings, adjacency_matrix, input_dim, projection_dim, num_epochs, batch_size, learning_rate, negative_ratio, temperature, device):    # print("0")
    model = DirectionalContrastiveModel(input_dim, projection_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = SoftNearestNeighborLoss(temperature=temperature)
    num_positive_edges = (adjacency_matrix == 1).sum().item()
    num_batches = num_positive_edges // batch_size
    print(f"num_positive_edges: {num_positive_edges}, num_batches: {num_batches}")

    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for x1, x2, labels in generate_batches(embeddings, adjacency_matrix, batch_size, num_batches, negative_ratio=negative_ratio):
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            optimizer.zero_grad()
            proj1, proj2 = model(x1, x2)
            embeddings_batch = torch.cat([proj1, proj2], dim=0)
            labels_batch = torch.cat([labels, labels], dim=0)
            loss = criterion(embeddings_batch, labels_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    return model

def train_snn(embeddings, adjacency_matrix, input_dim, projection_dim, num_epochs, batch_size, learning_rate, negative_ratio, temperature, device):
    # print("0")
    model = ContrastiveModel(input_dim, projection_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = SoftNearestNeighborLoss(temperature=temperature)
    num_positive_edges = (adjacency_matrix == 1).sum().item()
    num_batches = num_positive_edges // batch_size
    print(f"num_positive_edges: {num_positive_edges}, num_batches: {num_batches}")

    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for x1, x2, labels in generate_batches(embeddings, adjacency_matrix, batch_size, num_batches, negative_ratio=negative_ratio):
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            optimizer.zero_grad()
            proj1, proj2 = model(x1, x2)
            embeddings_batch = torch.cat([proj1, proj2], dim=0)
            labels_batch = torch.cat([labels, labels], dim=0)
            loss = criterion(embeddings_batch, labels_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    return model

# %%
def get_k_metrics(true_labels, predicted_scores):
    k = int(np.sum(true_labels))
    sorted_indices = np.argsort(predicted_scores)[::-1]
    sorted_labels = true_labels[sorted_indices]
    precision_k = np.sum(sorted_labels[:k]) / k
    recall_k = np.sum(sorted_labels[:k]) / np.sum(true_labels)
    return precision_k, recall_k

def format_metric_with_std(name, metrics, metric_key):
    mean = metrics['means'][metric_key]
    std = metrics['stds'][metric_key]
    ci = metrics['ci'][metric_key]
    return f"{name}: {mean:.4f} +- {std:.4f} (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])"

def print_evaluation_results(metrics):
    print("Evaluation Results:")
    print("=" * 50)
    print(format_metric_with_std("AUC-ROC", metrics, 'auc_roc'))
    print(format_metric_with_std("AUC-PR", metrics, 'auc_pr'))
    print(format_metric_with_std("F1 Score", metrics, 'f1_score'))
    print(format_metric_with_std("MCC", metrics, 'mcc'))
    print(format_metric_with_std("K-Precision", metrics, 'k_precision'))
    print(format_metric_with_std("K-Recall", metrics, 'k_recall'))

def visualize_clusters(embeddings, cluster_assignments, save_path):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_assignments, cmap='viridis', s=50, alpha=0.7)
    plt.title('t-SNE Visualization of Clusters')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.colorbar(scatter, label='Cluster')
    plt.savefig(save_path)
    plt.close()

def compute_clustering_metrics(embeddings, cluster_assignments):
    # Ensure embeddings are in the correct format
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    silhouette_avg = silhouette_score(embeddings_scaled, cluster_assignments)
    davies_bouldin = davies_bouldin_score(embeddings_scaled, cluster_assignments)
    calinski_harabasz = calinski_harabasz_score(embeddings_scaled, cluster_assignments)

    print(f"Silhouette Score: {silhouette_avg:.4f}")
    print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
    print(f"Calinski-Harabasz Index: {calinski_harabasz:.4f}")

    return {
        'silhouette_score': silhouette_avg,
        'davies_bouldin_index': davies_bouldin,
        'calinski_harabasz_index': calinski_harabasz
    }

class GPClassificationModel(ApproximateGP):
    def __init__(self, inducing_points, model_type='standard', direction_weight=1.0):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, 
            variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        
        self.model_type = model_type
        self.mean_module = gpytorch.means.ConstantMean()
        
        if model_type == 'directional':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                DirectionalRBFKernel(
                    src_weight=2.0,
                    tgt_weight=1.0,
                    dir_weight=5.0)
            )
            self.direction_weight = direction_weight
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class DirectionalRBFKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True
    
    def __init__(self, src_weight=1.0, tgt_weight=1.0, dir_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.src_weight = nn.Parameter(torch.tensor(src_weight))
        self.tgt_weight = nn.Parameter(torch.tensor(tgt_weight))
        self.dir_weight = nn.Parameter(torch.tensor(dir_weight))

    def forward(self, x1, x2, diag=False, **params):
        """
        x1: (batch_size, d)
        x2: (batch_size, d)
        
        If diag=True, we only need the diagonal of the kernel matrix, i.e., K(x_i, x_i).
        That means the distance is zero -> kernel = 1.0 for RBF (before scaling).
        """
        if diag:
            # If x1 and x2 are the same input, the diagonal is just 1 for each entry
            # (distance = 0 -> exp(-0.5*0)=1).
            # Return shape = [batch_size]
            return torch.ones(x1.size(0), dtype=x1.dtype, device=x1.device)

        # Otherwise, compute the full covariance matrix:
        D = (x1.shape[-1] - 1) // 2  # Assume last dim is direction
        dir1, dir2 = x1[..., -1], x2[..., -1]
        
        src1, tgt1 = x1[..., :D], x1[..., D:-1]
        src2, tgt2 = x2[..., :D], x2[..., D:-1]
        
        # Component-wise distances
        dist_src = self.src_weight * (src1.unsqueeze(-2) - src2.unsqueeze(-3)).pow(2).sum(-1)
        dist_tgt = self.tgt_weight * (tgt1.unsqueeze(-2) - tgt2.unsqueeze(-3)).pow(2).sum(-1)
        dist_dir = self.dir_weight * (dir1.unsqueeze(-1) - dir2.unsqueeze(-2)).pow(2)
        
        return torch.exp(-0.5 * (dist_src + dist_tgt + dist_dir) / self.lengthscale.pow(2))

def train_gp_model(projected_embeddings, adjacency_matrix, device, 
                  model_type='standard', direction_weight=1.0,
                  inducing_points_num=500, num_epochs=100, 
                  batch_size=1024, run_seed=42):
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)

    # Prepare training data with directional information
    train_edges = np.argwhere((adjacency_matrix == 1) | (adjacency_matrix == 0))
    train_labels = adjacency_matrix[train_edges[:, 0], train_edges[:, 1]].astype(int)

    emb_i = torch.tensor(projected_embeddings[train_edges[:, 0]], dtype=torch.float32)
    emb_j = torch.tensor(projected_embeddings[train_edges[:, 1]], dtype=torch.float32)
    
    if model_type == 'directional':
        direction = torch.ones(len(train_edges), 1, dtype=torch.float32)  # Ensure same dtype
        print("Direction tensor:", direction, direction.type())
        print("emb_i tensor:", emb_i, emb_i.type())
        print("emb_j tensor:", emb_j, emb_j.type())
        X_train = torch.cat([emb_i, emb_j, direction], dim=1)
    else:
        X_train = torch.cat([emb_i, emb_j], dim=1)

    X_train = X_train.to(device)
    y_train = torch.from_numpy(train_labels).float().to(device)
    
    # Initialize inducing points
    inducing_points = X_train[:min(inducing_points_num, X_train.shape[0])]

    # Initialize model
    model = GPClassificationModel(
        inducing_points=inducing_points.to(device),
        model_type=model_type,
        direction_weight=direction_weight
    ).to(device)
    likelihood = BernoulliLikelihood().to(device)

    num_data = y_train.size(0)
    mll = VariationalELBO(likelihood, model, num_data=num_data)
    optimizer = torch.optim.AdamW([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.005)

    model.train()
    likelihood.train()
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
    
    return model, likelihood, X_train, y_train

def evaluate_bayesian_model_gp(model, likelihood, projected_embeddings, 
                              adjacency_matrix, H, device):
    """
    Evaluate the trained GP model on validation or test data.

    Args:
        model: Trained GP model
        likelihood: Trained likelihood
        projected_embeddings: numpy array of shape (num_genes, embedding_dim)
        adjacency_matrix: Adjacency matrix for the evaluation set with values:
            -1 for unknown edges (edges to predict)
        H: Ground truth adjacency matrix (complete network)
        device: Torch device ('cuda' or 'cpu')

    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    num_genes = projected_embeddings.shape[0]

    # Prepare test data
    test_edges = np.argwhere(adjacency_matrix == -1)
    true_labels = H[test_edges[:, 0], test_edges[:, 1]].astype(int)
    emb_i_test = torch.tensor(projected_embeddings[test_edges[:, 0]], dtype=torch.float32)
    emb_j_test = torch.tensor(projected_embeddings[test_edges[:, 1]], dtype=torch.float32)
    if model.model_type == 'directional':
        direction = torch.ones(len(test_edges), 1, dtype=torch.float32)
        X_test = torch.cat([emb_i_test, emb_j_test, direction], dim=1)
    else:
        X_test = torch.cat([emb_i_test, emb_j_test], dim=1)

    # Convert to torch tensors
    X_test = X_test.to(device)
    y_test = torch.from_numpy(true_labels).float().to(device)

    # Evaluation
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_dist = model(X_test)
        test_preds = likelihood(test_dist)
        predicted_probs = test_preds.mean.cpu().numpy()
        y_test_np = y_test.cpu().numpy()

    # Compute evaluation metrics
    metrics = {}
    print("unique values in y_test: ", np.unique(y_test_np))
    metrics['auc_roc'] = roc_auc_score(y_test_np, predicted_probs)
    precision, recall, thresholds = precision_recall_curve(y_test_np, predicted_probs)
    metrics['auc_pr'] = auc(recall, precision)

    # For binary predictions, choose threshold 0.5
    y_pred = (predicted_probs >= 0.5).astype(int)
    metrics['f1_score'] = f1_score(y_test_np, y_pred, zero_division=0)
    metrics['confusion_matrix'] = confusion_matrix(y_test_np, y_pred)
    metrics['mcc'] = matthews_corrcoef(y_test_np, y_pred)
    k_precision, k_recall = get_k_metrics(y_test_np, predicted_probs)
    metrics['k_precision'] = k_precision
    metrics['k_recall'] = k_recall

    # Store additional data for analysis
    metrics.update({
        'precision_curve': precision,
        'recall_curve': recall,
        'thresholds': thresholds,
        'true_labels': y_test_np,
        'probabilities': predicted_probs,
        'num_positive': np.sum(y_test_np == 1),
        'num_negative': np.sum(y_test_np == 0)
    })

    return metrics

def evaluate_bayesian_model_bb(projected_embeddings, adjacency_matrix, H, run_seed=42):
    num_genes = projected_embeddings.shape[0]

    # Clustering gene embeddings using BayesianGaussianMixture
    # scaler = StandardScaler()
    # standardized_embeddings = scaler.fit_transform(projected_embeddings)

    max_components = num_genes
    bgm = BayesianGaussianMixture(
        n_components=max_components,
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=1e-2,
        max_iter=1000,
        n_init=1,
        random_state=run_seed
    )
    cluster_assignments = bgm.fit_predict(projected_embeddings)
    num_clusters = np.unique(cluster_assignments).shape[0]
    print(f"Number of clusters found: {num_clusters}")
    
    unique_clusters = np.unique(cluster_assignments)
    num_clusters = len(unique_clusters)
    cluster_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_clusters)}
    mapped_cluster_assignments = np.array([cluster_mapping[label] for label in cluster_assignments])

    # Initialize counts for edges between clusters
    edge_counts = np.zeros((num_clusters, num_clusters))
    total_counts = np.zeros((num_clusters, num_clusters))

    # Use adjacency_matrix to compute counts of edges between clusters
    positive_edges = np.argwhere(adjacency_matrix == 1)
    negative_edges = np.argwhere(adjacency_matrix == 0)

    # Update counts for positive edges
    for edge in positive_edges:
        i, j = edge
        ci = mapped_cluster_assignments[i]
        cj = mapped_cluster_assignments[j]
        edge_counts[ci, cj] += 1
        total_counts[ci, cj] += 1

    # Update counts for negative edges
    for edge in negative_edges:
        i, j = edge
        ci = mapped_cluster_assignments[i]
        cj = mapped_cluster_assignments[j]
        total_counts[ci, cj] += 1

    print("total_counts: ", total_counts)
    
    # Compute edge probabilities between clusters using Beta priors
    alpha_prior = 1.0
    beta_prior = 1.0
    theta = (edge_counts + alpha_prior) / (total_counts + alpha_prior + beta_prior)
    print("theta: ", theta)
 
    # For edges to be predicted (adjacency_matrix == -1), compute P(e_{ij} = 1) = theta_{c_i c_j}
    test_edges = np.argwhere(adjacency_matrix == -1)
    true_labels = H[test_edges[:, 0], test_edges[:, 1]].astype(int)
    predicted_probs = []
    for edge in test_edges:
        i, j = edge
        ci = mapped_cluster_assignments[i]
        cj = mapped_cluster_assignments[j]
        prob = theta[ci, cj]
        predicted_probs.append(prob)
    predicted_probs = np.array(predicted_probs)

    # Evaluate metrics
    metrics = {}
    metrics['auc_roc'] = roc_auc_score(true_labels, predicted_probs)
    precision, recall, thresholds = precision_recall_curve(true_labels, predicted_probs)
    metrics['auc_pr'] = auc(recall, precision)

    # For binary predictions, you can choose a threshold (e.g., 0.5)
    y_pred = (predicted_probs >= 0.5).astype(int)
    metrics['f1_score'] = f1_score(true_labels, y_pred, zero_division=0)
    metrics['confusion_matrix'] = confusion_matrix(true_labels, y_pred)
    metrics['mcc'] = matthews_corrcoef(true_labels, y_pred)
    k_precision, k_recall = get_k_metrics(true_labels, predicted_probs)
    metrics['k_precision'] = k_precision
    metrics['k_recall'] = k_recall

    # Store additional data for analysis
    metrics.update({
        'precision_curve': precision,
        'recall_curve': recall,
        'thresholds': thresholds,
        'true_labels': true_labels,
        'probabilities': predicted_probs,
        'num_positive': np.sum(true_labels == 1),
        'num_negative': np.sum(true_labels == 0)
    })

    return metrics, mapped_cluster_assignments

def evaluate_bayesian_model2(projected_embeddings, adjacency_matrix, H, alpha, prior_matrix=None):
    """
    Evaluate the model using the Bayesian framework with prior incorporation and normalization.

    Args:
        projected_embeddings: numpy array of shape (num_genes, embedding_dim)
        adjacency_matrix: numpy array with values:
            -1 for unknown edges (test set)
            1 for positive edges (train/validation set)
            0 for negative edges (train/validation set)
        H: Ground truth adjacency matrix (complete network)
        alpha: Hyperparameter in the likelihood function
        prior_matrix: numpy array representing P(e | G) for all edges (optional)

    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    # Only consider edges in the test set
    test_edges = np.argwhere(adjacency_matrix == -1)
    true_labels = H[test_edges[:, 0], test_edges[:, 1]].astype(int)

    # Compute distances between embeddings
    emb1 = projected_embeddings[test_edges[:, 0]]
    emb2 = projected_embeddings[test_edges[:, 1]]
    distances = np.linalg.norm(emb1 - emb2, axis=1)

    # Compute likelihoods
    likelihoods = np.exp(-alpha * distances)

    # Compute priors (uniform if not provided)
    if prior_matrix is not None:
        priors = prior_matrix[test_edges[:, 0], test_edges[:, 1]]
    else:
        priors = np.ones_like(likelihoods)

    # Compute posterior probabilities (normalize likelihood * prior)
    posterior_probs = likelihoods * priors
    posterior_probs = posterior_probs / np.sum(posterior_probs)

    # Evaluate metrics
    metrics = {}
    metrics['auc_roc'] = roc_auc_score(true_labels, posterior_probs)
    precision, recall, thresholds = precision_recall_curve(true_labels, posterior_probs)
    metrics['auc_pr'] = auc(recall, precision)

    # For binary predictions, use median as threshold
    threshold = np.median(posterior_probs)
    y_pred = (posterior_probs >= threshold).astype(int)
    metrics['f1_score'] = f1_score(true_labels, y_pred, zero_division=0)
    metrics['confusion_matrix'] = confusion_matrix(true_labels, y_pred)
    metrics['mcc'] = matthews_corrcoef(true_labels, y_pred)
    k_precision, k_recall = get_k_metrics(true_labels, posterior_probs)
    metrics['k_precision'] = k_precision
    metrics['k_recall'] = k_recall

    # Store additional data for analysis
    metrics.update({
        'precision_curve': precision,
        'recall_curve': recall,
        'thresholds': thresholds,
        'true_labels': true_labels,
        'probabilities': posterior_probs,
        'distances': distances,
        'num_positive': np.sum(true_labels == 1),
        'num_negative': np.sum(true_labels == 0),
        'priors': priors
    })

    return metrics

def evaluate_bayesian_model(projected_embeddings, adjacency_matrix, H, alpha):
    """
    Evaluate the model using the Bayesian framework.

    Args:
        projected_embeddings: numpy array of shape (num_genes, embedding_dim)
        adjacency_matrix: numpy array with values:
            -1 for unknown edges (test set)
            1 for positive edges (train/validation set)
            0 for negative edges (train/validation set)
        H: Ground truth adjacency matrix (complete network)
        alpha: Hyperparameter in the likelihood function

    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    # Only consider edges in the test set (adjacency_matrix == -1)
    test_edges = np.argwhere(adjacency_matrix == -1)
    true_labels = H[test_edges[:, 0], test_edges[:, 1]].astype(int)  # Ground truth labels

    # Compute distances between embeddings
    emb1 = projected_embeddings[test_edges[:, 0]]
    emb2 = projected_embeddings[test_edges[:, 1]]
    distances = np.linalg.norm(emb1 - emb2, axis=1)

    # Compute likelihoods
    likelihoods = np.exp(-alpha * distances)

    # Assuming uniform prior, posterior probabilities are proportional to likelihoods
    posterior_probs = likelihoods  # P(e | x, G) ∝ P(x | e, G)

    # Evaluate metrics
    metrics = {}
    metrics['auc_roc'] = roc_auc_score(true_labels, posterior_probs)
    precision, recall, thresholds = precision_recall_curve(true_labels, posterior_probs)
    metrics['auc_pr'] = auc(recall, precision)

    # For binary predictions, you can choose a threshold (e.g., median)
    threshold = np.median(posterior_probs)
    y_pred = (posterior_probs >= threshold).astype(int)
    metrics['f1_score'] = f1_score(true_labels, y_pred, zero_division=0)
    metrics['confusion_matrix'] = confusion_matrix(true_labels, y_pred)
    metrics['mcc'] = matthews_corrcoef(true_labels, y_pred)
    k_precision, k_recall = get_k_metrics(true_labels, posterior_probs)
    metrics['k_precision'] = k_precision
    metrics['k_recall'] = k_recall

    # Store additional data for analysis
    metrics.update({
        'precision_curve': precision,
        'recall_curve': recall,
        'thresholds': thresholds,
        'true_labels': true_labels,
        'probabilities': posterior_probs,
        'distances': distances,
        'num_positive': np.sum(true_labels == 1),
        'num_negative': np.sum(true_labels == 0)
    })

    return metrics

def evaluate_contrastive_model(model, embeddings, adjacency_matrix, loss_function, device):
    """Evaluation using full test set to maintain consistency"""
    # print("1")
    model.eval()
    with torch.no_grad():
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
        projected_embeddings = model.projection(embeddings_tensor)
        projected_embeddings = nn.functional.normalize(projected_embeddings, p=2, dim=1)
        projected_embeddings = projected_embeddings.cpu().numpy()
        # print("2")
        # Get all edges in test set
        test_edges = np.argwhere(adjacency_matrix != -1)  # -1 for ignored edges
        true_labels = adjacency_matrix[test_edges[:, 0], test_edges[:, 1]]
        # print("3")
        # Compute scores for all edges
        emb1 = projected_embeddings[test_edges[:, 0]]
        emb2 = projected_embeddings[test_edges[:, 1]]
        similarity_scores = np.sum(emb1 * emb2, axis=1)
        # print("4")
        if loss_function == 'SNN':
            similarity_scores = 1 - similarity_scores
        probabilities = 1 / (1 + np.exp(-similarity_scores))
        # print("5")
        print(f"Evaluating on {len(true_labels)} edges")
        print(f"Positive edges: {np.sum(true_labels == 1)}")
        print(f"Negative edges: {np.sum(true_labels == 0)}")
        
        metrics = {}
        print("true_labels: ", true_labels)
        print("true_labels.shape: ", true_labels.shape)
        print("probabilities: ", probabilities)
        print("probabilities.shape: ", probabilities.shape)
        metrics['auc_roc'] = roc_auc_score(true_labels, probabilities)
        precision, recall, thresholds = precision_recall_curve(true_labels, probabilities)
        metrics['auc_pr'] = auc(recall, precision)
        y_pred = (probabilities >= 0.5).astype(int)
        print("y_pred: ", y_pred)
        print("y_pred.shape: ", y_pred.shape)
        metrics['f1_score'] = f1_score(true_labels, y_pred, zero_division=0)
        metrics['confusion_matrix'] = confusion_matrix(true_labels, y_pred)
        metrics['mcc'] = matthews_corrcoef(true_labels, y_pred)
        k_precision, k_recall = get_k_metrics(true_labels, probabilities)
        metrics['k_precision'] = k_precision
        metrics['k_recall'] = k_recall
        
        # Store all data
        metrics.update({
            'precision_curve': precision,
            'recall_curve': recall,
            'thresholds': thresholds,
            'true_labels': true_labels,
            'probabilities': probabilities,
            'num_positive': np.sum(true_labels == 1),
            'num_negative': np.sum(true_labels == 0)
        })
        
        return metrics

def evaluate_without_posterior(embeddings, adjacency_matrix, H):
    test_edges = np.argwhere(adjacency_matrix == -1)
    true_labels = H[test_edges[:, 0], test_edges[:, 1]].astype(int)

    emb1 = embeddings[test_edges[:, 0]]
    emb2 = embeddings[test_edges[:, 1]]
    distances = np.linalg.norm(emb1 - emb2, axis=1)
    scores = -distances

    metrics = {}
    metrics['auc_roc'] = roc_auc_score(true_labels, scores)
    precision, recall, thresholds = precision_recall_curve(true_labels, scores)
    metrics['auc_pr'] = auc(recall, precision)
    
    threshold = np.median(scores)
    y_pred = (scores >= threshold).astype(int)
    metrics['f1_score'] = f1_score(true_labels, y_pred, zero_division=0)
    metrics['confusion_matrix'] = confusion_matrix(true_labels, y_pred)
    metrics['mcc'] = matthews_corrcoef(true_labels, y_pred)
    k_precision, k_recall = get_k_metrics(true_labels, scores)
    metrics['k_precision'] = k_precision
    metrics['k_recall'] = k_recall

    metrics.update({
        'precision_curve': precision,
        'recall_curve': recall,
        'thresholds': thresholds,
        'true_labels': true_labels,
        'probabilities': scores,
        'num_positive': np.sum(true_labels == 1),
        'num_negative': np.sum(true_labels == 0)
    })

    return metrics

# %%
def plot_distribution(scores, labels, set_name, log_dir):
        plt.figure(figsize=(12, 6))
        probabilities = expit(scores)
        sns.kdeplot(probabilities[labels == 0], fill=True, color="skyblue", label="Negative", cut=0)
        sns.kdeplot(probabilities[labels == 1], fill=True, color="red", label="Positive", cut=0)
        
        plt.title(f'{set_name} Set Predicted Probabilities Distribution')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.legend()
        total = len(labels)
        pos_prop = np.sum(labels == 1) / total
        neg_prop = 1 - pos_prop
        plt.text(0.05, 0.95, f"Negative: {neg_prop:.2%}\nPositive: {pos_prop:.2%}", 
                transform=plt.gca().transAxes, verticalalignment='top')
        plt.savefig(os.path.join(log_dir, f'{set_name.lower()}_distribution.png'))
        plt.close()

def log_experiment(result):
    dataset_id = result['dataset_id']
    embedding_type = result['embedding_type']
    loss_function = result['loss_function']
    base_dir = f'./logs/{embedding_type}_{loss_function}_DS{dataset_id}_yunwei/'
    os.makedirs(base_dir, exist_ok=True)
    existing_versions = [int(d.split('_')[1]) for d in os.listdir(base_dir) if d.startswith('version_')]
    next_version = max(existing_versions + [0]) + 1
    log_dir = os.path.join(base_dir, f'version_{next_version}')
    os.makedirs(log_dir, exist_ok=True)
    
    # Save parameters and results
    params = {k: v for k, v in result.items() 
              if k not in ['model_state_dict', 'projected_embeddings',
                           'train_labels', 'train_scores',
                           'valid_labels', 'valid_scores',
                           'test_labels', 'test_scores']}
    with open(os.path.join(log_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=4)
    
    # Generate and save plots
    plot_distribution(result['train_scores'], result['train_labels'], 'Train', log_dir)
    if 'valid_scores' in result:
        plot_distribution(result['valid_scores'], result['valid_labels'], 'Validation', log_dir)
    plot_distribution(result['test_scores'], result['test_labels'], 'Test', log_dir)
    
    # Save model and embeddings
    torch.save(result['model_state_dict'], 
               os.path.join(log_dir, f'cl_model_{embedding_type}_{loss_function}.pth'))
    np.save(os.path.join(log_dir, f'projected_embeddings_{embedding_type}_{loss_function}.npy'),
            result['projected_embeddings'])
    
    print(f"Experiment logged at: {log_dir}")
    
    # ---------------------------
    #  Append row to master CSV
    # ---------------------------
    master_csv_path = "./master_results.csv"
    # Ensure file exists with header
    file_existed = os.path.isfile(master_csv_path)
    
    # Extract test metrics from result (already has mean ± std, e.g. "0.9123 +- 0.0456")
    test_auc = result['test_auc']       # e.g. "0.9234 +- 0.0123"
    test_pr  = result['test_pr_auc']    # e.g. "0.8456 +- 0.0345"
    test_time = result['test_time']
    # If you also want the time, you need to measure/record it in run_experiment 
    # (e.g. by using time.time() at the start/end) and store e.g. `result['time']`

    # Lookup (net_type, cell_type)
    if dataset_id in DATASET_INFO_MAPPING:
        net_type, cell_type = DATASET_INFO_MAPPING[dataset_id]
    else:
        net_type, cell_type = ("UnknownNet", f"Dataset_{dataset_id}")

    # If you want to parse the "xx.xx +- yy.yy" format into separate columns, you can.  
    # For example, if test_auc = "0.9234 +- 0.0123", you can do:
    #   auc_mean_str, auc_std_str = test_auc.split(" +- ")
    # or keep it as a single string column.

    csv_header = [
        "Network", "CellType", "Dataset_ID", 
        "EmbeddingType", "LossFunction", 
        "Test_AUC", "Test_PR", "Test_Time"
    ]
    csv_row = [
        net_type, cell_type, dataset_id,
        embedding_type, loss_function,
        test_auc, test_pr, test_time
    ]
    
    # Append row
    with open(master_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_existed:
            writer.writerow(csv_header)
        writer.writerow(csv_row)

    print(f"Experiment logged at: {log_dir}")
    return log_dir

def save_split_data(dataset_id, split_name, expression_data, network_data, gene_names, cell_names, base_dir='./data/splits'):
    dataset_dir = os.path.join(base_dir, f'DS{dataset_id}')
    split_dir = os.path.join(dataset_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    # genes = [f'Gene_{i}' for i in range(expression_data.shape[0])]
    # cells = [f'Cell_{i}' for i in range(expression_data.shape[1])]
    expression_df = pd.DataFrame(expression_data, index=gene_names, columns=cell_names)
    expression_df.to_csv(os.path.join(split_dir, 'ExpressionData.csv'))

    positive_edges = np.argwhere(network_data == 1)
    negative_edges = np.argwhere(network_data == 0)
    
    pos_edge_df = pd.DataFrame(positive_edges, columns=['Gene1', 'Gene2'])
    pos_edge_df['Gene1'] = pos_edge_df['Gene1'].apply(lambda x: gene_names[x])
    pos_edge_df['Gene2'] = pos_edge_df['Gene2'].apply(lambda x: gene_names[x])
    pos_edge_df.to_csv(os.path.join(split_dir, 'pos_refNetwork.csv'), index=False)
    
    # Negative edges
    neg_edge_df = pd.DataFrame(negative_edges, columns=['Gene1', 'Gene2'])
    neg_edge_df['Gene1'] = neg_edge_df['Gene1'].apply(lambda x: gene_names[x])
    neg_edge_df['Gene2'] = neg_edge_df['Gene2'].apply(lambda x: gene_names[x])
    neg_edge_df.to_csv(os.path.join(split_dir, 'neg_refNetwork.csv'), index=False)

    # For "refNetwork.csv" (positive edges only)
    ref_edge_df = pd.DataFrame(positive_edges, columns=['Gene1', 'Gene2'])
    ref_edge_df['Gene1'] = ref_edge_df['Gene1'].apply(lambda x: gene_names[x])
    ref_edge_df['Gene2'] = ref_edge_df['Gene2'].apply(lambda x: gene_names[x])
    ref_edge_df.to_csv(os.path.join(split_dir, 'refNetwork.csv'), index=False)
    
    # 3) Save NumPy arrays if desired
    np.save(os.path.join(split_dir, 'expression.npy'), expression_data)
    np.save(os.path.join(split_dir, 'network.npy'), network_data)


def save_split_info(dataset_id, train_ratio, split_info, base_dir='./data/splits'):
    dataset_dir = os.path.join(base_dir, f'DS{dataset_id}')
    os.makedirs(dataset_dir, exist_ok=True)
    
    info_file = os.path.join(dataset_dir, f'split_info_{train_ratio:.2f}.json')
    with open(info_file, 'w') as f:
        json.dump(split_info, f, indent=4)

def plot_performance_curves(true_labels, pred_probs, save_dir, prefix=''):
    """
    Generate and save ROC curve, PR curve, and probability distribution plots
    
    Args:
        true_labels: numpy array of true binary labels
        pred_probs: numpy array of predicted probabilities
        save_dir: directory to save the plots
        prefix: prefix for the saved files (e.g., 'train', 'valid', 'test')
    """
    print("before: ", true_labels.shape, pred_probs.shape)
    valid_mask = true_labels != -1
    true_labels = true_labels[valid_mask]
    pred_probs = pred_probs[valid_mask]
    print("after: ", true_labels.shape, pred_probs.shape)

    os.makedirs(save_dir, exist_ok=True)
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{prefix} ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{prefix}_roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(true_labels, pred_probs)
    pr_auc = auc(recall, precision)
    
    # Calculate random baseline (proportion of positive samples)
    baseline = np.sum(true_labels) / len(true_labels)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.axhline(y=baseline, color='navy', linestyle='--', label=f'Baseline ({baseline:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{prefix} Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{prefix}_pr_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Probability Distribution
    plt.figure(figsize=(12, 8))
    sns.kdeplot(data=pred_probs[true_labels == 0], label='Negative Class', fill=True)
    sns.kdeplot(data=pred_probs[true_labels == 1], label='Positive Class', fill=True)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title(f'{prefix} Prediction Probability Distribution')
    plt.legend()
    plt.grid(True)
    
    # Add text box with class proportions
    total = len(true_labels)
    pos_prop = np.sum(true_labels == 1) / total
    neg_prop = 1 - pos_prop
    
    plt.text(0.02, 0.98, 
             f'Class Distribution:\nNegative: {neg_prop:.1%}\nPositive: {pos_prop:.1%}',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.savefig(os.path.join(save_dir, f'{prefix}_prob_dist.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Save summary statistics
    stats = {
        'ROC_AUC': roc_auc,
        'PR_AUC': pr_auc,
        'Positive_Ratio': pos_prop,
        'Mean_Positive_Prob': np.mean(pred_probs[true_labels == 1]),
        'Mean_Negative_Prob': np.mean(pred_probs[true_labels == 0]),
        'Median_Positive_Prob': np.median(pred_probs[true_labels == 1]),
        'Median_Negative_Prob': np.median(pred_probs[true_labels == 0])
    }
    
    with open(os.path.join(save_dir, f'{prefix}_stats.txt'), 'w') as f:
        for key, value in stats.items():
            f.write(f'{key}: {value:.4f}\n')

def visualize_all_splits(train_labels, train_probs, valid_labels, valid_probs, 
                        test_labels, test_probs, save_dir):
    for labels, probs, prefix in [
        (train_labels, train_probs, 'train'),
        (valid_labels, valid_probs, 'valid'),
        (test_labels, test_probs, 'test')
    ]:
        plot_performance_curves(labels, probs, save_dir, prefix)

def identify_target_genes(adjacency_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify target genes (nodes with out-degree = 0) in the network.
    
    Args:
        adjacency_matrix: Network adjacency matrix where H[i,j] = 1 means gene i regulates gene j
    
    Returns:
        target_genes: Indices of target genes
        non_target_genes: Indices of non-target genes
    """
    # Calculate out-degrees for each gene
    out_degrees = np.sum(adjacency_matrix, axis=1)
    
    # Identify target genes (out-degree = 0)
    target_genes = np.where(out_degrees == 0)[0]
    non_target_genes = np.where(out_degrees > 0)[0]
    
    print(f"Found {len(target_genes)} target genes and {len(non_target_genes)} regulator genes")
    return target_genes, non_target_genes

def create_target_cv_splits(adjacency_matrix: np.ndarray, 
                                    target_genes: np.ndarray,
                                    k_folds: int = 5,
                                    seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create k-fold cross-validation splits based on target genes.
    Only use plausible negative edges (involving active regulators).
    
    Args:
        adjacency_matrix: Network adjacency matrix
        target_genes: Indices of target genes
        k_folds: Number of folds
        seed: Random seed
    """
    num_genes = adjacency_matrix.shape[0]
    splits = []
    rng = np.random.RandomState(seed)
    
    # Identify active regulators (genes that regulate at least one target)
    active_regulators = set()
    for target in target_genes:
        regulators = np.where(adjacency_matrix[:, target] == 1)[0]
        active_regulators.update(regulators)
    active_regulators = np.array(list(active_regulators))
    
    print(f"Number of active regulators: {len(active_regulators)}")
    
    # Split target genes into k folds
    shuffled_targets = target_genes.copy()
    rng.shuffle(shuffled_targets)
    target_folds = np.array_split(shuffled_targets, k_folds)
    
    for i in range(k_folds):
        train_adj = -np.ones_like(adjacency_matrix)
        test_adj = -np.ones_like(adjacency_matrix)
        
        # Current fold's target genes are test set
        test_targets = target_folds[i]
        train_targets = np.concatenate([target_folds[j] for j in range(k_folds) if j != i])
        
        # Training set: all edges except those to test target genes
        train_adj = adjacency_matrix.copy()
        train_adj[:, test_targets] = -1
        
        # For training set, only keep edges involving active regulators
        mask = np.zeros_like(train_adj, dtype=bool)
        for reg in active_regulators:
            mask[reg, :] = True
        train_adj[~mask] = -1
        
        # Test set: handle edges to test target genes (balanced)
        for target in test_targets:
            # Get positive edges
            positive_regulators = np.where(adjacency_matrix[:, target] == 1)[0]
            potential_negative_regulators = np.setdiff1d(active_regulators, positive_regulators)
            # if len(potential_negative_regulators) > len(positive_regulators):
            #     sampled_negatives = rng.choice(
            #         potential_negative_regulators,
            #         size=len(positive_regulators),
            #         replace=False
            #     )
            # else:
            #     sampled_negatives = potential_negative_regulators
            test_adj[positive_regulators, target] = 1
            # test_adj[sampled_negatives, target] = 0
            test_adj[potential_negative_regulators, target] = 0

        splits.append((train_adj, test_adj))
    
    return splits


def print_cv_split_statistics(splits: List[Tuple[np.ndarray, np.ndarray]], 
                            target_genes: np.ndarray):
    """Print statistics for each CV fold"""
    def count_edges(adj):
        mask = adj != -1
        if not np.any(mask):
            return 0, 0
        return np.sum(adj[mask] == 1), np.sum(adj[mask] == 0)
    
    print(f"\nCross-validation Statistics (k={len(splits)}):")
    print(f"Total target genes: {len(target_genes)}")
    print(f"Average targets per fold: {len(target_genes)/len(splits):.1f}")
    
    for i, (train_adj, test_adj) in enumerate(splits):
        train_pos, train_neg = count_edges(train_adj)
        test_pos, test_neg = count_edges(test_adj)
        
        print(f"\nFold {i+1}:")
        print(f"  Training Set:")
        print(f"    Positive edges: {train_pos}")
        print(f"    Negative edges: {train_neg}")
        print(f"    Positive ratio: {train_pos/(train_pos + train_neg):.3f}")
        print(f"  Test Set:")
        print(f"    Positive edges: {test_pos}")
        print(f"    Negative edges: {test_neg}")
        print(f"    Positive ratio: {test_pos/(test_pos + test_neg):.3f}")
        print(f"    Average regulators per target: {test_pos/np.sum(np.any(test_adj != -1, axis=0)):.1f}")

def save_kfold_data(dataset_id: int, ds_noisy: np.ndarray, cv_splits: List[Tuple[np.ndarray, np.ndarray]], 
                    target_genes: np.ndarray, base_dir='./data/splits'):
    """
    Save k-fold CV data with proper type conversion for JSON serialization
    """
    dataset_dir = os.path.join(base_dir, f'DS{dataset_id}')
    
    # Convert numpy types to native Python types for overall info
    split_info = {
        'num_folds': len(cv_splits),
        'num_genes': int(ds_noisy.shape[0]),
        'num_cells': int(ds_noisy.shape[1]),
        'num_target_genes': int(len(target_genes)),
        'target_genes': target_genes.tolist(),
        'timestamp': datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    info_file = os.path.join(dataset_dir, 'kfold_info.json')
    os.makedirs(dataset_dir, exist_ok=True)
    with open(info_file, 'w') as f:
        json.dump(split_info, f, indent=4)

    # Save data for each fold
    for fold_idx, (train_adj, test_adj) in enumerate(cv_splits):
        fold_dir = os.path.join(dataset_dir, f'fold_{fold_idx+1}')
        
        # Save train data
        train_dir = os.path.join(fold_dir, 'train')
        os.makedirs(train_dir, exist_ok=True)
        save_fold_split_data(train_dir, ds_noisy, train_adj)
        
        # Save test data
        test_dir = os.path.join(fold_dir, 'test')
        os.makedirs(test_dir, exist_ok=True)
        save_fold_split_data(test_dir, ds_noisy, test_adj)
        
        # Convert numpy types to native Python types for fold info
        fold_info = {
            'fold_number': int(fold_idx + 1),
            'num_train_positives': int(np.sum(train_adj == 1)),
            'num_train_negatives': int(np.sum(train_adj == 0)),
            'num_test_positives': int(np.sum(test_adj == 1)),
            'num_test_negatives': int(np.sum(test_adj == 0)),
            'train_target_genes': [int(x) for x in np.where(np.any(train_adj == 1, axis=0))[0]],
            'test_target_genes': [int(x) for x in np.where(np.any(test_adj == 1, axis=0))[0]]
        }
        
        with open(os.path.join(fold_dir, 'fold_info.json'), 'w') as f:
            json.dump(fold_info, f, indent=4)

def save_fold_split_data(split_dir: str, expression_data: np.ndarray, network_data: np.ndarray):
    """
    Save data for a specific split (train or test) within a fold
    """
    # Save expression data
    genes = [f'Gene_{i}' for i in range(expression_data.shape[0])]
    cells = [f'Cell_{i}' for i in range(expression_data.shape[1])]
    expression_df = pd.DataFrame(expression_data, index=genes, columns=cells)
    expression_df.to_csv(os.path.join(split_dir, 'ExpressionData.csv'))
    
    # Save network data
    positive_edges = np.argwhere(network_data == 1)
    negative_edges = np.argwhere(network_data == 0)
    
    # Save positive edges
    pos_edge_df = pd.DataFrame(positive_edges, columns=['Gene1', 'Gene2'])
    pos_edge_df['Gene1'] = pos_edge_df['Gene1'].apply(lambda x: f'Gene_{x}')
    pos_edge_df['Gene2'] = pos_edge_df['Gene2'].apply(lambda x: f'Gene_{x}')
    pos_edge_df.to_csv(os.path.join(split_dir, 'pos_refNetwork.csv'), index=False)
    
    # Save negative edges
    neg_edge_df = pd.DataFrame(negative_edges, columns=['Gene1', 'Gene2'])
    neg_edge_df['Gene1'] = neg_edge_df['Gene1'].apply(lambda x: f'Gene_{x}')
    neg_edge_df['Gene2'] = neg_edge_df['Gene2'].apply(lambda x: f'Gene_{x}')
    neg_edge_df.to_csv(os.path.join(split_dir, 'neg_refNetwork.csv'), index=False)
    
    # Save reference network (positive edges only)
    edge_df = pd.DataFrame(positive_edges, columns=['Gene1', 'Gene2'])
    edge_df['Gene1'] = edge_df['Gene1'].apply(lambda x: f'Gene_{x}')
    edge_df['Gene2'] = edge_df['Gene2'].apply(lambda x: f'Gene_{x}')
    edge_df.to_csv(os.path.join(split_dir, 'refNetwork.csv'), index=False)
    
    # Save numpy arrays
    np.save(os.path.join(split_dir, 'expression.npy'), expression_data)
    np.save(os.path.join(split_dir, 'network.npy'), network_data)

# %%
def run_kfold_experiments(gpu_id, dataset_id, embeddings, input_dim, output_dim, 
                         num_epochs, batch_size, learning_rate, loss_function, 
                         embedding_type, negative_ratio, temperature, result_queue,
                         k_folds=5, evaluation_strategy='gp', alpha=1.0, cl=True):
    """
    Wrapper to run k-fold CV experiments while preserving all evaluation strategies and metrics.
    """
    if torch.cuda.is_available() and gpu_id is not None:
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(gpu_id)
    else:
        device = torch.device('cpu')

    # Load dataset info and ground truth network
    datasets = get_datasets()
    dataset_info = next((dataset for dataset in datasets if dataset['dataset_id'] == dataset_id), None)
    if dataset_info is None:
        raise ValueError(f"Dataset ID {dataset_id} not found in available datasets")
    
    ds_clean, ds_noisy, gene_names, cell_names = load_data(dataset_info)
    
    if dataset_id < 1000:
        gt_grn_file = f'./SERGIO/data_sets/{dataset_info["folder_name"]}/gt_GRN.csv'
        H = load_ground_truth_grn(gt_grn_file)
    else:
        gt_grn_file = dataset_info['network_file']
        H = load_ground_truth_grn(gt_grn_file, gene_names=gene_names)

    # Create k-fold splits
    target_genes, non_target_genes = identify_target_genes(H)
    cv_splits = create_target_cv_splits(H, target_genes, k_folds=k_folds)
    print_cv_split_statistics(cv_splits, target_genes)

    # Store results for each fold
    fold_results = []
    best_model_overall = None
    best_valid_auc_overall = -1

    save_kfold_data(dataset_id, ds_noisy, cv_splits, target_genes)
    
    # Run experiment for each fold
    for fold_idx, (train_adj, test_adj) in enumerate(cv_splits):
        print(f"\nProcessing Fold {fold_idx + 1}/{k_folds}")    
        # Set seed for this fold
        fold_seed = 42 + fold_idx
        random.seed(fold_seed)
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)
        torch.cuda.manual_seed_all(fold_seed)
        
        # Train contrastive model
        if loss_function == 'SNN':
            model = train_snn(embeddings, train_adj, input_dim, output_dim, 
                            num_epochs, batch_size, learning_rate, negative_ratio, 
                            temperature, device)
        elif loss_function == 'CL':
            model = train_cl(embeddings, train_adj, input_dim, output_dim, 
                           num_epochs, batch_size, learning_rate, negative_ratio, device)
        elif loss_function == 'CEL':
            model = train_cel(embeddings, train_adj, input_dim, output_dim, 
                            num_epochs, batch_size, learning_rate, negative_ratio, device)
        elif loss_function == 'BCE':
            model = train_bce(embeddings, train_adj, input_dim, output_dim, 
                            num_epochs, batch_size, learning_rate, negative_ratio, device)
        
        # Get projected embeddings
        with torch.no_grad():
            embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
            projected_embeddings = model.projection(embeddings_tensor).cpu().numpy()

        # Evaluate based on strategy
        fold_dir = f'./results/cl/DS{dataset_id}/fold_{fold_idx+1}'
        os.makedirs(fold_dir, exist_ok=True)

        if evaluation_strategy == 'bgm':
            train_metrics, mca_train = evaluate_bayesian_model_bb(projected_embeddings, train_adj, H)
            visualize_clusters(projected_embeddings, mca_train, 
                             os.path.join(fold_dir, 'clusters_train.png'))
            clustering_metrics_train = compute_clustering_metrics(projected_embeddings, mca_train)
            test_metrics, mca_test = evaluate_bayesian_model_bb(projected_embeddings, test_adj, H)
            visualize_clusters(projected_embeddings, mca_test, 
                             os.path.join(fold_dir, 'clusters_test.png'))
            clustering_metrics_test = compute_clustering_metrics(projected_embeddings, mca_test)

        elif evaluation_strategy == 'gp':
            gp_model, likelihood, X_train, y_train = train_gp_model(
                projected_embeddings, train_adj, device, 
                model_type='directional', 
                direction_weight=1.0,
                inducing_points_num=500, num_epochs=200, 
                batch_size=1024, run_seed=fold_seed
            )
            train_metrics = evaluate_bayesian_model_gp(gp_model, likelihood, 
                                                     projected_embeddings, train_adj, H, device)
            test_metrics = evaluate_bayesian_model_gp(gp_model, likelihood, 
                                                    projected_embeddings, test_adj, H, device)
            # Save GP model
            torch.save({
                'model_state_dict': gp_model.state_dict(),
                'likelihood_state_dict': likelihood.state_dict(),
                'inducing_points': gp_model.variational_strategy.inducing_points.detach(),
                'train_x': X_train,
                'train_y': y_train
            }, os.path.join(fold_dir, f'gp_model_{embedding_type}_{loss_function}.pth'))

        elif evaluation_strategy == 'naive':
            prior_matrix = np.zeros_like(H, dtype=float)
            positive_edges = np.argwhere(train_adj == 1)
            negative_edges = np.argwhere(train_adj == 0)
            prior_matrix[positive_edges[:, 0], positive_edges[:, 1]] = 1.0
            prior_matrix[negative_edges[:, 0], negative_edges[:, 1]] = 0.0
            train_metrics = evaluate_bayesian_model2(projected_embeddings, train_adj, H, alpha, prior_matrix)
            test_metrics = evaluate_bayesian_model2(projected_embeddings, test_adj, H, alpha, prior_matrix)

        else:
            train_metrics = evaluate_contrastive_model(model, embeddings, train_adj, loss_function, device)
            test_metrics = evaluate_contrastive_model(model, embeddings, test_adj, loss_function, device)

        print("train_metrics: ", train_metrics)
        print("test_metrics: ", test_metrics)

        # Save visualization plots
        plt.figure(figsize=(12, 6))
        plt.plot(train_metrics['recall_curve'], train_metrics['precision_curve'], 
                label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Fold {fold_idx + 1} Train Precision-Recall curve')
        plt.legend()
        plt.savefig(os.path.join(fold_dir, 'train_pr_curve.png'))
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.plot(test_metrics['recall_curve'], test_metrics['precision_curve'], 
                label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Fold {fold_idx + 1} Test Precision-Recall curve')
        plt.legend()
        plt.savefig(os.path.join(fold_dir, 'test_pr_curve.png'))
        plt.close()

        # Save models and embeddings
        torch.save(model.state_dict(), 
                  os.path.join(fold_dir, f'contrastive_model_{embedding_type}_{loss_function}.pth'))
        np.save(os.path.join(fold_dir, f'projected_embeddings_{embedding_type}_{loss_function}.npy'), 
                projected_embeddings)

        # Track best model
        if test_metrics['auc_roc'] > best_valid_auc_overall:
            best_valid_auc_overall = test_metrics['auc_roc']
            best_model_overall = model

        # Store fold results
        fold_results.append({
            'fold': fold_idx + 1,
            'train_metrics': {
                'auc_roc': train_metrics['auc_roc'],
                'auc_pr': train_metrics['auc_pr'],
                'f1_score': train_metrics['f1_score'],
                'mcc': train_metrics['mcc'],
                'k_precision': train_metrics['k_precision'],
                'k_recall': train_metrics['k_recall']
            },
            'test_metrics': {
                'auc_roc': test_metrics['auc_roc'],
                'auc_pr': test_metrics['auc_pr'],
                'f1_score': test_metrics['f1_score'],
                'mcc': test_metrics['mcc'],
                'k_precision': test_metrics['k_precision'],
                'k_recall': test_metrics['k_recall']
            }
        })

    # Aggregate results across folds
    metric_keys = ['auc_roc', 'auc_pr', 'f1_score', 'mcc', 'k_precision', 'k_recall']
    final_metrics = {
        'train': {'means': {}, 'stds': {}, 'ci': {}},
        'test': {'means': {}, 'stds': {}, 'ci': {}}
    }

    for split in ['train', 'test']:
        for metric in metric_keys:
            values = [fold[f'{split}_metrics'][metric] for fold in fold_results]
            final_metrics[split]['means'][metric] = np.mean(values)
            final_metrics[split]['stds'][metric] = np.std(values)
            ci = np.percentile(values, [2.5, 97.5])
            final_metrics[split]['ci'][metric] = ci

    for split in final_metrics:
        for metric_type in final_metrics[split]:
            for key in final_metrics[split][metric_type]:
                if isinstance(final_metrics[split][metric_type][key], np.ndarray):
                    final_metrics[split][metric_type][key] = final_metrics[split][metric_type][key].tolist()
                elif isinstance(final_metrics[split][metric_type][key], np.floating):
                    final_metrics[split][metric_type][key] = float(final_metrics[split][metric_type][key])
    for fold in fold_results:
        for metric_set in ['train_metrics', 'test_metrics']:
            for key, value in fold[metric_set].items():
                if isinstance(value, np.ndarray):
                    fold[metric_set][key] = value.tolist()
                elif isinstance(value, np.floating):
                    fold[metric_set][key] = float(value)
    result_queue.put({
        'dataset_id': dataset_id,
        'input_dim': input_dim,
        'output_dim': output_dim,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'loss_function': loss_function,
        'embedding_type': embedding_type,
        'negative_ratio': negative_ratio,
        'temperature': temperature if loss_function == 'SNN' else None,
        'train_auc': f"{final_metrics['train']['means']['auc_roc']:.4f} +- {final_metrics['train']['stds']['auc_roc']:.4f}",
        'test_auc': f"{final_metrics['test']['means']['auc_roc']:.4f} +- {final_metrics['test']['stds']['auc_roc']:.4f}",
        'train_pr_auc': f"{final_metrics['train']['means']['auc_pr']:.4f} +- {final_metrics['train']['stds']['auc_pr']:.4f}",
        'test_pr_auc': f"{final_metrics['test']['means']['auc_pr']:.4f} +- {final_metrics['test']['stds']['auc_pr']:.4f}",
        'train_f1': f"{final_metrics['train']['means']['f1_score']:.4f} +- {final_metrics['train']['stds']['f1_score']:.4f}",
        'test_f1': f"{final_metrics['test']['means']['f1_score']:.4f} +- {final_metrics['test']['stds']['f1_score']:.4f}",
        'train_k_precision': f"{final_metrics['train']['means']['k_precision']:.4f} +- {final_metrics['train']['stds']['k_precision']:.4f}",
        'test_k_precision': f"{final_metrics['test']['means']['k_precision']:.4f} +- {final_metrics['test']['stds']['k_precision']:.4f}",
        'train_mcc': f"{final_metrics['train']['means']['mcc']:.4f} +- {final_metrics['train']['stds']['mcc']:.4f}",
        'test_mcc': f"{final_metrics['test']['means']['mcc']:.4f} +- {final_metrics['test']['stds']['mcc']:.4f}",
        'train_k_recall': f"{final_metrics['train']['means']['k_recall']:.4f} +- {final_metrics['train']['stds']['k_recall']:.4f}",
        'test_k_recall': f"{final_metrics['test']['means']['k_recall']:.4f} +- {final_metrics['test']['stds']['k_recall']:.4f}",
        'train_labels': train_metrics['true_labels'],        
        'train_scores': train_metrics['probabilities'],
        'test_labels': test_metrics['true_labels'],
        'test_scores': test_metrics['probabilities'],
        'model_state_dict': best_model_overall.state_dict(),
        'final_metrics': final_metrics,
        'fold_results': fold_results,
        'projected_embeddings': projected_embeddings,
    })

def run_experiment(gpu_id, dataset_id, train_ratio, output_dim, num_epochs, batch_size, learning_rate,
                   loss_function, embedding_type, negative_ratio, temperature,
                   result_queue, evaluation_strategy, cl_mode, model_type, n_runs=1, alpha=1.0):
    if torch.cuda.is_available() and gpu_id is not None:
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(gpu_id)
    else:
        device = torch.device('cpu')
    # if dataset_id in DATASET_INFO_MAPPING:
    #     net_type, cell_type_str = DATASET_INFO_MAPPING[dataset_id]
    #     organism, scale_str = cell_type_str.split()  # e.g. "hESC", "500"
    #     base_dir = f"../bni/GENELink/processed/{net_type}/{organism} {scale_str}"
    #     exp_file  = os.path.join(base_dir, "BL--ExpressionData.csv")
    #     train_file= os.path.join(base_dir, "Train_set.csv")
    #     val_file  = os.path.join(base_dir, "Validation_set.csv")
    #     test_file = os.path.join(base_dir, "Test_set.csv")
    #     data_dict = load_transformer_data_for_contrastive(
    #         exp_file=exp_file,
    #         train_file=train_file,
    #         test_file=test_file,
    #         val_file=val_file
    #     )
    #     embeddings = data_dict['features']
    #     if not os.path.exists(f'./results/cl/DS{dataset_id}'):
    #         os.makedirs(f'./results/cl/DS{dataset_id}')
    #     np.save(f'./results/cl/DS{dataset_id}/fa_embeddings.npy', embeddings)
    #     input_dim = embeddings.shape[1]
    #     G_train = data_dict['G_train']
    #     G_valid = data_dict['G_valid']
    #     G_test  = data_dict['G_test']
        
    #     # Construct adjacency H (like your original code)
    #     num_genes = embeddings.shape[0]
    #     H = np.zeros((num_genes, num_genes))
    #     train_pos = np.argwhere(G_train == 1)
    #     train_neg = np.argwhere(G_train == 0)
    #     H[tuple(zip(*train_pos))] = 1
    #     H[tuple(zip(*train_neg))] = 0
    #     valid_pos = np.argwhere(G_valid == 1)
    #     valid_neg = np.argwhere(G_valid == 0)
    #     H[tuple(zip(*valid_pos))] = 1 
    #     H[tuple(zip(*valid_neg))] = 0
    #     test_pos = np.argwhere(G_test == 1)
    #     test_neg = np.argwhere(G_test == 0)
    #     H[tuple(zip(*test_pos))] = 1
    #     H[tuple(zip(*test_neg))] = 0
    #     print("G_train: ", G_train, "shape: ", G_train.shape)
    #     print("G_valid: ", G_valid, "shape: ", G_valid.shape)
    #     print("G_test: ", G_test, "shape: ", G_test.shape)
    #     print("H: ", H, "shape: ", H.shape)
    #     print("embeddings: ", embeddings)
    #     print("embeddings.shape: ", embeddings.shape)
    #     print(f"Loaded dataset {dataset_id}: net_type={net_type}, cell_type={cell_type_str}")
    #     print("G_train shape:", G_train.shape)
    # else:


    datasets = get_datasets()
    dataset_info = next((dataset for dataset in datasets if dataset['dataset_id'] == dataset_id), None)
    print(f"Dataset Info: {dataset_info}")
    if dataset_info is None:
        raise ValueError(f"Dataset ID {dataset_id} not found in available datasets")
    dataset_id = dataset_info['dataset_id']
    print(f"\nProcessing Dataset {dataset_id}...")
    ds_clean, ds_noisy, gene_names, cell_names = load_data(dataset_info)
    num_genes = ds_noisy.shape[0]
    num_cells = ds_noisy.shape[1]
    embeddings = generate_embeddings(ds_noisy, embedding_type=embedding_type, n_components=64)
    print(embeddings[:5])
    input_dim = embeddings.shape[1]
    if dataset_id < 1000:
        # simulated data
        gt_grn_file = f'./SERGIO/data_sets/{dataset_info["folder_name"]}/gt_GRN.csv'
        H = load_ground_truth_grn(gt_grn_file)
    else:
        # real data
        print(f"gene_names: {len(gene_names)}")
        gt_grn_file = dataset_info['network_file']
        H = load_ground_truth_grn(gt_grn_file, gene_names=gene_names)
    
    valid_ratio = (1-train_ratio)/2
    test_ratio = (1-train_ratio)/2
    
    G = sample_partial_grn(H, sample_ratio=train_ratio + valid_ratio)
    G_train, G_valid = split_train_valid(G, train_ratio=train_ratio/(train_ratio+valid_ratio))
    G_test = get_test_set(H, G)
    # G_train_valid = G_train + G_valid

    split_info = {
        'train_ratio': train_ratio,
        'valid_ratio': valid_ratio,
        'test_ratio': test_ratio,
        'num_genes': num_genes,
        'num_cells': num_cells,
        'timestamp': datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    save_split_info(dataset_id, train_ratio, split_info)

    for split_name, exp_data, net_data in [
        ('train', ds_noisy, G_train),
        ('valid', ds_noisy, G_valid),
        ('test', ds_noisy, G_test)
    ]:
        save_split_data(dataset_id, split_name, exp_data, net_data, gene_names, cell_names)
        split_dir = os.path.join('./data/splits', f'DS{dataset_id}', split_name)
        normalized_data = pd.DataFrame(exp_data, index=gene_names, columns=cell_names)
        normalized_data.to_csv(os.path.join(split_dir, 'bin-normalized-matrix.csv'))

    
    all_train_metrics = []
    all_valid_metrics = []
    all_test_metrics = []
    best_model = None
    best_valid_auc = -1
    run_times = []
    for run in range(n_runs):
        # Set different seed for each run
        run_seed = 42 + run
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        torch.cuda.manual_seed_all(run_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        seed_everything(run_seed)
        model = None
        run_start = time.time()
        if cl_mode is None:
            projected_embeddings = embeddings
        elif cl_mode == "standard":
            if loss_function == 'SNN':
                model = train_snn(embeddings, G_train, input_dim, output_dim, num_epochs, batch_size, learning_rate, negative_ratio, temperature, device)
            elif loss_function == 'CL':
                model = train_cl(embeddings, G_train, input_dim, output_dim, num_epochs, batch_size, learning_rate, negative_ratio, device)
            elif loss_function == 'CEL':
                model = train_cel(embeddings, G_train, input_dim, output_dim, num_epochs, batch_size, learning_rate, negative_ratio, device)
            elif loss_function == 'BCE':
                model = train_bce(embeddings, G_train, input_dim, output_dim, num_epochs, batch_size, learning_rate, negative_ratio, device)
        
            with torch.no_grad():
                embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
                projected_embeddings = model.projection(embeddings_tensor).cpu().numpy()
        elif cl_mode == "directional":
            if loss_function == 'SNN':
                model = train_snn_directional(embeddings, G_train, input_dim, output_dim, num_epochs, batch_size, learning_rate, negative_ratio, temperature, device)
        
            with torch.no_grad():
                embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
                projected_embeddings = model.get_embeddings(embeddings_tensor, combine_mode="avg").cpu().numpy()

        if evaluation_strategy == 'naive':
            prior_matrix = np.zeros_like(H, dtype=float)
            positive_edges = np.argwhere(G_train == 1)
            negative_edges = np.argwhere(G_train == 0)
            unknown_edges = np.argwhere(G_train == -1)
            prior_matrix[positive_edges[:, 0], positive_edges[:, 1]] = 1.0
            prior_matrix[negative_edges[:, 0], negative_edges[:, 1]] = 0.0
            num_positive = len(positive_edges)
            num_negative = len(negative_edges)
            positive_rate = num_positive / (num_positive + num_negative)
            prior_matrix[unknown_edges[:, 0], unknown_edges[:, 1]] = positive_rate
        if evaluation_strategy == 'bgm':
            train_metrics, mca_train = evaluate_bayesian_model_bb(projected_embeddings, G_train, H)
            visualize_clusters(projected_embeddings, mca_train, f'./results/clusters/DS{dataset_id}_clusters_train.png')
            clustering_metrics_train = compute_clustering_metrics(projected_embeddings, mca_train)
            valid_metrics, mca_valid = evaluate_bayesian_model_bb(projected_embeddings, G_valid, H)
            visualize_clusters(projected_embeddings, mca_valid, f'./results/clusters/DS{dataset_id}_clusters_valid.png')
            clustering_metrics_valid = compute_clustering_metrics(projected_embeddings, mca_valid)
            test_metrics, mca_test = evaluate_bayesian_model_bb(projected_embeddings, G_test, H)
            visualize_clusters(projected_embeddings, mca_test, f'./results/clusters/DS{dataset_id}_clusters_test.png')
            clustering_metrics_test = compute_clustering_metrics(projected_embeddings, mca_test)
        elif evaluation_strategy == 'gp':
            gp_model, likelihood, X_train, y_train = train_gp_model(
                                                    projected_embeddings,
                                                    G_train,
                                                    device,
                                                    model_type=model_type,  # or 'standard'
                                                    direction_weight=1.0,      # only used for directional
                                                    inducing_points_num=100, num_epochs=200, batch_size=1024, run_seed=run_seed)
            train_metrics = evaluate_bayesian_model_gp(gp_model, likelihood, projected_embeddings, G_train, H, device)
            valid_metrics = evaluate_bayesian_model_gp(gp_model, likelihood, projected_embeddings, G_valid, H, device)
            test_metrics = evaluate_bayesian_model_gp(gp_model, likelihood, projected_embeddings, G_test, H, device)
            print("train_metrics: ", train_metrics)
            print("valid_metrics: ", valid_metrics)
            print("test_metrics: ", test_metrics)
            model_save_path = f'./results/cl/DS{dataset_id}/gp_model_{embedding_type}_{loss_function}.pth'
            torch.save({
                'model_state_dict': gp_model.state_dict(),
                'likelihood_state_dict': likelihood.state_dict(),
                'inducing_points': gp_model.variational_strategy.inducing_points.detach(),
                'train_x': X_train,
                'train_y': y_train
            }, model_save_path)
        elif evaluation_strategy == 'naive':
            train_metrics = evaluate_bayesian_model2(projected_embeddings, G_train, H, alpha, prior_matrix)
            valid_metrics = evaluate_bayesian_model2(projected_embeddings, G_valid, H, alpha, prior_matrix)
            test_metrics = evaluate_bayesian_model2(projected_embeddings, G_test, H, alpha, prior_matrix)
        elif cl:
            train_metrics = evaluate_contrastive_model(model, embeddings, G_train, loss_function, device)
            valid_metrics = evaluate_contrastive_model(model, embeddings, G_valid, loss_function, device)
            test_metrics = evaluate_contrastive_model(model, embeddings, G_test, loss_function, device)
        else:
            train_metrics = evaluate_without_posterior(projected_embeddings, G_train, H)
            valid_metrics = evaluate_without_posterior(projected_embeddings, G_valid, H)
            test_metrics = evaluate_without_posterior(projected_embeddings, G_test, H)

        run_end = time.time()
        one_run_time = run_end - run_start
        run_times.append(one_run_time)
        # print(1)
        
        # print("Train Metrics: ", train_metrics)
        # plot precision and recall curve
        plt.figure(figsize=(12, 6))
        plt.plot(train_metrics['recall_curve'], train_metrics['precision_curve'], label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Train Precision-Recall curve')
        plt.legend()
        if not os.path.exists(f'./results/cl/DS{dataset_id}'):
            os.makedirs(f'./results/cl/DS{dataset_id}')
        plt.savefig(f'./results/cl/DS{dataset_id}/train_precision_recall_curve.png')
        # print(2)
        # print("Validation Metrics: ", valid_metrics)
        plt.figure(figsize=(12, 6))
        plt.plot(valid_metrics['recall_curve'], valid_metrics['precision_curve'], label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Validation Precision-Recall curve')
        plt.legend()
        plt.savefig(f'./results/cl/DS{dataset_id}/valid_precision_recall_curve.png')
        # print(3)
        # print("Test Metrics: ", test_metrics)
        plt.figure(figsize=(12, 6))
        plt.plot(test_metrics['recall_curve'], test_metrics['precision_curve'], label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Test Precision-Recall curve')
        plt.legend()
        plt.savefig(f'./results/cl/DS{dataset_id}/test_precision_recall_curve.png')
        # print(4)
        all_train_metrics.append(train_metrics)
        all_valid_metrics.append(valid_metrics)
        all_test_metrics.append(test_metrics)
        # print(5)
        if valid_metrics['auc_roc'] > best_valid_auc:
            best_valid_auc = valid_metrics['auc_roc']
            best_model = model
        # print(6)
        metric_keys = ['auc_roc', 'auc_pr', 'f1_score', 'mcc', 'k_precision', 'k_recall']
        final_metrics = {
            'train': {'means': {}, 'stds': {}},
            'valid': {'means': {}, 'stds': {}},
            'test': {'means': {}, 'stds': {}}
        }
        # print(7)
        for metric in metric_keys:
            # Training metrics
            values = [m[metric] for m in all_train_metrics]
            final_metrics['train']['means'][metric] = np.mean(values)
            final_metrics['train']['stds'][metric] = np.std(values)
            
            # Validation metrics
            values = [m[metric] for m in all_valid_metrics]
            final_metrics['valid']['means'][metric] = np.mean(values)
            final_metrics['valid']['stds'][metric] = np.std(values)
            
            # Test metrics
            values = [m[metric] for m in all_test_metrics]
            final_metrics['test']['means'][metric] = np.mean(values)
            final_metrics['test']['stds'][metric] = np.std(values)
        
        # Use the best model for final embeddings
        if not cl_mode:
            projected_embeddings = embeddings
        elif cl_mode == "standard":
            best_model.eval()
            with torch.no_grad():
                embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
                projected_embeddings = best_model.projection(embeddings_tensor).cpu().numpy()
        elif cl_mode == "directional":
            if loss_function == 'SNN':
                model = train_snn_directional(embeddings, G_train, input_dim, output_dim, num_epochs, batch_size, learning_rate, negative_ratio, temperature, device)
        
            with torch.no_grad():
                projected_embeddings = model.get_embeddings(embeddings_tensor, combine_mode="avg").cpu().numpy()
    
    time_mean = np.mean(run_times)
    time_std = np.std(run_times)
    time_str = f"{time_mean:.2f} +- {time_std:.2f}"

    save_dir = f'./results/visualizations/DS{dataset_id}_yunwei/{embedding_type}_{loss_function}'
    # print(8)
    # visualize_all_splits(
    #     train_metrics['true_labels'], train_metrics['probabilities'],
    #     valid_metrics['true_labels'], valid_metrics['probabilities'],
    #     test_metrics['true_labels'], test_metrics['probabilities'],
    #     save_dir
    # )
    # print(9)
    result_queue.put({
        'dataset_id': dataset_id,
        'cl': cl_mode,
        'evaluation_strategy': evaluation_strategy,
        'embedding_type': embedding_type,
        'train_ratio': train_ratio,
        'input_dim': input_dim,
        'output_dim': output_dim,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'loss_function': loss_function,
        'negative_ratio': negative_ratio,
        'temperature': temperature if loss_function == 'SNN' else None,
        'train_auc': f"{final_metrics['train']['means']['auc_roc']:.4f} +- {final_metrics['train']['stds']['auc_roc']:.4f}",
        'valid_auc': f"{final_metrics['valid']['means']['auc_roc']:.4f} +- {final_metrics['valid']['stds']['auc_roc']:.4f}",
        'test_auc': f"{final_metrics['test']['means']['auc_roc']:.4f} +- {final_metrics['test']['stds']['auc_roc']:.4f}",
        'train_pr_auc': f"{final_metrics['train']['means']['auc_pr']:.4f} +- {final_metrics['train']['stds']['auc_pr']:.4f}",
        'valid_pr_auc': f"{final_metrics['valid']['means']['auc_pr']:.4f} +- {final_metrics['valid']['stds']['auc_pr']:.4f}",
        'test_pr_auc': f"{final_metrics['test']['means']['auc_pr']:.4f} +- {final_metrics['test']['stds']['auc_pr']:.4f}",
        'test_time': time_str,
        'train_f1': f"{final_metrics['train']['means']['f1_score']:.4f} +- {final_metrics['train']['stds']['f1_score']:.4f}",
        'valid_f1': f"{final_metrics['valid']['means']['f1_score']:.4f} +- {final_metrics['valid']['stds']['f1_score']:.4f}",
        'test_f1': f"{final_metrics['test']['means']['f1_score']:.4f} +- {final_metrics['test']['stds']['f1_score']:.4f}",
        'train_k_precision': f"{final_metrics['train']['means']['k_precision']:.4f} +- {final_metrics['train']['stds']['k_precision']:.4f}",
        'valid_k_precision': f"{final_metrics['valid']['means']['k_precision']:.4f} +- {final_metrics['valid']['stds']['k_precision']:.4f}",
        'test_k_precision': f"{final_metrics['test']['means']['k_precision']:.4f} +- {final_metrics['test']['stds']['k_precision']:.4f}",
        'train_mcc': f"{final_metrics['train']['means']['mcc']:.4f} +- {final_metrics['train']['stds']['mcc']:.4f}",
        'valid_mcc': f"{final_metrics['valid']['means']['mcc']:.4f} +- {final_metrics['valid']['stds']['mcc']:.4f}",
        'test_mcc': f"{final_metrics['test']['means']['mcc']:.4f} +- {final_metrics['test']['stds']['mcc']:.4f}",
        'train_k_recall': f"{final_metrics['train']['means']['k_recall']:.4f} +- {final_metrics['train']['stds']['k_recall']:.4f}",
        'valid_k_recall': f"{final_metrics['valid']['means']['k_recall']:.4f} +- {final_metrics['valid']['stds']['k_recall']:.4f}",
        'test_k_recall': f"{final_metrics['test']['means']['k_recall']:.4f} +- {final_metrics['test']['stds']['k_recall']:.4f}",
        # 'train_confusion_matrix': train_metrics['confusion_matrix'],
        # 'valid_confusion_matrix': valid_metrics['confusion_matrix'],
        # 'test_confusion_matrix': test_metrics['confusion_matrix'],
        # 'train_precision_curve': train_metrics['precision_curve'],
        # 'train_recall_curve': train_metrics['recall_curve'],
        # 'train_thresholds': train_metrics['thresholds'],
        # 'valid_precision_curve': valid_metrics['precision_curve'],
        # 'valid_recall_curve': valid_metrics['recall_curve'],
        # 'valid_thresholds': valid_metrics['thresholds'],
        # 'test_precision_curve': test_metrics['precision_curve'],
        # 'test_recall_curve': test_metrics['recall_curve'],
        # 'test_thresholds': test_metrics['thresholds'],
        'train_labels': train_metrics['true_labels'],
        'train_scores': train_metrics['probabilities'],
        'valid_labels': valid_metrics['true_labels'],
        'valid_scores': valid_metrics['probabilities'],
        'test_labels': test_metrics['true_labels'],
        'test_scores': test_metrics['probabilities'],
        'model_state_dict': best_model.state_dict() if best_model is not None else None,
        'projected_embeddings': projected_embeddings,
        # 'train_clustering_metrics': clustering_metrics_train,
        # 'valid_clustering_metrics': clustering_metrics_valid,
        # 'test_clustering_metrics': clustering_metrics_test
        # 'model': gp_model if evaluation_strategy == 'gp' else None,
        # 'likelihood': likelihood if evaluation_strategy == 'gp' else None,
        # 'X_train': X_train if evaluation_strategy == 'gp' else None,
        # 'y_train': y_train if evaluation_strategy == 'gp' else None
    })
    print(9.01)

def logger_process(result_queue, num_experiments):
    experiments_logged = 0
    while experiments_logged < num_experiments:
        try:
            result = result_queue.get(timeout=1)
            print("Got result, logging...")
            log_dir = log_experiment(result)
            experiments_logged += 1
        except queue.Empty:
            continue

import random

def process_batch(batch):
    processes = []
    for args in batch:
        gpu_id, embedding_type, contrastive_learning, model_type, posterior_estimation, output_dim, num_epochs, train_ratio, dataset_id, batch_size, learning_rate, loss_function, negative_ratio, temperature = args

        p = multiprocessing.Process(target=run_experiment, args=(
            gpu_id, dataset_id, train_ratio, output_dim, num_epochs, batch_size, learning_rate,
            loss_function, embedding_type, negative_ratio, temperature, result_queue, posterior_estimation, contrastive_learning, model_type))
        # p = multiprocessing.Process(target=run_kfold_experiments, args=(
        #     gpu_id, dataset_id, embeddings, input_dim, output_dim, num_epochs, batch_size, learning_rate,
        #     loss_function, embedding_type, negative_ratio, temperature, result_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

# if __name__ == '__main__':
#     multiprocessing.set_start_method('spawn', force=True)
#     # Set random seed even though this does not guarantee complete reproducibility
#     random.seed(42)
#     np.random.seed(42)
#     torch.manual_seed(42)
#     torch.cuda.manual_seed_all(42)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     seed_everything(42)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     num_gpus = torch.cuda.device_count()

#     # single parameters
#     embedding_types = ['phate']  # ['raw', 'pca', 'fa']
#     contrastive_learning_options = ["directional"] # None, "standard", "directional"
#     posterior_estimations = ['naive'] # 'none', 'naive', 'bgm', 'gp'
#     model_types = ['standard'] # 'standard', 'directional' only active with 'gp'
#     output_dims = [16] # [16, 32, 64, 128, 256]
#     num_epochs_list = [50] # [5, 10, 50, 100, 200]
#     train_ratios = [0.8]
#     # dataset_ids = [k for k in DATASET_INFO_MAPPING.keys() if str(k).startswith("170") or  str(k).startswith("180") ] # 1,2,3,1001,1002,1003,1004,1005,1006, 1501, 1502, 1503, 1504  , 1005, 1006
#     # dataset_ids = [k for k in DATASET_INFO_MAPPING.keys()]
#     dataset_ids = [1701]
#     dataset_ids = sorted(dataset_ids)
#     batch_sizes = [32] # [16, 32, 64]
#     learning_rates = [1e-3] # [1e-3, 5e-3, 1e-4, 5e-4]
#     loss_functions = ['SNN']  # ['BCE', 'CEL', 'CL', 'SNN']
#     negative_ratios = [5] # [5, 10, 20, 50, 100, 200, 500]
#     temperatures = [1]

#     # embedding_types = ['pca']  # ['pca', 'vae']
#     # output_dims = [16, 32, 64] # [16, 32, 64, 128, 256]
#     # num_epochs_list = [10, 50, 200] # [5, 10, 50, 100, 200]
#     # batch_sizes = [32, 64] # [16, 32, 64]z
#     # learning_rates = [1e-3, 1e-4, 5e-4] # [1e-3, 5e-3, 1e-4, 5e-4]
#     # loss_functions = ['SNN']  # ['BCE', 'CEL', 'CL', 'SNN']
#     # negative_ratios = [10, 50, 100, 500] # [5, 10, 20, 50, 100, 200, 500]
#     # temperatures = [0.1, 1.0, 10, 100]  # [0.1, 0.5, 1.0, 2.0, 5.0]
# # I have some progress. I implemented the logic of combing the gene expersion and use only non-cell-type-specific network for evaluation. The first issue is that the result dataset is too large, there are 2543 genes which leads to 4 times the size in the pos&neg edge list.
# # So I dropped the number of genes from 1000 to 100. That makes our dataset has similar size as our DS1 to 3, 374 pos edges and 83349 neg edges in the valid and test sets.
# # Our model turns out to get very decent performance, compariable to what we have for DS1 to 3. We define these two datasets (human and mouse) as DS1007 and DS1008.
# # Then I want to double check w
#     experiments = []
#     for embedding_type, contrastive_learning, model_type, posterior_estimation, output_dim, num_epochs, train_ratio, dataset_id, batch_size, learning_rate, loss_function, negative_ratio in product(
#         embedding_types, contrastive_learning_options, model_types, posterior_estimations, output_dims, num_epochs_list, train_ratios, dataset_ids, batch_sizes, learning_rates, loss_functions, negative_ratios
#     ):
#         if loss_function == 'SNN':
#             for temperature in temperatures:
#                 experiments.append((embedding_type, contrastive_learning, model_type, posterior_estimation, output_dim, num_epochs, train_ratio, dataset_id, batch_size, learning_rate, loss_function, negative_ratio, temperature))
#         else:
#             experiments.append((embedding_type, contrastive_learning, model_type, posterior_estimation, output_dim, num_epochs, train_ratio, dataset_id, batch_size, learning_rate, loss_function, negative_ratio, None))

#     print(f"Running {len(experiments)} experiments on {num_gpus} GPUs.")
#     print(experiments)

#     experiments_with_gpus = []
#     for idx, exp in enumerate(experiments):
#         if num_gpus > 0:
#             gpu_id = idx % num_gpus
#         else:
#             gpu_id = None
#         experiments_with_gpus.append((gpu_id,) + exp)
    
#     result_queue = multiprocessing.Queue()
#     logger = multiprocessing.Process(target=logger_process, args=(result_queue, len(experiments)))
#     logger.start()
    
#     batch_process_size = max(num_gpus * 2, 1)
#     for i in range(0, len(experiments_with_gpus), batch_process_size):
#         batch = experiments_with_gpus[i:i+batch_process_size]
#         process_batch(batch)

#     logger.join()
#     print("All experiments completed and logged.")


def register_custom_adata(h5ad_path: str,
                          net_csv: str,
                          disease: str,
                          tissue: str,
                          dataset_name: str = "custom_h5ad",
                          starting_id: int = 9000):
    taken = {d['dataset_id'] for d in get_datasets()}
    ds_id = max(taken | {starting_id - 1}) + 1

    _GLOBAL_CUSTOM.append({
        'dataset_id'   : ds_id,
        'dataset_name' : dataset_name,
        'h5ad_file'    : h5ad_path,
        'disease'      : disease,
        'tissue'       : tissue,
        'network_file' : net_csv,
    })
    print(f"Registered h5ad dataset «{dataset_name}» as DS{ds_id}")


from pathlib import Path

register_custom_adata(
    h5ad_path="datasets/63ff2c52-cb63-44f0-bac3-d0b33373e312.h5ad",
    net_csv   ="data/net.csv",
    disease   ="Crohn disease",
    tissue    ="lamina propria of mucosa of colon",
    dataset_name="human_colon"
)

custom_id = max(d['dataset_id'] for d in get_datasets())

run_experiment(
    gpu_id=0,
    dataset_id=1702,
    train_ratio=0.8,
    output_dim=16,
    num_epochs=50,
    batch_size=32,
    learning_rate=1e-3,
    loss_function='SNN',
    embedding_type='phate',
    negative_ratio=5,
    temperature=1.0,
    result_queue=Queue(),
    evaluation_strategy='naive',
    cl_mode="directional",
    model_type='standard',
)
