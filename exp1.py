import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, f1_score
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls import VariationalELBO
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
import pandas as pd
import anndata
import matplotlib.pyplot as plt
from pathlib import Path

class DirectionalRBFKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True
    
    def __init__(self, src_weight=1.0, tgt_weight=1.0, dir_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.src_weight = nn.Parameter(torch.tensor(src_weight))
        self.tgt_weight = nn.Parameter(torch.tensor(tgt_weight))
        self.dir_weight = nn.Parameter(torch.tensor(dir_weight))

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            return torch.ones(x1.size(0), dtype=x1.dtype, device=x1.device)

        D = (x1.shape[-1] - 1) // 2
        dir1, dir2 = x1[..., -1], x2[..., -1]
        
        src1, tgt1 = x1[..., :D], x1[..., D:-1]
        src2, tgt2 = x2[..., :D], x2[..., D:-1]
        
        dist_src = self.src_weight * (src1.unsqueeze(-2) - src2.unsqueeze(-3)).pow(2).sum(-1)
        dist_tgt = self.tgt_weight * (tgt1.unsqueeze(-2) - tgt2.unsqueeze(-3)).pow(2).sum(-1)
        dist_dir = self.dir_weight * (dir1.unsqueeze(-1) - dir2.unsqueeze(-2)).pow(2)
        
        return torch.exp(-0.5 * (dist_src + dist_tgt + dist_dir) / self.lengthscale.pow(2))

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
        with torch.no_grad():
            src_emb = self.projection_source(x)
            tgt_emb = self.projection_target(x)
            if combine_mode == "avg":
                return 0.5*(src_emb + tgt_emb)
            elif combine_mode == "cat":
                return torch.cat([src_emb, tgt_emb], dim=-1)
            else:
                return src_emb

class SoftNearestNeighborLoss(nn.Module):
    def __init__(self, temperature=10., cos_distance=True):
        super().__init__()
        self.temperature = temperature
        self.cos_distance = cos_distance

    def pairwise_cos_distance(self, A, B):
        query_embeddings = F.normalize(A, dim=1)
        key_embeddings = F.normalize(B, dim=1)
        distances = 1 - torch.matmul(query_embeddings, key_embeddings.T)
        return distances

    def forward(self, embeddings, labels):
        batch_size = embeddings.shape[0]
        eps = 1e-9

        if self.cos_distance:
            pairwise_dist = self.pairwise_cos_distance(embeddings, embeddings)
        else:
            pairwise_dist = torch.cdist(embeddings, embeddings, p=2)

        pairwise_dist = pairwise_dist / self.temperature
        negexpd = torch.exp(-pairwise_dist)

        pairs_y = torch.broadcast_to(labels, (batch_size, batch_size))
        mask = pairs_y == torch.transpose(pairs_y, 0, 1)
        mask = mask.float()

        ones = torch.ones([batch_size, batch_size], dtype=torch.float32).to(embeddings.device)
        dmask = ones - torch.eye(batch_size, dtype=torch.float32).to(embeddings.device)

        alcn = torch.sum(torch.multiply(negexpd, dmask), dim=1)
        sacn = torch.sum(torch.multiply(negexpd, mask), dim=1)

        loss = -torch.log((sacn+eps)/alcn).mean()
        return loss

class CustomBeacon:
    def __init__(self, gene_embeddings=None, n_components=32, embedding_type="phate", 
                 projection_dim=32, device="cuda"):
        self.gene_embeddings = gene_embeddings
        self.n_components = n_components
        self.embedding_type = embedding_type
        self.projection_dim = projection_dim
        self.device = device
        self.projection_model = None
        self.gp_model = None
        self.gp_likelihood = None
        self.trained = False
        
    def fit(self, gene_expr, prior_network, contrastive_epochs=10, gp_epochs=50,
            batch_size=64, lr=0.001, negative_ratio=10, temperature=10.0,
            gp_model_type='directional'):
        
        from sklearn.preprocessing import StandardScaler
        
        # Use original embeddings or generate new ones
        if self.gene_embeddings is None:
            print("Generating gene embeddings...")
            if isinstance(gene_expr, pd.DataFrame):
                # Use the generate_embeddings function from your existing code
                from sklearn.decomposition import PCA, FactorAnalysis
                X = gene_expr.values
                X = StandardScaler().fit_transform(X)
                
                if self.embedding_type.lower() == "pca":
                    model = PCA(n_components=self.n_components, random_state=0)
                    self.gene_embeddings = model.fit_transform(X)
                elif self.embedding_type.lower() in ["fa", "factoranalysis"]:
                    model = FactorAnalysis(n_components=self.n_components, random_state=0)
                    self.gene_embeddings = model.fit_transform(X)
                elif self.embedding_type.lower() == "phate":
                    try:
                        import phate
                        phate_op = phate.PHATE(n_components=self.n_components, random_state=0)
                        self.gene_embeddings = phate_op.fit_transform(X)
                    except ImportError:
                        print("PHATE not installed. Using PCA instead.")
                        model = PCA(n_components=self.n_components, random_state=0)
                        self.gene_embeddings = model.fit_transform(X)
                else:
                    model = PCA(n_components=self.n_components, random_state=0)
                    self.gene_embeddings = model.fit_transform(X)
            else:
                # Use the function from your existing code
                # Keeping this part simple for integration
                print("Using provided embeddings directly.")
                self.gene_embeddings = gene_expr
        
        num_genes = self.gene_embeddings.shape[0]
        input_dim = self.gene_embeddings.shape[1]
        
        # Create adjacency matrix from prior network
        adjacency_matrix = np.zeros((num_genes, num_genes), dtype=int)
        
        # Convert network to list of tuples if it's a DataFrame
        if isinstance(prior_network, pd.DataFrame):
            if 'source' in prior_network.columns and 'target' in prior_network.columns:
                network_edges = list(zip(prior_network['source'], prior_network['target']))
            elif 'Gene1' in prior_network.columns and 'Gene2' in prior_network.columns:
                network_edges = list(zip(prior_network['Gene1'], prior_network['Gene2']))
            else:
                raise ValueError("Prior network DataFrame should have 'source'/'target' or 'Gene1'/'Gene2' columns")
        else:
            network_edges = prior_network
            
        # Get gene indices
        gene_indices = {}
        for i in range(num_genes):
            gene_indices[i] = i  # Assuming indices already match
            
        # Fill adjacency matrix
        for edge in network_edges:
            source, target = edge
            # Convert gene names to indices if strings provided
            if isinstance(source, str) and isinstance(target, str):
                if hasattr(gene_expr, 'index'):
                    source = gene_expr.index.get_loc(source) if source in gene_expr.index else None
                    target = gene_expr.index.get_loc(target) if target in gene_expr.index else None
                else:
                    source = None
                    target = None
                
            if source is not None and target is not None:
                adjacency_matrix[source, target] = 1
                adjacency_matrix[target, source] = 1  # Make symmetric
        
        # Train contrastive model (using your existing function)
        self.projection_model = self._train_directional_model(
            self.gene_embeddings, 
            adjacency_matrix, 
            input_dim, 
            self.projection_dim,
            contrastive_epochs, 
            batch_size, 
            lr, 
            negative_ratio, 
            temperature
        )
        
        # Get projected embeddings
        projected_embeddings = self._get_projected_embeddings(self.gene_embeddings)
        
        # Train GP model
        self.gp_model, self.gp_likelihood = self._train_gp_model(
            projected_embeddings, 
            adjacency_matrix,
            model_type=gp_model_type,
            num_epochs=gp_epochs,
            batch_size=batch_size
        )
        
        self.trained = True
        return self
    
    def _train_directional_model(self, embeddings, adjacency_matrix, input_dim, projection_dim, 
                               num_epochs, batch_size, learning_rate, 
                               negative_ratio, temperature):
        from numpy.random import shuffle
        
        model = DirectionalContrastiveModel(input_dim, projection_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = SoftNearestNeighborLoss(temperature=temperature)
        
        positive_edges = np.argwhere(adjacency_matrix == 1)
        negative_edges = np.argwhere(adjacency_matrix == 0)
        num_positive_edges = len(positive_edges)
        num_batches = max(1, num_positive_edges // batch_size)
        
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
        
        print(f"Training contrastive model for {num_epochs} epochs...")
        for epoch in tqdm(range(num_epochs)):
            total_loss = 0
            model.train()
            
            # Shuffle edges for this epoch
            shuffle(positive_edges)
            shuffle(negative_edges)
            
            for i in range(num_batches):
                # Get positive edges for this batch
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_positive_edges)
                batch_positive = positive_edges[start_idx:end_idx]
                
                # Sample negative edges
                num_negatives = len(batch_positive) * negative_ratio
                if num_negatives > len(negative_edges):
                    batch_negative = negative_edges[np.random.choice(len(negative_edges), size=num_negatives, replace=True)]
                else:
                    batch_negative = negative_edges[np.random.choice(len(negative_edges), size=num_negatives, replace=False)]
                
                # Combine and shuffle
                batch_edges = np.concatenate([batch_positive, batch_negative])
                batch_labels = np.concatenate([np.ones(len(batch_positive)), np.zeros(len(batch_negative))])
                
                shuffle_idx = np.random.permutation(len(batch_edges))
                batch_edges = batch_edges[shuffle_idx]
                batch_labels = batch_labels[shuffle_idx]
                
                # Get embeddings
                x1 = embeddings_tensor[batch_edges[:, 0]]
                x2 = embeddings_tensor[batch_edges[:, 1]]
                labels = torch.tensor(batch_labels, dtype=torch.float32).to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                proj1, proj2 = model(x1, x2)
                embeddings_batch = torch.cat([proj1, proj2], dim=0)
                labels_batch = torch.cat([labels, labels], dim=0)
                
                loss = criterion(embeddings_batch, labels_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / num_batches
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        return model
    
    def _get_projected_embeddings(self, embeddings):
        self.projection_model.eval()
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            projected = self.projection_model.get_embeddings(embeddings_tensor, combine_mode="avg")
            return projected.cpu().numpy()
    
    def _train_gp_model(self, projected_embeddings, adjacency_matrix, 
                      model_type='directional', direction_weight=1.0,
                      inducing_points_num=500, num_epochs=50, 
                      batch_size=1024):
        
        # Prepare training data
        train_edges = np.argwhere((adjacency_matrix == 1) | (adjacency_matrix == 0))
        train_labels = adjacency_matrix[train_edges[:, 0], train_edges[:, 1]].astype(int)

        emb_i = torch.tensor(projected_embeddings[train_edges[:, 0]], dtype=torch.float32)
        emb_j = torch.tensor(projected_embeddings[train_edges[:, 1]], dtype=torch.float32)
        
        if model_type == 'directional':
            direction = torch.ones(len(train_edges), 1, dtype=torch.float32)
            X_train = torch.cat([emb_i, emb_j, direction], dim=1)
        else:
            X_train = torch.cat([emb_i, emb_j], dim=1)

        X_train = X_train.to(self.device)
        y_train = torch.tensor(train_labels, dtype=torch.float32).to(self.device)
        
        # Initialize inducing points
        inducing_points = X_train[:min(inducing_points_num, X_train.shape[0])].clone()

        # Initialize model
        model = GPClassificationModel(
            inducing_points=inducing_points,
            model_type=model_type,
            direction_weight=direction_weight
        ).to(self.device)
        likelihood = BernoulliLikelihood().to(self.device)

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

        print(f"Training GP model for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            epoch_loss = 0
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = model(x_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
        
        return model, likelihood
    
    def predict(self, gene_pairs=None):
        """
        Make predictions for gene pairs using the trained models.
        
        Args:
            gene_pairs: numpy array of gene pairs to predict, shape (n_pairs, 2)
                        If None, predict all possible pairs
        
        Returns:
            pandas DataFrame with columns [source, target, score]
        """
        import pandas as pd
        
        if not self.trained:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        num_genes = self.gene_embeddings.shape[0]
        projected_embeddings = self._get_projected_embeddings(self.gene_embeddings)
        
        if gene_pairs is None:
            # If no specific pairs provided, predict all possible pairs
            all_pairs = []
            for i in range(num_genes):
                for j in range(i+1, num_genes):  # Upper triangular only
                    all_pairs.append((i, j))
            gene_pairs = np.array(all_pairs)
        
        # Prepare features
        emb_i = torch.tensor(projected_embeddings[gene_pairs[:, 0]], dtype=torch.float32)
        emb_j = torch.tensor(projected_embeddings[gene_pairs[:, 1]], dtype=torch.float32)
        
        if self.gp_model.model_type == 'directional':
            direction = torch.ones(len(gene_pairs), 1, dtype=torch.float32)
            X_eval = torch.cat([emb_i, emb_j, direction], dim=1)
        else:
            X_eval = torch.cat([emb_i, emb_j], dim=1)
        
        X_eval = X_eval.to(self.device)
        
        # Evaluate
        self.gp_model.eval()
        self.gp_likelihood.eval()
        
        batch_size = 1024
        predictions = []
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(0, X_eval.shape[0], batch_size):
                X_batch = X_eval[i:i+batch_size]
                observed_pred = self.gp_likelihood(self.gp_model(X_batch))
                pred_batch = observed_pred.mean.cpu().numpy()
                predictions.append(pred_batch)
        
        predictions = np.concatenate(predictions)
        
        # Create results dataframe
        results = pd.DataFrame({
            'source': gene_pairs[:, 0],
            'target': gene_pairs[:, 1],
            'score': predictions
        })
        
        return results

def run_beacon_experiment(gene_expr, prior_network, phate_op=None, phate_emb=None, 
                          n_negative_samples=None, random_seed=42, **beacon_params):
    """
    Efficient implementation of leave-one-out validation using CustomBeacon.
    This fully replaces the external BEACON library with a more efficient approach.
    
    The function signature is kept the same for compatibility.
    """
    import numpy as np
    import pandas as pd
    from itertools import product
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Format prior network for validation
    if 'type' not in prior_network.columns:
        prior_network = prior_network.copy()
        prior_network['type'] = 1
    
    # Get all genes in the network
    all_genes = set(prior_network['source'].unique()) | set(prior_network['target'].unique())
    
    # Check if all network genes are in the expression data
    expr_genes = gene_expr.index if hasattr(gene_expr, 'index') else np.arange(gene_expr.shape[0])
    missing_genes = all_genes - set(expr_genes)
    if missing_genes:
        print(f"Warning: {len(missing_genes)} genes in the network are missing from expression data")
        
        # Filter out edges with missing genes
        prior_network = prior_network[
            prior_network['source'].isin(expr_genes) & 
            prior_network['target'].isin(expr_genes)
        ].reset_index(drop=True)
        
        all_genes = set(prior_network['source'].unique()) | set(prior_network['target'].unique())
        print(f"After filtering, {len(all_genes)} genes and {len(prior_network)} edges remain")
    
    # Generate negative edges
    if n_negative_samples is None:
        n_negative_samples = len(prior_network)
    
    genes_list = list(all_genes)
    all_pairs = [(g1, g2) for g1, g2 in product(genes_list, genes_list) if g1 != g2]
    
    positive_pairs = set(zip(prior_network['source'], prior_network['target']))
    negative_candidates = list(set(all_pairs) - positive_pairs)
    
    if len(negative_candidates) < n_negative_samples:
        print(f"Warning: only {len(negative_candidates)} negative candidates available")
        n_negative_samples = len(negative_candidates)
    
    sampled_indices = np.random.choice(len(negative_candidates), n_negative_samples, replace=False)
    negative_edges = pd.DataFrame([
        {'source': negative_candidates[i][0], 'target': negative_candidates[i][1], 'type': 0}
        for i in sampled_indices
    ])
    
    # Combine positive and negative edges
    all_edges = pd.concat([
        prior_network[['source', 'target', 'type']],
        negative_edges[['source', 'target', 'type']]
    ]).reset_index(drop=True)
    
    # ------------------------------------------------------------------------
    # STEP 1: Generate embeddings ONCE (use provided embeddings if available)
    # ------------------------------------------------------------------------

    X = gene_expr.values if hasattr(gene_expr, 'values') else gene_expr
    X = StandardScaler().fit_transform(X)
    import phate
    phate_op = phate.PHATE(n_components=32, random_state=0)
    embeddings = phate_op.fit_transform(X)
 
    
    # ------------------------------------------------------------------------
    # STEP 2: Train the contrastive model ONCE
    # ------------------------------------------------------------------------
    print("Training contrastive model once for all validation runs...")
    input_dim = embeddings.shape[1]
    projection_dim = 32  # Default value
    device = "cuda"
    
    # Create adjacency matrix
    num_genes = len(all_genes)
    adjacency_matrix = -np.ones((num_genes, num_genes), dtype=int)
    
    # Map gene names to indices if needed
    gene_to_idx = {}
    for i, gene in enumerate(all_genes):
        gene_to_idx[gene] = i
    
    # Fill adjacency matrix with training data
    for _, row in all_edges.iterrows():
        source = row['source']
        target = row['target']
        edge_type = row['type']
        
        if isinstance(source, str) and isinstance(target, str):
            source_idx = gene_to_idx[source]
            target_idx = gene_to_idx[target]
        else:
            source_idx = source
            target_idx = target
            
        adjacency_matrix[source_idx, target_idx] = edge_type
    
    # Train directional contrastive model
    contrastive_model = DirectionalContrastiveModel(input_dim, projection_dim).to(device)
    contrastive_optimizer = torch.optim.Adam(contrastive_model.parameters(), lr=0.001)
    contrastive_criterion = SoftNearestNeighborLoss(temperature=10.0)
    
    # Train contrastive model with all edges
    contrastive_model.train()
    positive_edges = np.argwhere(adjacency_matrix == 1)
    negative_edges = np.argwhere(adjacency_matrix == 0)
    num_positive_edges = len(positive_edges)
    batch_size = 64
    num_batches = max(1, num_positive_edges // batch_size)
    
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
    
    # One training loop for the contrastive model
    num_epochs = 5
    for epoch in range(num_epochs):
        total_loss = 0
        
        # Shuffle edges for this epoch
        np.random.shuffle(positive_edges)
        np.random.shuffle(negative_edges)
        
        for i in range(num_batches):
            # Get positive edges for this batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_positive_edges)
            batch_positive = positive_edges[start_idx:end_idx]
            
            # Sample negative edges
            num_negatives = len(batch_positive) * 10
            if num_negatives > len(negative_edges):
                batch_negative = negative_edges[np.random.choice(len(negative_edges), size=num_negatives, replace=True)]
            else:
                batch_negative = negative_edges[np.random.choice(len(negative_edges), size=num_negatives, replace=False)]
            
            # Combine and shuffle
            batch_edges = np.concatenate([batch_positive, batch_negative])
            batch_labels = np.concatenate([np.ones(len(batch_positive)), np.zeros(len(batch_negative))])
            
            shuffle_idx = np.random.permutation(len(batch_edges))
            batch_edges = batch_edges[shuffle_idx]
            batch_labels = batch_labels[shuffle_idx]
            
            # Get embeddings
            x1 = embeddings_tensor[batch_edges[:, 0]]
            x2 = embeddings_tensor[batch_edges[:, 1]]
            labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)
            
            # Forward pass
            contrastive_optimizer.zero_grad()
            proj1, proj2 = contrastive_model(x1, x2)
            embeddings_batch = torch.cat([proj1, proj2], dim=0)
            labels_batch = torch.cat([labels, labels], dim=0)
            
            loss = contrastive_criterion(embeddings_batch, labels_batch)
            loss.backward()
            contrastive_optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Get projected embeddings
    contrastive_model.eval()
    with torch.no_grad():
        projected_embeddings = contrastive_model.get_embeddings(embeddings_tensor).cpu().numpy()
    
    # ------------------------------------------------------------------------
    # STEP 3: Perform efficient leave-one-out validation
    # ------------------------------------------------------------------------
    print(f"Running leave-one-out validation with {len(all_edges)} edges...")
    results = {
        'true_edges': [],
        'predictions': [],
        'edge_types': [],
    }
    
    # Create indices mapping for source/target to graph indices
    source_to_idx = {}
    target_to_idx = {}
    for i, gene in enumerate(all_genes):
        source_to_idx[gene] = i
        target_to_idx[gene] = i
    
    from tqdm import tqdm
    
    # Use a shared set of inducing points from the full dataset for efficiency
    num_edges = len(all_edges)
    train_edges = np.argwhere((adjacency_matrix == 1) | (adjacency_matrix == 0))
    
    # Prepare all features for GP model, reuse for all leave-one-out runs
    emb_i = torch.tensor(projected_embeddings[train_edges[:, 0]], dtype=torch.float32)
    emb_j = torch.tensor(projected_embeddings[train_edges[:, 1]], dtype=torch.float32)
    direction = torch.ones(len(train_edges), 1, dtype=torch.float32)
    X_all = torch.cat([emb_i, emb_j, direction], dim=1).to(device)
    
    # Select inducing points once
    inducing_points_num = min(500, X_all.shape[0])
    inducing_indices = np.random.choice(X_all.shape[0], inducing_points_num, replace=False)
    inducing_points = X_all[inducing_indices].clone()
    
    # Now run leave-one-out validation
    for i in tqdm(range(num_edges)):
        left_out_edge = all_edges.iloc[i]
        source, target = left_out_edge['source'], left_out_edge['target']
        true_label = left_out_edge['type']
        
        # Convert to numeric indices if needed
        if isinstance(source, str) and isinstance(target, str):
            if hasattr(gene_expr, 'index'):
                source_idx = gene_expr.index.get_loc(source) if source in gene_expr.index else None
                target_idx = gene_expr.index.get_loc(target) if target in gene_expr.index else None
            else:
                source_idx = source_to_idx.get(source)
                target_idx = target_to_idx.get(target)
        else:
            source_idx = source
            target_idx = target
        
        # Skip if indices are invalid
        if source_idx is None or target_idx is None:
            continue
        
        try:
            # Mask the current edge for training
            mask = ~((train_edges[:, 0] == source_idx) & (train_edges[:, 1] == target_idx))
            X_train = X_all[mask]
            y_train = torch.tensor(adjacency_matrix[train_edges[mask, 0], train_edges[mask, 1]], 
                                  dtype=torch.float32).to(device)
            
            # Initialize GP model
            model = GPClassificationModel(
                inducing_points=inducing_points,
                model_type='directional',
                direction_weight=1.0
            ).to(device)
            likelihood = BernoulliLikelihood().to(device)
            
            # Train GP model (with a small number of epochs for efficiency)
            model.train()
            likelihood.train()
            
            optimizer = torch.optim.AdamW([
                {'params': model.parameters()},
                {'params': likelihood.parameters()},
            ], lr=0.005)
            
            mll = VariationalELBO(likelihood, model, num_data=y_train.size(0))
            
            # Use a smaller batch size and fewer epochs for LOO validation
            train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=256, shuffle=True)
            
            # Train for fewer epochs in leave-one-out
            num_epochs = 5
            for epoch in range(num_epochs):
                for x_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    output = model(x_batch)
                    loss = -mll(output, y_batch)
                    loss.backward()
                    optimizer.step()
            
            # Predict for the left-out edge
            model.eval()
            likelihood.eval()
            
            # Create features for the held-out edge
            left_out_emb_i = torch.tensor(projected_embeddings[source_idx], dtype=torch.float32).unsqueeze(0)
            left_out_emb_j = torch.tensor(projected_embeddings[target_idx], dtype=torch.float32).unsqueeze(0)
            left_out_dir = torch.ones(1, 1, dtype=torch.float32)
            X_test = torch.cat([left_out_emb_i, left_out_emb_j, left_out_dir], dim=1).to(device)
            
            # Make prediction
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = likelihood(model(X_test))
                pred_prob = observed_pred.mean.cpu().numpy()[0]
            
            # Store results
            results['true_edges'].append((source, target))
            results['predictions'].append(pred_prob)
            results['edge_types'].append(true_label)
            
        except Exception as e:
            print(f"Error processing edge {source} -> {target}: {e}")
            continue
    
    # Calculate metrics
    results['edge_types'] = np.array(results['edge_types'])
    results['predictions'] = np.array(results['predictions'])
    
    try:
        # Calculate ROC curve and AUC
        from sklearn.metrics import roc_curve, precision_recall_curve, auc
        
        fpr, tpr, _ = roc_curve(results['edge_types'], results['predictions'])
        results['auroc'] = auc(fpr, tpr)
        
        # Calculate Precision-Recall curve and AUC
        precision, recall, _ = precision_recall_curve(results['edge_types'], results['predictions'])
        results['aupr'] = auc(recall, precision)
        
        # Store ROC and PR curves
        results['roc_curve'] = {'fpr': fpr, 'tpr': tpr}
        results['pr_curve'] = {'precision': precision, 'recall': recall}
        
        print(f"AUROC: {results['auroc']:.4f}, AUPR: {results['aupr']:.4f}")
    except Exception as e:
        print(f"Error calculating metrics: {e}")
    
    return results











def load_network(network_path="data/net.csv"):
    """Load the gene regulatory network from CSV"""
    net_df = pd.read_csv(network_path)
    
    # Format the network data for the CustomBeacon model
    formatted_network = []
    for _, row in net_df.iterrows():
        formatted_network.append({
            'source': row['Gene1'],
            'target': row['Gene2'],
            'type': 1  # Positive edge
        })
    
    return pd.DataFrame(formatted_network)

def load_expression_data(dataset_id, disease, tissue, imputation_method=None):
    """
    Load gene expression data - either raw or imputed
    
    Args:
        dataset_id: ID of the dataset
        disease: Disease name
        tissue: Tissue name
        imputation_method: If None, use raw data; otherwise use the imputed data
        
    Returns:
        DataFrame with genes as rows and cells as columns
    """
    # Load the original AnnData object
    adata = anndata.read_h5ad(f"./datasets/{dataset_id}.h5ad")
    
    # Filter to the specific disease and tissue
    mask = (adata.obs["disease"] == disease) & (adata.obs["tissue"] == tissue)
    adata_subset = adata[mask]
    
    # Get raw data if no imputation method is specified
    if imputation_method is None:
        expr_matrix = adata_subset.X.toarray() if hasattr(adata_subset.X, 'toarray') else adata_subset.X
        print(f"Using raw expression data: {expr_matrix.shape}")
    else:
        # Load the imputed data
        imputed_path = os.path.join("output", imputation_method, dataset_id, disease, tissue + ".npy")
        if not os.path.exists(imputed_path):
            raise FileNotFoundError(f"Imputed data not found at {imputed_path}")
        
        expr_matrix = np.load(imputed_path)
        print(f"Using {imputation_method} imputed data: {expr_matrix.shape}")
    
    # Create DataFrame with gene names as index and cells as columns
    gene_names = adata_subset.var_names.tolist()
    expr_df = pd.DataFrame(expr_matrix.T, columns=adata_subset.obs_names, index=gene_names)
    
    return expr_df

def run_network_inference(expr_df, prior_network, run_name, n_runs=2):
    """
    Run network inference with the specified expression data multiple times and average results
    
    Args:
        expr_df: Expression data DataFrame
        prior_network: Network prior knowledge
        run_name: Name for this run
        n_runs: Number of times to run inference to average results
        
    Returns:
        Dictionary with averaged results
    """
    # Store results from each run
    all_auroc = []
    all_aupr = []
    
    for run in range(1, n_runs+1):
        try:
            print(f"\nRunning CustomBeacon with {run_name} (run {run}/{n_runs})...")
            beacon = CustomBeacon()
            beacon.fit(
                gene_expr=expr_df,
                prior_network=prior_network,
                contrastive_epochs=10,
                gp_epochs=20
            )
            
            # Run leave-one-out validation
            print(f"Running leave-one-out validation for {run_name} (run {run}/{n_runs})...")
            loo_results = run_beacon_experiment(
                gene_expr=expr_df,
                prior_network=prior_network,
                n_negative_samples=len(prior_network)
            )
            
            # Store this run's results
            all_auroc.append(loo_results['auroc'])
            all_aupr.append(loo_results['aupr'])
            
            # Print intermediate results
            print(f"Run {run}/{n_runs} for {run_name}:")
            print(f"AUROC: {loo_results['auroc']:.4f}")
            print(f"AUPR: {loo_results['aupr']:.4f}")
            
        except Exception as e:
            print(f"Error in run {run} for {run_name}: {e}")
    
    # Calculate average metrics
    if len(all_auroc) > 0:
        avg_auroc = np.mean(all_auroc)
        avg_aupr = np.mean(all_aupr)
        std_auroc = np.std(all_auroc)
        std_aupr = np.std(all_aupr)
        
        # Print averaged results
        print(f"\nAveraged results for {run_name} ({len(all_auroc)} successful runs):")
        print(f"AUROC: {avg_auroc:.4f} ± {std_auroc:.4f}")
        print(f"AUPR: {avg_aupr:.4f} ± {std_aupr:.4f}")
        
        # Save metrics
        with open(f"results/{run_name}_metrics.txt", "w") as f:
            f.write(f"AUROC: {avg_auroc:.4f} ± {std_auroc:.4f}\n")
            f.write(f"AUPR: {avg_aupr:.4f} ± {std_aupr:.4f}\n")
            f.write(f"Individual runs:\n")
            for i, (auroc, aupr) in enumerate(zip(all_auroc, all_aupr)):
                f.write(f"Run {i+1}: AUROC={auroc:.4f}, AUPR={aupr:.4f}\n")
        
        return {
            'auroc': avg_auroc,
            'aupr': avg_aupr,
            'auroc_std': std_auroc,
            'aupr_std': std_aupr,
            'all_auroc': all_auroc,
            'all_aupr': all_aupr
        }
    else:
        print(f"No successful runs for {run_name}")
        return None

def compare_all_methods(dataset_id, disease, tissue, n_runs=2):
    """Compare network inference performance across all methods"""
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Load network data
    prior_network = load_network()
    print(f"Loaded network with {len(prior_network)} edges")
    
    # Methods to compare
    methods = ["SAUCIE", "MAGIC", "deepImpute", "scScope", "scVI", "knn_smoothing"]
    
    # Check if we have consensus data
    consensus_dir = os.path.join("output", "consensus")
    if os.path.exists(consensus_dir):
        methods.append("consensus")
    
    # First run with raw data
    raw_expr = load_expression_data(dataset_id, disease, tissue, imputation_method=None)
    raw_results = run_network_inference(raw_expr, prior_network, "raw", n_runs=n_runs)
    
    # Run with each imputation method
    all_results = {"raw": raw_results}
    
    for method in methods:
        try:
            # Special case for consensus if it's stored differently
            if method == "consensus":
                # Adjust path as needed for your consensus data
                consensus_path = os.path.join("output", "consensus", dataset_id, disease, tissue + ".npy")
                if not os.path.exists(consensus_path):
                    print(f"Consensus data not found at {consensus_path}, skipping")
                    continue
            
            imputed_expr = load_expression_data(dataset_id, disease, tissue, imputation_method=method)
            method_results = run_network_inference(imputed_expr, prior_network, method, n_runs=n_runs)
            all_results[method] = method_results
        except Exception as e:
            print(f"Error processing method {method}: {e}")
    
    # Compare all results
    compare_results(all_results)
    return all_results

def compare_results(results_dict):
    """Create comparison plots and tables with error bars"""
    methods = []
    auroc_values = []
    aupr_values = []
    auroc_errors = []
    aupr_errors = []
    
    for method, results in results_dict.items():
        if results:
            methods.append(method)
            auroc_values.append(results['auroc'])
            aupr_values.append(results['aupr'])
            auroc_errors.append(results.get('auroc_std', 0))
            aupr_errors.append(results.get('aupr_std', 0))
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Method': methods,
        'AUROC': auroc_values,
        'AUROC_std': auroc_errors,
        'AUPR': aupr_values,
        'AUPR_std': aupr_errors
    })
    
    # Save to CSV
    comparison_df.to_csv("results/method_comparison.csv", index=False)
    
    # Create plots with error bars
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    bars = plt.bar(methods, auroc_values, yerr=auroc_errors, capsize=5, color='skyblue', alpha=0.8)
    plt.title('AUROC Comparison', fontsize=14)
    plt.xlabel('Method', fontsize=12)
    plt.ylabel('AUROC', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on the bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{auroc_values[i]:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.subplot(2, 1, 2)
    bars = plt.bar(methods, aupr_values, yerr=aupr_errors, capsize=5, color='lightgreen', alpha=0.8)
    plt.title('AUPR Comparison', fontsize=14)
    plt.xlabel('Method', fontsize=12)
    plt.ylabel('AUPR', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on the bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{aupr_values[i]:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig("results/method_comparison.png", dpi=300, bbox_inches='tight')
    
    # Create heatmap for easy visual comparison
    plt.figure(figsize=(10, 6))
    
    # Create a matrix for the heatmap
    heatmap_data = np.array([auroc_values, aupr_values])
    
    im = plt.imshow(heatmap_data, cmap='viridis')
    
    # Add labels
    plt.yticks([0, 1], ['AUROC', 'AUPR'])
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Score Value')
    
    # Add text annotations on the heatmap
    for i in range(2):
        for j in range(len(methods)):
            value = heatmap_data[i, j]
            text = plt.text(j, i, f'{value:.4f}',
                          ha="center", va="center", color="white" if value > 0.5 else "black")
    
    plt.title('Performance Metrics Across Methods')
    plt.tight_layout()
    plt.savefig("results/method_comparison_heatmap.png", dpi=300, bbox_inches='tight')
    
    plt.close('all')
    
    print("\nResults comparison:")
    print(comparison_df.to_string(index=False))

if __name__ == "__main__":
    # Set these to your specific dataset
    dataset_id = "63ff2c52-cb63-44f0-bac3-d0b33373e312"
    disease = "Crohn disease"
    tissue = "lamina propria of mucosa of colon"
    
    # Set the number of runs for averaging
    n_runs = 2
    
    all_results = compare_all_methods(dataset_id, disease, tissue, n_runs=n_runs)
    
    # Save the full results object for later analysis
    import pickle
    with open("results/full_results.pkl", "wb") as f:
        pickle.dump(all_results, f)

