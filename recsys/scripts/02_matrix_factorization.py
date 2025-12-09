"""
02_matrix_factorization.py
===========================
Matrix Factorization Recommender System

Implements:
- SVD (Singular Value Decomposition)
- ALS (Alternating Least Squares)
- Funk SVD (Gradient Descent-based)

Features:
- Interactive graph visualization showing latent space
- Comprehensive statistical reports
- Hyperparameter exploration
- Multiple dataset support
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from pyvis.network import Network
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create output directories
os.makedirs("../outputs/matrix_factorization", exist_ok=True)
os.makedirs("../visualizations", exist_ok=True)
os.makedirs("../reports/matrix_factorization", exist_ok=True)


class MatrixFactorization:
    """
    Matrix Factorization Recommender System
    
    Decomposes user-item matrix into latent factor matrices:
    R ≈ U @ V^T
    
    Where:
    - U: User latent factors (n_users × n_factors)
    - V: Item latent factors (n_items × n_factors)
    """
    
    def __init__(
        self,
        n_factors: int = 50,
        method: str = 'svd',  # 'svd', 'als', 'funk_svd'
        n_epochs: int = 20,
        learning_rate: float = 0.005,
        regularization: float = 0.02,
        random_state: int = 42
    ):
        self.n_factors = n_factors
        self.method = method
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.random_state = random_state
        
        # Model parameters
        self.user_factors = None  # U matrix
        self.item_factors = None  # V matrix
        self.user_biases = None
        self.item_biases = None
        self.global_mean = 0
        
        # Mappings
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        
        # Training history
        self.training_history = []
        
        # Statistics
        self.stats = {
            'n_users': 0,
            'n_items': 0,
            'n_ratings': 0,
            'sparsity': 0
        }
        
        np.random.seed(random_state)
    
    def fit(self, ratings_df: pd.DataFrame) -> 'MatrixFactorization':
        """
        Fit the matrix factorization model
        
        Args:
            ratings_df: DataFrame with columns [user_id, item_id, rating]
        """
        print(f"\n🔧 Fitting Matrix Factorization ({self.method.upper()})...")
        print(f"   n_factors: {self.n_factors}")
        print(f"   n_epochs: {self.n_epochs}")
        
        # Create mappings
        unique_users = ratings_df['user_id'].unique()
        unique_items = ratings_df['item_id'].unique()
        
        self.user_mapping = {u: i for i, u in enumerate(unique_users)}
        self.item_mapping = {it: i for i, it in enumerate(unique_items)}
        self.reverse_user_mapping = {i: u for u, i in self.user_mapping.items()}
        self.reverse_item_mapping = {i: it for it, i in self.item_mapping.items()}
        
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        # Global mean
        self.global_mean = ratings_df['rating'].mean()
        
        # Build sparse matrix
        row_indices = ratings_df['user_id'].map(self.user_mapping).values
        col_indices = ratings_df['item_id'].map(self.item_mapping).values
        ratings = ratings_df['rating'].values
        
        R = csr_matrix(
            (ratings, (row_indices, col_indices)),
            shape=(n_users, n_items)
        )
        
        # Update stats
        self.stats['n_users'] = n_users
        self.stats['n_items'] = n_items
        self.stats['n_ratings'] = len(ratings_df)
        self.stats['sparsity'] = 1 - len(ratings_df) / (n_users * n_items)
        
        # Fit based on method
        if self.method == 'svd':
            self._fit_svd(R)
        elif self.method == 'als':
            self._fit_als(R, ratings_df)
        elif self.method == 'funk_svd':
            self._fit_funk_svd(ratings_df)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        print(f"   ✅ Model fitted: {n_users} users, {n_items} items")
        print(f"   📊 Latent space: {self.n_factors} dimensions")
        
        return self
    
    def _fit_svd(self, R: csr_matrix):
        """Fit using truncated SVD"""
        print("   Computing SVD decomposition...")
        
        # Center the matrix
        R_dense = R.toarray()
        R_centered = R_dense - self.global_mean
        R_centered[R_dense == 0] = 0  # Keep missing values as 0
        
        # Compute SVD
        k = min(self.n_factors, min(R.shape) - 1)
        U, sigma, Vt = svds(csr_matrix(R_centered), k=k)
        
        # Sort by singular values (descending)
        idx = np.argsort(-sigma)
        sigma = sigma[idx]
        U = U[:, idx]
        Vt = Vt[idx, :]
        
        # Store factors
        self.user_factors = U * np.sqrt(sigma)
        self.item_factors = Vt.T * np.sqrt(sigma)
        
        # Initialize biases to 0 for SVD
        self.user_biases = np.zeros(self.stats['n_users'])
        self.item_biases = np.zeros(self.stats['n_items'])
    
    def _fit_als(self, R: csr_matrix, ratings_df: pd.DataFrame):
        """Fit using Alternating Least Squares"""
        print("   Running ALS optimization...")
        
        n_users = self.stats['n_users']
        n_items = self.stats['n_items']
        
        # Initialize factors randomly
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        R_dense = R.toarray()
        mask = R_dense > 0  # Mask for observed entries
        
        reg = self.regularization
        
        for epoch in range(self.n_epochs):
            # Fix items, update users
            for u in range(n_users):
                rated_items = np.where(mask[u])[0]
                if len(rated_items) == 0:
                    continue
                
                V_rated = self.item_factors[rated_items]
                r_u = R_dense[u, rated_items]
                
                A = V_rated.T @ V_rated + reg * len(rated_items) * np.eye(self.n_factors)
                b = V_rated.T @ r_u
                self.user_factors[u] = np.linalg.solve(A, b)
            
            # Fix users, update items
            for i in range(n_items):
                raters = np.where(mask[:, i])[0]
                if len(raters) == 0:
                    continue
                
                U_raters = self.user_factors[raters]
                r_i = R_dense[raters, i]
                
                A = U_raters.T @ U_raters + reg * len(raters) * np.eye(self.n_factors)
                b = U_raters.T @ r_i
                self.item_factors[i] = np.linalg.solve(A, b)
            
            # Calculate loss
            predictions = self.user_factors @ self.item_factors.T
            errors = (R_dense - predictions) * mask
            mse = np.sum(errors ** 2) / mask.sum()
            rmse = np.sqrt(mse)
            
            self.training_history.append({'epoch': epoch + 1, 'rmse': rmse})
            
            if (epoch + 1) % 5 == 0:
                print(f"      Epoch {epoch + 1}/{self.n_epochs}: RMSE = {rmse:.4f}")
        
        # Initialize biases to 0 for ALS
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
    
    def _fit_funk_svd(self, ratings_df: pd.DataFrame):
        """Fit using Funk SVD (SGD-based)"""
        print("   Running Funk SVD (SGD) optimization...")
        
        n_users = self.stats['n_users']
        n_items = self.stats['n_items']
        
        # Initialize
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        
        lr = self.learning_rate
        reg = self.regularization
        
        # Convert to numpy arrays for speed
        user_indices = ratings_df['user_id'].map(self.user_mapping).values
        item_indices = ratings_df['item_id'].map(self.item_mapping).values
        ratings = ratings_df['rating'].values
        n_ratings = len(ratings)
        
        for epoch in range(self.n_epochs):
            # Shuffle data
            indices = np.random.permutation(n_ratings)
            
            total_error = 0
            for idx in indices:
                u = user_indices[idx]
                i = item_indices[idx]
                r = ratings[idx]
                
                # Predict
                pred = (self.global_mean + 
                       self.user_biases[u] + 
                       self.item_biases[i] + 
                       np.dot(self.user_factors[u], self.item_factors[i]))
                
                # Error
                error = r - pred
                total_error += error ** 2
                
                # Update biases
                self.user_biases[u] += lr * (error - reg * self.user_biases[u])
                self.item_biases[i] += lr * (error - reg * self.item_biases[i])
                
                # Update factors
                user_factor_old = self.user_factors[u].copy()
                self.user_factors[u] += lr * (error * self.item_factors[i] - reg * self.user_factors[u])
                self.item_factors[i] += lr * (error * user_factor_old - reg * self.item_factors[i])
            
            rmse = np.sqrt(total_error / n_ratings)
            self.training_history.append({'epoch': epoch + 1, 'rmse': rmse})
            
            if (epoch + 1) % 5 == 0:
                print(f"      Epoch {epoch + 1}/{self.n_epochs}: RMSE = {rmse:.4f}")
    
    def predict(self, user_id: Any, item_id: Any) -> float:
        """Predict rating for a user-item pair"""
        if user_id not in self.user_mapping:
            return self.global_mean
        if item_id not in self.item_mapping:
            return self.global_mean
        
        u = self.user_mapping[user_id]
        i = self.item_mapping[item_id]
        
        pred = (self.global_mean + 
               self.user_biases[u] + 
               self.item_biases[i] + 
               np.dot(self.user_factors[u], self.item_factors[i]))
        
        return np.clip(pred, 1, 5)
    
    def recommend(
        self,
        user_id: Any,
        n_recommendations: int = 10,
        exclude_rated: bool = True,
        rated_items: Optional[set] = None
    ) -> List[Tuple[Any, float]]:
        """Generate top-N recommendations for a user"""
        if user_id not in self.user_mapping:
            return self._get_popular_items(n_recommendations)
        
        u = self.user_mapping[user_id]
        
        # Predict all item scores
        scores = (self.global_mean + 
                 self.user_biases[u] + 
                 self.item_biases + 
                 np.dot(self.user_factors[u], self.item_factors.T))
        
        # Get items to exclude
        if rated_items is None:
            rated_items = set()
        
        # Sort and filter
        item_scores = []
        for i, score in enumerate(scores):
            item_id = self.reverse_item_mapping[i]
            if not exclude_rated or item_id not in rated_items:
                item_scores.append((item_id, np.clip(score, 1, 5)))
        
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        return item_scores[:n_recommendations]
    
    def _get_popular_items(self, n: int) -> List[Tuple[Any, float]]:
        """Return items with highest average predicted score (cold start)"""
        avg_scores = self.global_mean + self.item_biases
        top_items = np.argsort(avg_scores)[-n:][::-1]
        return [(self.reverse_item_mapping[i], avg_scores[i]) for i in top_items]
    
    def get_similar_items(self, item_id: Any, n: int = 10) -> List[Tuple[Any, float]]:
        """Get most similar items based on latent factors"""
        if item_id not in self.item_mapping:
            return []
        
        i = self.item_mapping[item_id]
        item_vector = self.item_factors[i]
        
        # Cosine similarity
        norms = np.linalg.norm(self.item_factors, axis=1)
        norms[norms == 0] = 1
        normalized = self.item_factors / norms[:, np.newaxis]
        
        item_norm = np.linalg.norm(item_vector)
        if item_norm > 0:
            item_vector_norm = item_vector / item_norm
        else:
            item_vector_norm = item_vector
        
        similarities = np.dot(normalized, item_vector_norm)
        similarities[i] = -1  # Exclude self
        
        top_idx = np.argsort(similarities)[-n:][::-1]
        
        return [(self.reverse_item_mapping[idx], similarities[idx]) for idx in top_idx]
    
    def get_user_embedding(self, user_id: Any) -> Optional[np.ndarray]:
        """Get user's latent factor embedding"""
        if user_id not in self.user_mapping:
            return None
        return self.user_factors[self.user_mapping[user_id]]
    
    def get_item_embedding(self, item_id: Any) -> Optional[np.ndarray]:
        """Get item's latent factor embedding"""
        if item_id not in self.item_mapping:
            return None
        return self.item_factors[self.item_mapping[item_id]]
    
    def evaluate(
        self,
        test_df: pd.DataFrame,
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """Evaluate model on test set"""
        print("\n📊 Evaluating model...")
        
        # Rating prediction metrics
        y_true = []
        y_pred = []
        
        for _, row in test_df.iterrows():
            y_true.append(row['rating'])
            y_pred.append(self.predict(row['user_id'], row['item_id']))
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        metrics = {'RMSE': rmse, 'MAE': mae}
        
        # Ranking metrics
        user_items = defaultdict(list)
        for _, row in test_df.iterrows():
            if row['rating'] >= 4:
                user_items[row['user_id']].append(row['item_id'])
        
        for k in k_values:
            precisions, recalls, ndcgs = [], [], []
            
            for user_id, relevant_items in user_items.items():
                if not relevant_items:
                    continue
                
                recommendations = self.recommend(user_id, n_recommendations=k, exclude_rated=False)
                recommended_items = [item for item, _ in recommendations]
                
                hits = len(set(recommended_items) & set(relevant_items))
                precision = hits / k if k > 0 else 0
                recall = hits / len(relevant_items) if relevant_items else 0
                
                dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(recommended_items) if item in relevant_items)
                idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant_items), k)))
                ndcg = dcg / idcg if idcg > 0 else 0
                
                precisions.append(precision)
                recalls.append(recall)
                ndcgs.append(ndcg)
            
            metrics[f'Precision@{k}'] = np.mean(precisions) if precisions else 0
            metrics[f'Recall@{k}'] = np.mean(recalls) if recalls else 0
            metrics[f'NDCG@{k}'] = np.mean(ndcgs) if ndcgs else 0
        
        # Coverage
        all_recommended = set()
        sample_users = list(self.user_mapping.keys())[:100]
        for user_id in sample_users:
            recs = self.recommend(user_id, n_recommendations=10)
            all_recommended.update([item for item, _ in recs])
        
        metrics['Coverage'] = len(all_recommended) / self.stats['n_items']
        
        return metrics


def create_interactive_visualization(
    mf_model: MatrixFactorization,
    train_df: pd.DataFrame,
    sample_users: int = 50,
    sample_items: int = 100,
    output_path: str = "../visualizations/matrix_factorization.html",
    metrics: dict = None
):
    """
    Create enhanced interactive graph visualization showing latent space relationships
    
    Uses t-SNE to project latent factors to 2D for visualization
    Features:
    - Academic presentation with methodology
    - Larger, readable nodes
    - Edge labels
    - Interactive highlighting
    - Full-screen mode
    """
    print("\n🎨 Creating enhanced interactive visualization...")
    
    # Try to import enhanced utilities
    try:
        from visualization_utils import (
            generate_enhanced_visualization,
            ALGORITHM_INFO
        )
        use_enhanced = True
    except ImportError:
        use_enhanced = False
        print("   Note: Using basic visualization")
    
    # Initialize Pyvis network with larger nodes
    net = Network(
        height="700px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#333333"
    )
    
    # Enhanced options with larger nodes
    net.set_options("""
    {
        "nodes": {
            "font": {
                "size": 14,
                "strokeWidth": 3,
                "strokeColor": "#ffffff"
            },
            "borderWidth": 2,
            "shadow": true
        },
        "edges": {
            "font": {"size": 11, "align": "middle"},
            "smooth": {"type": "continuous"}
        },
        "physics": {
            "barnesHut": {
                "gravitationalConstant": -15000,
                "springLength": 250,
                "springConstant": 0.008,
                "damping": 0.5
            },
            "stabilization": {"iterations": 150}
        },
        "interaction": {
            "hover": true,
            "hoverConnectedEdges": true,
            "selectConnectedEdges": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": true
        }
    }
    """)
    
    # Sample users and items
    user_ids = list(mf_model.user_mapping.keys())[:sample_users]
    item_ids = list(mf_model.item_mapping.keys())[:sample_items]
    
    # Get embeddings
    user_embeddings = np.array([mf_model.user_factors[mf_model.user_mapping[u]] for u in user_ids])
    item_embeddings = np.array([mf_model.item_factors[mf_model.item_mapping[i]] for i in item_ids])
    
    # Combine for t-SNE
    all_embeddings = np.vstack([user_embeddings, item_embeddings])
    
    # Apply t-SNE for 2D visualization
    print("   Applying t-SNE for 2D projection...")
    if len(all_embeddings) > 50:
        perplexity = min(30, len(all_embeddings) - 1)
    else:
        perplexity = max(5, len(all_embeddings) // 3)
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    user_positions = embeddings_2d[:len(user_ids)]
    item_positions = embeddings_2d[len(user_ids):]
    
    # Scale positions for visualization
    scale = 600
    user_positions = user_positions * scale
    item_positions = item_positions * scale
    
    edge_count = 0
    
    # Add user nodes with LARGER sizes
    for i, user_id in enumerate(user_ids):
        x, y = user_positions[i]
        bias = mf_model.user_biases[mf_model.user_mapping[user_id]]
        embed_norm = np.linalg.norm(mf_model.user_factors[mf_model.user_mapping[user_id]])
        
        net.add_node(
            f"U_{user_id}",
            label=f"U{user_id}",
            color="#4285f4",
            size=25 + min(embed_norm * 5, 15),  # Much larger base size
            x=float(x),
            y=float(y),
            title=f"<div style='padding:10px;'>"
                  f"<strong style='font-size:14px;'>👤 User {user_id}</strong><br><br>"
                  f"<b>User Bias:</b> {bias:.3f}<br>"
                  f"<b>Embedding Norm:</b> {embed_norm:.3f}<br>"
                  f"<b>Latent Factors:</b> {mf_model.n_factors}"
                  f"</div>",
            group="user",
            shape="dot"
        )
    
    # Add item nodes with LARGER sizes
    for i, item_id in enumerate(item_ids):
        x, y = item_positions[i]
        bias = mf_model.item_biases[mf_model.item_mapping[item_id]]
        embed_norm = np.linalg.norm(mf_model.item_factors[mf_model.item_mapping[item_id]])
        
        net.add_node(
            f"I_{item_id}",
            label=f"I{item_id}",
            color="#34a853",
            size=22 + min(embed_norm * 4, 12),  # Larger base size
            x=float(x),
            y=float(y),
            title=f"<div style='padding:10px;'>"
                  f"<strong style='font-size:14px;'>📦 Item {item_id}</strong><br><br>"
                  f"<b>Item Bias:</b> {bias:.3f}<br>"
                  f"<b>Embedding Norm:</b> {embed_norm:.3f}<br>"
                  f"<b>Position:</b> t-SNE projected"
                  f"</div>",
            group="item",
            shape="dot"
        )
    
    # Collect recommendations for tables
    all_recommendations = []
    
    # Add recommendation edges with descriptive labels
    for user_id in user_ids[:20]:
        recommendations = mf_model.recommend(user_id, n_recommendations=3)
        for item_id, score in recommendations:
            reason = f"Latent factor alignment (dot product = {score:.2f})"
            all_recommendations.append((item_id, score, reason))
            
            if f"I_{item_id}" in [n['id'] for n in net.nodes]:
                label_text = f"pred {score:.1f}"
                net.add_edge(
                    f"U_{user_id}",
                    f"I_{item_id}",
                    color="#ea4335",
                    width=2 + (score - 3) * 0.5,
                    label=label_text,
                    title=f"<div style='padding:8px;'><b>🎯 Recommendation</b><br><br>"
                          f"<b>For:</b> User {user_id}<br>"
                          f"<b>Item:</b> {item_id}<br>"
                          f"<b>Predicted:</b> {score:.2f}/5<br>"
                          f"<b>Method:</b> Latent factor dot product</div>",
                    dashes=True,
                    font={"size": 9, "color": "#ea4335", "background": "rgba(255,255,255,0.8)"}
                )
                edge_count += 1
    
    # Add similarity edges between items with descriptive labels
    for item_id in item_ids[:30]:
        similar = mf_model.get_similar_items(item_id, n=2)
        for sim_item_id, sim_score in similar:
            if f"I_{sim_item_id}" in [n['id'] for n in net.nodes] and sim_score > 0.5:
                label_text = f"sim {sim_score:.2f}"
                net.add_edge(
                    f"I_{item_id}",
                    f"I_{sim_item_id}",
                    color="#999999",
                    width=sim_score * 2.5,
                    label=label_text,
                    title=f"<div style='padding:8px;'><b>🔗 Item Similarity</b><br><br>"
                          f"<b>Items:</b> {item_id} ↔ {sim_item_id}<br>"
                          f"<b>Cosine Similarity:</b> {sim_score:.3f}<br>"
                          f"<b>Meaning:</b> Items share similar latent characteristics</div>",
                    font={"size": 9, "color": "#666666", "background": "rgba(255,255,255,0.8)"}
                )
                edge_count += 1
    
    # Prepare recommendations data for tables
    unique_recs = {}
    for item_id, score, reason in all_recommendations:
        if item_id not in unique_recs or score > unique_recs[item_id][1]:
            unique_recs[item_id] = (item_id, score, reason)
    sorted_recs = sorted(unique_recs.values(), key=lambda x: -x[1])[:20]
    
    recommendations_data = {
        'recommendations': sorted_recs,
        'best_fit': [],
        'worst_fit': [],
        'avoided': []
    }
    
    # Graph statistics
    graph_stats = {
        "Users": len(user_ids),
        "Items": len(item_ids),
        "Edges": edge_count,
        "Latent Factors": mf_model.n_factors
    }
    
    # Node and edge types for legend
    node_types = [
        {"color": "#4285f4", "label": "Users", "description": "User latent vectors (t-SNE projected)"},
        {"color": "#34a853", "label": "Items", "description": "Item latent vectors (t-SNE projected)"}
    ]
    
    edge_types = [
        {"color": "#ea4335", "label": "Prediction (pred X.X)", "style": "dashed", "description": "Predicted user-item preference score"},
        {"color": "#999999", "label": "Similarity (sim X.XX)", "style": "solid", "description": "Cosine similarity in latent space"}
    ]
    
    # Generate enhanced visualization if available
    if use_enhanced:
        generate_enhanced_visualization(
            net=net,
            algorithm_key="matrix_factorization",
            output_path=output_path,
            metrics=metrics,
            graph_stats=graph_stats,
            node_types=node_types,
            edge_types=edge_types,
            recommendations_data=recommendations_data
        )
    else:
        # Fallback: save with basic legend
        legend_html = """
        <div style="position: absolute; top: 10px; right: 10px; background: white; 
                    padding: 20px; border-radius: 10px; box-shadow: 0 2px 15px rgba(0,0,0,0.15);
                    font-family: Arial, sans-serif; font-size: 13px; max-width: 280px;">
            <h3 style="margin: 0 0 15px 0; font-size: 16px; color: #333; border-bottom: 2px solid #e94560; padding-bottom: 10px;">
                Matrix Factorization
            </h3>
            <p style="font-size: 12px; color: #666; margin: 0 0 15px 0;">
                t-SNE projection of latent factors to 2D space. 
                Similar users/items appear closer together.
            </p>
            <div style="margin: 8px 0;">
                <span style="display: inline-block; width: 18px; height: 18px; 
                            background: #4285f4; border-radius: 50%; margin-right: 10px; vertical-align: middle;"></span>
                <strong>User Embeddings</strong>
            </div>
            <div style="margin: 8px 0;">
                <span style="display: inline-block; width: 18px; height: 18px; 
                            background: #34a853; border-radius: 50%; margin-right: 10px; vertical-align: middle;"></span>
                <strong>Item Embeddings</strong>
            </div>
            <hr style="margin: 12px 0; border: none; border-top: 1px solid #ddd;">
            <div style="margin: 8px 0;">
                <span style="display: inline-block; width: 30px; height: 3px; 
                            background: #ea4335; margin-right: 10px; vertical-align: middle;"></span>
                Recommendations
            </div>
            <div style="margin: 8px 0;">
                <span style="display: inline-block; width: 30px; height: 2px; 
                            background: #cccccc; margin-right: 10px; vertical-align: middle;"></span>
                Item Similarity
            </div>
        </div>
        """
        
        net.save_graph(output_path)
        
        with open(output_path, 'r') as f:
            html_content = f.read()
        
        html_content = html_content.replace('</body>', f'{legend_html}</body>')
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"   ✅ Visualization saved to: {output_path}")
    
    return output_path


def generate_statistical_report(
    mf_model: MatrixFactorization,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metrics: Dict[str, float],
    output_dir: str = "../reports/matrix_factorization"
):
    """Generate comprehensive statistical report with figures"""
    print("\n📈 Generating statistical report...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Matrix Factorization Analysis ({mf_model.method.upper()})', fontsize=14)
    
    # 1. Training History (if available)
    ax1 = axes[0, 0]
    if mf_model.training_history:
        epochs = [h['epoch'] for h in mf_model.training_history]
        rmses = [h['rmse'] for h in mf_model.training_history]
        ax1.plot(epochs, rmses, 'b-o', markersize=4)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('RMSE')
        ax1.set_title('Training Convergence')
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'SVD: Single decomposition\n(no iterations)', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Training History')
    
    # 2. User Bias Distribution
    ax2 = axes[0, 1]
    ax2.hist(mf_model.user_biases, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--')
    ax2.set_xlabel('Bias Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('User Bias Distribution')
    
    # 3. Item Bias Distribution
    ax3 = axes[0, 2]
    ax3.hist(mf_model.item_biases, bins=50, color='coral', edgecolor='white', alpha=0.7)
    ax3.axvline(x=0, color='red', linestyle='--')
    ax3.set_xlabel('Bias Value')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Item Bias Distribution')
    
    # 4. Singular Values / Factor Norms
    ax4 = axes[1, 0]
    user_norms = np.linalg.norm(mf_model.user_factors, axis=1)
    item_norms = np.linalg.norm(mf_model.item_factors, axis=1)
    ax4.hist(user_norms, bins=30, alpha=0.6, label='Users', color='blue')
    ax4.hist(item_norms, bins=30, alpha=0.6, label='Items', color='green')
    ax4.set_xlabel('Embedding Norm')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Embedding Vector Norms')
    ax4.legend()
    
    # 5. Evaluation Metrics
    ax5 = axes[1, 1]
    metric_names = ['RMSE', 'MAE', 'Precision@10', 'Recall@10', 'NDCG@10']
    metric_values = [metrics.get(m, 0) for m in metric_names]
    bars = ax5.bar(range(len(metric_names)), metric_values, 
                   color=['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#9b59b6'])
    ax5.set_xticks(range(len(metric_names)))
    ax5.set_xticklabels(metric_names, rotation=45, ha='right')
    ax5.set_ylabel('Score')
    ax5.set_title('Evaluation Metrics')
    for bar, val in zip(bars, metric_values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 6. Latent Factor Heatmap (sample)
    ax6 = axes[1, 2]
    sample_factors = mf_model.item_factors[:20, :min(20, mf_model.n_factors)]
    sns.heatmap(sample_factors, ax=ax6, cmap='RdBu_r', center=0,
                xticklabels=False, yticklabels=False)
    ax6.set_xlabel('Latent Factors')
    ax6.set_ylabel('Items (sample)')
    ax6.set_title('Item Latent Factors Heatmap')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/analysis_figures.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Generate text report
    report = f"""
================================================================================
MATRIX FACTORIZATION - STATISTICAL REPORT
================================================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CONFIGURATION
-------------
Method: {mf_model.method.upper()}
n_factors: {mf_model.n_factors}
n_epochs: {mf_model.n_epochs}
Learning Rate: {mf_model.learning_rate}
Regularization: {mf_model.regularization}

DATASET STATISTICS
------------------
Number of Users: {mf_model.stats['n_users']:,}
Number of Items: {mf_model.stats['n_items']:,}
Number of Ratings: {mf_model.stats['n_ratings']:,}
Matrix Sparsity: {mf_model.stats['sparsity']:.4%}

LATENT FACTOR STATISTICS
------------------------
User Factor Matrix Shape: {mf_model.user_factors.shape}
Item Factor Matrix Shape: {mf_model.item_factors.shape}
User Factor Mean: {mf_model.user_factors.mean():.6f}
User Factor Std: {mf_model.user_factors.std():.6f}
Item Factor Mean: {mf_model.item_factors.mean():.6f}
Item Factor Std: {mf_model.item_factors.std():.6f}

BIAS STATISTICS
---------------
User Bias Mean: {mf_model.user_biases.mean():.4f}
User Bias Std: {mf_model.user_biases.std():.4f}
Item Bias Mean: {mf_model.item_biases.mean():.4f}
Item Bias Std: {mf_model.item_biases.std():.4f}
Global Mean: {mf_model.global_mean:.4f}

EVALUATION METRICS
------------------
RMSE: {metrics.get('RMSE', 0):.4f}
MAE: {metrics.get('MAE', 0):.4f}
Precision@5: {metrics.get('Precision@5', 0):.4f}
Precision@10: {metrics.get('Precision@10', 0):.4f}
Precision@20: {metrics.get('Precision@20', 0):.4f}
Recall@5: {metrics.get('Recall@5', 0):.4f}
Recall@10: {metrics.get('Recall@10', 0):.4f}
Recall@20: {metrics.get('Recall@20', 0):.4f}
NDCG@5: {metrics.get('NDCG@5', 0):.4f}
NDCG@10: {metrics.get('NDCG@10', 0):.4f}
NDCG@20: {metrics.get('NDCG@20', 0):.4f}
Coverage: {metrics.get('Coverage', 0):.4%}

TRAINING HISTORY
----------------
"""
    
    if mf_model.training_history:
        report += f"Final Training RMSE: {mf_model.training_history[-1]['rmse']:.4f}\n"
        report += f"Total Epochs: {len(mf_model.training_history)}\n"
    else:
        report += "N/A (SVD uses direct decomposition)\n"
    
    report += "\n================================================================================\n"
    
    with open(f"{output_dir}/statistical_report.txt", 'w') as f:
        f.write(report)
    
    with open(f"{output_dir}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"   ✅ Report saved to: {output_dir}/")
    
    return report


def main():
    """Main execution function"""
    print("=" * 70)
    print("MATRIX FACTORIZATION RECOMMENDER SYSTEM")
    print("=" * 70)
    
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Import data loader
    from data_loader import DatasetGenerator
    
    # Generate dataset
    print("\n📦 Loading dataset...")
    generator = DatasetGenerator(output_dir="../data")
    
    dataset = generator.generate_movielens_style(
        n_users=500,
        n_items=1000,
        n_ratings=50000
    )
    
    ratings_df = dataset['ratings']
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
    print(f"   Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    # Test all three methods
    methods = ['svd', 'als', 'funk_svd']
    
    for method in methods:
        print("\n" + "=" * 70)
        print(f"TRAINING {method.upper()} MODEL")
        print("=" * 70)
        
        model = MatrixFactorization(
            n_factors=30,
            method=method,
            n_epochs=20,
            learning_rate=0.005,
            regularization=0.1  # Increased from 0.02 to prevent overfitting
        )
        model.fit(train_df)
        
        # Evaluate
        metrics = model.evaluate(test_df)
        print(f"\n📊 {method.upper()} Metrics:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        # Generate visualization
        create_interactive_visualization(
            model, train_df,
            sample_users=40,
            sample_items=80,
            output_path=f"../visualizations/matrix_factorization_{method}.html"
        )
        
        # Generate report
        generate_statistical_report(
            model, train_df, test_df, metrics,
            output_dir=f"../reports/matrix_factorization/{method}"
        )
    
    print("\n" + "=" * 70)
    print("✅ MATRIX FACTORIZATION COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Outputs:")
    print(f"   - Visualizations: ../visualizations/")
    print(f"   - Reports: ../reports/matrix_factorization/")


if __name__ == "__main__":
    main()
