"""
01_collaborative_filtering.py
==============================
Collaborative Filtering Recommender System

Implements:
- User-User Collaborative Filtering (kNN)
- Item-Item Collaborative Filtering (kNN)
- Both using Cosine Similarity

Features:
- Interactive graph visualization (Pyvis)
- Comprehensive statistical reports
- Hyperparameter exploration
- Multiple dataset support
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from pyvis.network import Network
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import DatasetGenerator, get_unified_format

# Create output directories
os.makedirs("../outputs/collaborative_filtering", exist_ok=True)
os.makedirs("../visualizations", exist_ok=True)
os.makedirs("../reports/collaborative_filtering", exist_ok=True)


class CollaborativeFilter:
    """
    Collaborative Filtering Recommender System
    
    Supports both User-User and Item-Item approaches
    with configurable similarity metrics and neighborhood sizes.
    """
    
    def __init__(
        self,
        method: str = 'user',  # 'user' or 'item'
        k_neighbors: int = 20,
        min_common_items: int = 3,
        similarity_metric: str = 'cosine'
    ):
        self.method = method
        self.k_neighbors = k_neighbors
        self.min_common_items = min_common_items
        self.similarity_metric = similarity_metric
        
        # Model state
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        self.global_mean = 0
        self.user_means = {}
        self.item_means = {}
        
        # Statistics
        self.stats = {
            'n_users': 0,
            'n_items': 0,
            'n_ratings': 0,
            'sparsity': 0,
            'avg_ratings_per_user': 0,
            'avg_ratings_per_item': 0
        }
        
    def fit(self, ratings_df: pd.DataFrame) -> 'CollaborativeFilter':
        """
        Fit the collaborative filtering model
        
        Args:
            ratings_df: DataFrame with columns [user_id, item_id, rating]
        """
        print(f"\n🔧 Fitting {self.method.upper()}-based Collaborative Filter...")
        print(f"   k_neighbors: {self.k_neighbors}")
        print(f"   min_common_items: {self.min_common_items}")
        print(f"   similarity_metric: {self.similarity_metric}")
        
        # Create user and item mappings
        unique_users = ratings_df['user_id'].unique()
        unique_items = ratings_df['item_id'].unique()
        
        self.user_mapping = {u: i for i, u in enumerate(unique_users)}
        self.item_mapping = {it: i for i, it in enumerate(unique_items)}
        self.reverse_user_mapping = {i: u for u, i in self.user_mapping.items()}
        self.reverse_item_mapping = {i: it for it, i in self.item_mapping.items()}
        
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        # Build user-item matrix
        row_indices = ratings_df['user_id'].map(self.user_mapping).values
        col_indices = ratings_df['item_id'].map(self.item_mapping).values
        ratings = ratings_df['rating'].values
        
        self.user_item_matrix = csr_matrix(
            (ratings, (row_indices, col_indices)),
            shape=(n_users, n_items)
        ).toarray()
        
        # Calculate means
        self.global_mean = ratings_df['rating'].mean()
        
        for user_idx in range(n_users):
            user_ratings = self.user_item_matrix[user_idx]
            mask = user_ratings > 0
            if mask.sum() > 0:
                self.user_means[user_idx] = user_ratings[mask].mean()
            else:
                self.user_means[user_idx] = self.global_mean
                
        for item_idx in range(n_items):
            item_ratings = self.user_item_matrix[:, item_idx]
            mask = item_ratings > 0
            if mask.sum() > 0:
                self.item_means[item_idx] = item_ratings[mask].mean()
            else:
                self.item_means[item_idx] = self.global_mean
        
        # Compute similarity matrix
        print("   Computing similarity matrix...")
        if self.method == 'user':
            self.similarity_matrix = self._compute_similarity(self.user_item_matrix)
        else:
            self.similarity_matrix = self._compute_similarity(self.user_item_matrix.T)
        
        # Update statistics
        self.stats['n_users'] = n_users
        self.stats['n_items'] = n_items
        self.stats['n_ratings'] = len(ratings_df)
        self.stats['sparsity'] = 1 - len(ratings_df) / (n_users * n_items)
        self.stats['avg_ratings_per_user'] = len(ratings_df) / n_users
        self.stats['avg_ratings_per_item'] = len(ratings_df) / n_items
        
        print(f"   ✅ Model fitted: {n_users} users, {n_items} items")
        print(f"   📊 Sparsity: {self.stats['sparsity']:.2%}")
        
        return self
    
    def _compute_similarity(self, matrix: np.ndarray) -> np.ndarray:
        """Compute pairwise similarity matrix"""
        if self.similarity_metric == 'cosine':
            # Add small epsilon to avoid division by zero
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normalized = matrix / norms
            similarity = np.dot(normalized, normalized.T)
        elif self.similarity_metric == 'pearson':
            # Center the data
            means = np.mean(matrix, axis=1, keepdims=True)
            centered = matrix - means
            norms = np.linalg.norm(centered, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normalized = centered / norms
            similarity = np.dot(normalized, normalized.T)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        # Set diagonal to 0 (no self-similarity)
        np.fill_diagonal(similarity, 0)
        
        return similarity
    
    def predict(self, user_id: Any, item_id: Any) -> float:
        """Predict rating for a user-item pair"""
        if user_id not in self.user_mapping:
            return self.global_mean
        if item_id not in self.item_mapping:
            return self.global_mean
            
        user_idx = self.user_mapping[user_id]
        item_idx = self.item_mapping[item_id]
        
        if self.method == 'user':
            return self._predict_user_based(user_idx, item_idx)
        else:
            return self._predict_item_based(user_idx, item_idx)
    
    def _predict_user_based(self, user_idx: int, item_idx: int) -> float:
        """User-based prediction"""
        # Find users who rated this item
        item_ratings = self.user_item_matrix[:, item_idx]
        raters = np.where(item_ratings > 0)[0]
        
        if len(raters) == 0:
            return self.user_means.get(user_idx, self.global_mean)
        
        # Get similarities to raters
        similarities = self.similarity_matrix[user_idx, raters]
        
        # Get top-k neighbors
        if len(raters) > self.k_neighbors:
            top_k_idx = np.argsort(similarities)[-self.k_neighbors:]
            neighbors = raters[top_k_idx]
            neighbor_sims = similarities[top_k_idx]
        else:
            neighbors = raters
            neighbor_sims = similarities
        
        # Filter out negative similarities
        mask = neighbor_sims > 0
        if mask.sum() == 0:
            return self.user_means.get(user_idx, self.global_mean)
        
        neighbors = neighbors[mask]
        neighbor_sims = neighbor_sims[mask]
        
        # Weighted average prediction
        neighbor_ratings = self.user_item_matrix[neighbors, item_idx]
        neighbor_means = np.array([self.user_means[n] for n in neighbors])
        
        user_mean = self.user_means.get(user_idx, self.global_mean)
        
        numerator = np.sum(neighbor_sims * (neighbor_ratings - neighbor_means))
        denominator = np.sum(np.abs(neighbor_sims))
        
        if denominator == 0:
            return user_mean
        
        prediction = user_mean + numerator / denominator
        
        # Clip to valid range
        return np.clip(prediction, 1, 5)
    
    def _predict_item_based(self, user_idx: int, item_idx: int) -> float:
        """Item-based prediction"""
        # Find items rated by this user
        user_ratings = self.user_item_matrix[user_idx]
        rated_items = np.where(user_ratings > 0)[0]
        
        if len(rated_items) == 0:
            return self.item_means.get(item_idx, self.global_mean)
        
        # Get similarities to rated items
        similarities = self.similarity_matrix[item_idx, rated_items]
        
        # Get top-k neighbors
        if len(rated_items) > self.k_neighbors:
            top_k_idx = np.argsort(similarities)[-self.k_neighbors:]
            neighbors = rated_items[top_k_idx]
            neighbor_sims = similarities[top_k_idx]
        else:
            neighbors = rated_items
            neighbor_sims = similarities
        
        # Filter out negative similarities
        mask = neighbor_sims > 0
        if mask.sum() == 0:
            return self.item_means.get(item_idx, self.global_mean)
        
        neighbors = neighbors[mask]
        neighbor_sims = neighbor_sims[mask]
        
        # Weighted average
        neighbor_ratings = user_ratings[neighbors]
        
        numerator = np.sum(neighbor_sims * neighbor_ratings)
        denominator = np.sum(neighbor_sims)
        
        if denominator == 0:
            return self.item_means.get(item_idx, self.global_mean)
        
        prediction = numerator / denominator
        
        return np.clip(prediction, 1, 5)
    
    def recommend(
        self,
        user_id: Any,
        n_recommendations: int = 10,
        exclude_rated: bool = True
    ) -> List[Tuple[Any, float]]:
        """
        Generate top-N recommendations for a user
        
        Returns:
            List of (item_id, predicted_score) tuples
        """
        if user_id not in self.user_mapping:
            # Cold start - return popular items
            return self._get_popular_items(n_recommendations)
        
        user_idx = self.user_mapping[user_id]
        user_ratings = self.user_item_matrix[user_idx]
        
        # Get candidate items
        if exclude_rated:
            candidates = np.where(user_ratings == 0)[0]
        else:
            candidates = np.arange(len(user_ratings))
        
        # Predict scores for all candidates
        scores = []
        for item_idx in candidates:
            item_id = self.reverse_item_mapping[item_idx]
            score = self.predict(user_id, item_id)
            scores.append((item_id, score))
        
        # Sort by score and return top N
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:n_recommendations]
    
    def _get_popular_items(self, n: int) -> List[Tuple[Any, float]]:
        """Return most popular items (for cold start)"""
        item_popularity = (self.user_item_matrix > 0).sum(axis=0)
        top_items = np.argsort(item_popularity)[-n:][::-1]
        
        return [(self.reverse_item_mapping[idx], self.item_means.get(idx, self.global_mean)) 
                for idx in top_items]
    
    def evaluate(
        self,
        test_df: pd.DataFrame,
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """
        Evaluate model on test set
        
        Returns dict with RMSE, MAE, Precision@K, Recall@K, NDCG@K
        """
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
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
        }
        
        # Ranking metrics (Precision@K, Recall@K, NDCG@K)
        # Group test set by user
        user_items = defaultdict(list)
        for _, row in test_df.iterrows():
            if row['rating'] >= 4:  # Consider as relevant
                user_items[row['user_id']].append(row['item_id'])
        
        for k in k_values:
            precisions = []
            recalls = []
            ndcgs = []
            
            for user_id, relevant_items in user_items.items():
                if len(relevant_items) == 0:
                    continue
                    
                # Get recommendations
                recommendations = self.recommend(user_id, n_recommendations=k)
                recommended_items = [item for item, _ in recommendations]
                
                # Calculate metrics
                hits = len(set(recommended_items) & set(relevant_items))
                precision = hits / k if k > 0 else 0
                recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0
                
                # NDCG
                dcg = 0
                for i, item in enumerate(recommended_items):
                    if item in relevant_items:
                        dcg += 1 / np.log2(i + 2)
                
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
    
    def get_similar_users(self, user_id: Any, n: int = 10) -> List[Tuple[Any, float]]:
        """Get most similar users"""
        if user_id not in self.user_mapping:
            return []
        
        user_idx = self.user_mapping[user_id]
        similarities = self.similarity_matrix[user_idx]
        top_idx = np.argsort(similarities)[-n:][::-1]
        
        return [(self.reverse_user_mapping[idx], similarities[idx]) for idx in top_idx]
    
    def get_similar_items(self, item_id: Any, n: int = 10) -> List[Tuple[Any, float]]:
        """Get most similar items"""
        if item_id not in self.item_mapping:
            return []
        
        item_idx = self.item_mapping[item_id]
        
        if self.method == 'item':
            similarities = self.similarity_matrix[item_idx]
        else:
            # Compute item similarities on-the-fly
            item_vectors = self.user_item_matrix.T
            item_vector = item_vectors[item_idx]
            similarities = cosine_similarity([item_vector], item_vectors)[0]
            similarities[item_idx] = 0
        
        top_idx = np.argsort(similarities)[-n:][::-1]
        
        return [(self.reverse_item_mapping[idx], similarities[idx]) for idx in top_idx]


def create_interactive_visualization(
    cf_model: CollaborativeFilter,
    train_df: pd.DataFrame,
    sample_users: int = 30,
    sample_items: int = 50,
    output_path: str = "../visualizations/collaborative_filtering.html",
    algorithm_variant: str = "collaborative_filtering",
    metrics: dict = None
):
    """
    Create enhanced interactive graph visualization showing recommendation flow
    
    Nodes: Users (blue), Items (green), Recommended Items (orange)
    Edges: Existing ratings (gray), Recommendations (red)
    
    Features:
    - Academic presentation with methodology description
    - Performance metrics table
    - Node highlighting on selection
    - Full-screen mode
    - Edge labels with rating/score values
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
        print("   Note: visualization_utils not found, using basic visualization")
    
    # Initialize Pyvis network with enhanced settings
    net = Network(
        height="700px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#333333",
        directed=True
    )
    
    # Enhanced physics settings for better layout and larger nodes
    net.set_options("""
    {
        "nodes": {
            "font": {
                "size": 14,
                "strokeWidth": 3,
                "strokeColor": "#ffffff"
            },
            "borderWidth": 2,
            "shadow": {
                "enabled": true,
                "color": "rgba(0,0,0,0.3)",
                "size": 10
            }
        },
        "edges": {
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.8}},
            "font": {"size": 11, "align": "middle", "background": "rgba(255,255,255,0.8)"},
            "smooth": {"type": "continuous"},
            "width": 2
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -80,
                "centralGravity": 0.01,
                "springLength": 200,
                "springConstant": 0.03,
                "damping": 0.4
            },
            "solver": "forceAtlas2Based",
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
    
    # Sample users and their rated items
    user_ids = list(cf_model.user_mapping.keys())[:sample_users]
    
    # Track items and statistics
    items_to_add = set()
    recommended_items = set()
    edge_count = 0
    
    # Add user nodes with larger size and better labels
    for user_id in user_ids:
        user_idx = cf_model.user_mapping[user_id]
        n_ratings = int((cf_model.user_item_matrix[user_idx] > 0).sum())
        avg_rating = cf_model.user_means.get(user_idx, 0)
        
        net.add_node(
            f"U_{user_id}",
            label=f"U{user_id}",
            color="#4285f4",
            size=25 + min(n_ratings * 0.8, 20),  # Larger base size
            title=f"<div style='padding:10px;'>"
                  f"<strong style='font-size:14px;'>👤 User {user_id}</strong><br><br>"
                  f"<b>Ratings Given:</b> {n_ratings}<br>"
                  f"<b>Avg Rating:</b> {avg_rating:.2f}<br>"
                  f"<b>Type:</b> {'Active' if n_ratings > 10 else 'Casual'} User"
                  f"</div>",
            group="user",
            shape="dot"
        )
        
        # Get user's rated items (sample)
        user_ratings = cf_model.user_item_matrix[user_idx]
        rated_idx = np.where(user_ratings > 0)[0][:10]
        
        for item_idx in rated_idx:
            item_id = cf_model.reverse_item_mapping[item_idx]
            items_to_add.add((item_id, user_ratings[item_idx]))
        
        # Get recommendations
        recommendations = cf_model.recommend(user_id, n_recommendations=5)
        for item_id, score in recommendations:
            recommended_items.add((item_id, score))
    
    # Add rated item nodes (green) with enhanced appearance
    for item_id, _ in list(items_to_add)[:sample_items]:
        if item_id in cf_model.item_mapping:
            item_idx = cf_model.item_mapping[item_id]
            n_ratings = int((cf_model.user_item_matrix[:, item_idx] > 0).sum())
            avg_rating = cf_model.item_means.get(item_idx, 0)
            
            net.add_node(
                f"I_{item_id}",
                label=f"I{item_id}",
                color="#34a853",
                size=20 + min(n_ratings * 0.5, 15),
                title=f"<div style='padding:10px;'>"
                      f"<strong style='font-size:14px;'>📦 Item {item_id}</strong><br><br>"
                      f"<b>Times Rated:</b> {n_ratings}<br>"
                      f"<b>Avg Rating:</b> {avg_rating:.2f}<br>"
                      f"<b>Popularity:</b> {'High' if n_ratings > 20 else 'Medium' if n_ratings > 5 else 'Low'}"
                      f"</div>",
                group="item",
                shape="dot"
            )
    
    # Add recommended item nodes (orange) with enhanced appearance
    for item_id, score in list(recommended_items)[:30]:
        if item_id in cf_model.item_mapping:
            item_idx = cf_model.item_mapping[item_id]
            n_ratings = int((cf_model.user_item_matrix[:, item_idx] > 0).sum())
            avg_rating = cf_model.item_means.get(item_idx, 0)
            
            node_id = f"I_{item_id}"
            if node_id not in [n['id'] for n in net.nodes]:
                net.add_node(
                    node_id,
                    label=f"R{item_id}",
                    color="#fbbc04",
                    size=22 + min(score * 3, 15),
                    title=f"<div style='padding:10px;'>"
                          f"<strong style='font-size:14px;'>⭐ Recommended Item {item_id}</strong><br><br>"
                          f"<b>Predicted Score:</b> {score:.2f}<br>"
                          f"<b>Actual Avg Rating:</b> {avg_rating:.2f}<br>"
                          f"<b>Confidence:</b> {'High' if score > 4 else 'Medium' if score > 3 else 'Low'}"
                          f"</div>",
                    group="recommended",
                    shape="dot"  # All nodes use circle shape for consistency
                )
    
    # Add edges for existing ratings with descriptive labels
    for user_id in user_ids:
        user_idx = cf_model.user_mapping[user_id]
        user_ratings = cf_model.user_item_matrix[user_idx]
        rated_idx = np.where(user_ratings > 0)[0][:10]
        
        for item_idx in rated_idx:
            item_id = cf_model.reverse_item_mapping[item_idx]
            if f"I_{item_id}" in [n['id'] for n in net.nodes]:
                rating = user_ratings[item_idx]
                # Descriptive label: "rated X/5"
                label_text = f"rated {rating:.0f}/5"
                net.add_edge(
                    f"U_{user_id}",
                    f"I_{item_id}",
                    color="#999999",
                    width=1.5 + rating * 0.3,
                    label=label_text,
                    title=f"<div style='padding:8px;'><b>📝 Existing Rating</b><br><br>"
                          f"<b>From:</b> User {user_id}<br>"
                          f"<b>To:</b> Item {item_id}<br>"
                          f"<b>Rating:</b> {rating:.1f} out of 5<br>"
                          f"<b>Type:</b> Historical interaction</div>",
                    font={"size": 9, "color": "#666666", "strokeWidth": 0, "background": "rgba(255,255,255,0.8)"}
                )
                edge_count += 1
    
    # Collect recommendation data for tables
    all_recommendations = []
    
    # Add edges for recommendations with descriptive labels
    for user_id in user_ids:
        recommendations = cf_model.recommend(user_id, n_recommendations=5)
        for item_id, score in recommendations:
            # Collect for table
            reason = f"Similar users liked this item (predicted {score:.2f})"
            all_recommendations.append((item_id, score, reason))
            
            if f"I_{item_id}" in [n['id'] for n in net.nodes]:
                # Descriptive label: "pred X.X"
                label_text = f"pred {score:.1f}"
                net.add_edge(
                    f"U_{user_id}",
                    f"I_{item_id}",
                    color="#ea4335",
                    width=2 + score * 0.4,
                    label=label_text,
                    title=f"<div style='padding:8px;'><b>🎯 Recommendation</b><br><br>"
                          f"<b>For:</b> User {user_id}<br>"
                          f"<b>Item:</b> {item_id}<br>"
                          f"<b>Predicted Score:</b> {score:.2f}/5<br>"
                          f"<b>Confidence:</b> {'High' if score > 4 else 'Medium' if score > 3 else 'Low'}<br>"
                          f"<b>Reason:</b> Based on similar user preferences</div>",
                    dashes=True,
                    font={"size": 9, "color": "#ea4335", "strokeWidth": 0, "background": "rgba(255,255,255,0.8)"}
                )
                edge_count += 1
    
    # Sort recommendations by score and get unique items
    unique_recs = {}
    for item_id, score, reason in all_recommendations:
        if item_id not in unique_recs or score > unique_recs[item_id][1]:
            unique_recs[item_id] = (item_id, score, reason)
    sorted_recs = sorted(unique_recs.values(), key=lambda x: -x[1])[:20]
    
    # Create best fit and worst fit from actual ratings (sample user)
    best_fit = []
    worst_fit = []
    sample_user_id = user_ids[0] if user_ids else None
    if sample_user_id:
        user_idx = cf_model.user_mapping[sample_user_id]
        user_ratings = cf_model.user_item_matrix[user_idx]
        rated_items = [(cf_model.reverse_item_mapping[i], user_ratings[i]) 
                       for i in np.where(user_ratings > 0)[0]]
        rated_items.sort(key=lambda x: -x[1])
        
        for item_id, rating in rated_items[:10]:
            n_ratings = int((cf_model.user_item_matrix[:, cf_model.item_mapping[item_id]] > 0).sum())
            popularity = f"{n_ratings} ratings"
            best_fit.append((item_id, rating, popularity))
        
        rated_items.sort(key=lambda x: x[1])
        for item_id, rating in rated_items[:5]:
            avg_global = cf_model.item_means.get(cf_model.item_mapping[item_id], 0)
            worst_fit.append((item_id, rating, f"{avg_global:.1f}"))
    
    # Prepare recommendations data for tables
    recommendations_data = {
        'recommendations': sorted_recs,
        'best_fit': best_fit,
        'worst_fit': worst_fit,
        'avoided': []  # Could add items with low predicted scores
    }
    
    # Calculate graph statistics
    graph_stats = {
        "Users": len(user_ids),
        "Items": len([n for n in net.nodes if n['id'].startswith('I_')]),
        "Edges": edge_count,
        "Recommendations": len(recommended_items)
    }
    
    # Node and edge type definitions for legend
    node_types = [
        {"color": "#4285f4", "label": "Users", "description": "User nodes (size = activity level)"},
        {"color": "#34a853", "label": "Rated Items", "description": "Items with existing ratings"},
        {"color": "#fbbc04", "label": "Recommended", "description": "Predicted recommendations"}
    ]
    
    edge_types = [
        {"color": "#999999", "label": "Rating (rated X/5)", "style": "solid", "description": "Historical user-item interaction"},
        {"color": "#ea4335", "label": "Prediction (pred X.X)", "style": "dashed", "description": "Algorithm-predicted preference"}
    ]
    
    # Generate enhanced visualization if available
    if use_enhanced:
        generate_enhanced_visualization(
            net=net,
            algorithm_key=algorithm_variant,
            output_path=output_path,
            metrics=metrics,
            graph_stats=graph_stats,
            node_types=node_types,
            edge_types=edge_types,
            recommendations_data=recommendations_data
        )
    else:
        # Fallback: save basic visualization with simple legend
        net.save_graph(output_path)
        
        # Add basic legend
        legend_html = """
        <div style="position: absolute; top: 10px; right: 10px; background: white; 
                    padding: 20px; border-radius: 10px; box-shadow: 0 2px 15px rgba(0,0,0,0.15);
                    font-family: Arial, sans-serif; font-size: 13px; max-width: 220px;">
            <h3 style="margin: 0 0 15px 0; font-size: 16px; color: #333; border-bottom: 2px solid #e94560; padding-bottom: 8px;">Legend</h3>
            <div style="margin: 8px 0;">
                <span style="display: inline-block; width: 18px; height: 18px; 
                            background: #4285f4; border-radius: 50%; margin-right: 10px; vertical-align: middle;"></span>
                <strong>Users</strong>
            </div>
            <div style="margin: 8px 0;">
                <span style="display: inline-block; width: 18px; height: 18px; 
                            background: #34a853; border-radius: 50%; margin-right: 10px; vertical-align: middle;"></span>
                <strong>Rated Items</strong>
            </div>
            <div style="margin: 8px 0;">
                <span style="display: inline-block; width: 18px; height: 18px; 
                            background: #fbbc04; border-radius: 50%; margin-right: 10px; vertical-align: middle;"></span>
                <strong>Recommended</strong>
            </div>
            <hr style="margin: 12px 0; border: none; border-top: 1px solid #ddd;">
            <div style="margin: 8px 0;">
                <span style="display: inline-block; width: 30px; height: 3px; 
                            background: #999999; margin-right: 10px; vertical-align: middle;"></span>
                Existing Rating
            </div>
            <div style="margin: 8px 0;">
                <span style="display: inline-block; width: 30px; height: 3px; 
                            background: #ea4335; margin-right: 10px; vertical-align: middle;
                            border-style: dashed; border-width: 2px 0 0 0; background: none; border-color: #ea4335;"></span>
                Recommendation
            </div>
        </div>
        """
        
        with open(output_path, 'r') as f:
            html_content = f.read()
        
        html_content = html_content.replace('</body>', f'{legend_html}</body>')
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"   ✅ Visualization saved to: {output_path}")
    
    return output_path


def generate_statistical_report(
    cf_model: CollaborativeFilter,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metrics: Dict[str, float],
    hyperparams: Dict[str, Any],
    output_dir: str = "../reports/collaborative_filtering"
):
    """Generate comprehensive statistical report with figures"""
    print("\n📈 Generating statistical report...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Collaborative Filtering Analysis ({cf_model.method.upper()}-based)', fontsize=14)
    
    # 1. Rating Distribution
    ax1 = axes[0, 0]
    train_df['rating'].hist(bins=5, ax=ax1, color='steelblue', edgecolor='white')
    ax1.set_xlabel('Rating')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Rating Distribution')
    
    # 2. Ratings per User Distribution
    ax2 = axes[0, 1]
    ratings_per_user = train_df.groupby('user_id').size()
    ratings_per_user.hist(bins=30, ax=ax2, color='coral', edgecolor='white')
    ax2.set_xlabel('Number of Ratings')
    ax2.set_ylabel('Number of Users')
    ax2.set_title('Ratings per User Distribution')
    ax2.set_xlim(0, ratings_per_user.quantile(0.95))
    
    # 3. Ratings per Item Distribution
    ax3 = axes[0, 2]
    ratings_per_item = train_df.groupby('item_id').size()
    ratings_per_item.hist(bins=30, ax=ax3, color='seagreen', edgecolor='white')
    ax3.set_xlabel('Number of Ratings')
    ax3.set_ylabel('Number of Items')
    ax3.set_title('Ratings per Item Distribution')
    ax3.set_xlim(0, ratings_per_item.quantile(0.95))
    
    # 4. Similarity Score Distribution
    ax4 = axes[1, 0]
    sim_values = cf_model.similarity_matrix.flatten()
    sim_values = sim_values[sim_values != 0]  # Remove zeros
    if len(sim_values) > 10000:
        sim_values = np.random.choice(sim_values, 10000)
    ax4.hist(sim_values, bins=50, color='purple', edgecolor='white', alpha=0.7)
    ax4.set_xlabel('Similarity Score')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'{cf_model.method.title()} Similarity Distribution')
    
    # 5. Evaluation Metrics Bar Chart
    ax5 = axes[1, 1]
    metric_names = ['RMSE', 'MAE', 'Precision@10', 'Recall@10', 'NDCG@10']
    metric_values = [metrics.get(m, 0) for m in metric_names]
    bars = ax5.bar(range(len(metric_names)), metric_values, color=['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#9b59b6'])
    ax5.set_xticks(range(len(metric_names)))
    ax5.set_xticklabels(metric_names, rotation=45, ha='right')
    ax5.set_ylabel('Score')
    ax5.set_title('Evaluation Metrics')
    for bar, val in zip(bars, metric_values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 6. Prediction Error Distribution
    ax6 = axes[1, 2]
    errors = []
    for _, row in test_df.sample(min(1000, len(test_df))).iterrows():
        pred = cf_model.predict(row['user_id'], row['item_id'])
        errors.append(row['rating'] - pred)
    ax6.hist(errors, bins=30, color='teal', edgecolor='white', alpha=0.7)
    ax6.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax6.set_xlabel('Prediction Error (Actual - Predicted)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Prediction Error Distribution')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/analysis_figures.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Generate text report
    report = f"""
================================================================================
COLLABORATIVE FILTERING - STATISTICAL REPORT
================================================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CONFIGURATION
-------------
Method: {cf_model.method.upper()}-based
k_neighbors: {hyperparams.get('k_neighbors', 'N/A')}
min_common_items: {hyperparams.get('min_common_items', 'N/A')}
Similarity Metric: {hyperparams.get('similarity_metric', 'N/A')}

DATASET STATISTICS
------------------
Number of Users: {cf_model.stats['n_users']:,}
Number of Items: {cf_model.stats['n_items']:,}
Number of Ratings: {cf_model.stats['n_ratings']:,}
Matrix Sparsity: {cf_model.stats['sparsity']:.4%}
Avg Ratings per User: {cf_model.stats['avg_ratings_per_user']:.2f}
Avg Ratings per Item: {cf_model.stats['avg_ratings_per_item']:.2f}

RATING STATISTICS
-----------------
Mean Rating: {train_df['rating'].mean():.3f}
Std Rating: {train_df['rating'].std():.3f}
Min Rating: {train_df['rating'].min()}
Max Rating: {train_df['rating'].max()}

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

SIMILARITY MATRIX STATISTICS
----------------------------
Mean Similarity: {cf_model.similarity_matrix.mean():.4f}
Max Similarity: {cf_model.similarity_matrix.max():.4f}
Non-zero Entries: {(cf_model.similarity_matrix != 0).sum():,}

================================================================================
"""
    
    with open(f"{output_dir}/statistical_report.txt", 'w') as f:
        f.write(report)
    
    # Save metrics as JSON
    with open(f"{output_dir}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"   ✅ Report saved to: {output_dir}/")
    
    return report


def run_hyperparameter_experiments(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = "../reports/collaborative_filtering"
):
    """Run experiments with different hyperparameters"""
    print("\n🔬 Running hyperparameter experiments...")
    
    results = []
    
    # Hyperparameter grid
    methods = ['user', 'item']
    k_values = [5, 10, 20, 30, 50]
    similarity_metrics = ['cosine', 'pearson']
    
    total_experiments = len(methods) * len(k_values) * len(similarity_metrics)
    experiment_num = 0
    
    for method in methods:
        for k in k_values:
            for sim_metric in similarity_metrics:
                experiment_num += 1
                print(f"   [{experiment_num}/{total_experiments}] "
                      f"method={method}, k={k}, similarity={sim_metric}")
                
                try:
                    model = CollaborativeFilter(
                        method=method,
                        k_neighbors=k,
                        similarity_metric=sim_metric
                    )
                    model.fit(train_df)
                    metrics = model.evaluate(test_df)
                    
                    results.append({
                        'method': method,
                        'k_neighbors': k,
                        'similarity_metric': sim_metric,
                        **metrics
                    })
                except Exception as e:
                    print(f"      ⚠️ Error: {e}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/hyperparameter_results.csv", index=False)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # RMSE by k and method
    ax1 = axes[0]
    for method in methods:
        subset = results_df[results_df['method'] == method]
        for sim in similarity_metrics:
            data = subset[subset['similarity_metric'] == sim]
            ax1.plot(data['k_neighbors'], data['RMSE'], 
                    marker='o', label=f'{method}-{sim}')
    ax1.set_xlabel('k_neighbors')
    ax1.set_ylabel('RMSE')
    ax1.set_title('RMSE vs k_neighbors')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # NDCG@10 by k and method
    ax2 = axes[1]
    for method in methods:
        subset = results_df[results_df['method'] == method]
        for sim in similarity_metrics:
            data = subset[subset['similarity_metric'] == sim]
            ax2.plot(data['k_neighbors'], data['NDCG@10'], 
                    marker='s', label=f'{method}-{sim}')
    ax2.set_xlabel('k_neighbors')
    ax2.set_ylabel('NDCG@10')
    ax2.set_title('NDCG@10 vs k_neighbors')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/hyperparameter_analysis.png", dpi=150)
    plt.close()
    
    print(f"   ✅ Hyperparameter results saved to: {output_dir}/")
    
    return results_df


def main():
    """Main execution function"""
    print("=" * 70)
    print("COLLABORATIVE FILTERING RECOMMENDER SYSTEM")
    print("=" * 70)
    
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Generate or load dataset
    print("\n📦 Loading dataset...")
    generator = DatasetGenerator(output_dir="../data")
    
    # Generate MovieLens-style dataset
    dataset = generator.generate_movielens_style(
        n_users=500,
        n_items=1000,
        n_ratings=50000
    )
    generator.save_datasets(dataset, "movielens")
    
    # Get unified format
    ratings_df = dataset['ratings']
    
    # Split into train/test
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
    print(f"   Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    # Dataset stats for visualization
    n_users = ratings_df['user_id'].nunique()
    n_items = ratings_df['item_id'].nunique()
    n_ratings = len(ratings_df)
    
    # Train User-based CF
    print("\n" + "=" * 70)
    print("TRAINING USER-BASED COLLABORATIVE FILTERING")
    print("=" * 70)
    
    user_cf = CollaborativeFilter(
        method='user',
        k_neighbors=20,
        min_common_items=3,
        similarity_metric='cosine'
    )
    user_cf.fit(train_df)
    
    # Evaluate
    user_metrics = user_cf.evaluate(test_df)
    print("\n📊 User-CF Metrics:")
    for metric, value in user_metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    # Prepare metrics for visualization
    viz_metrics = {
        'RMSE': user_metrics.get('RMSE', 0),
        'MAE': user_metrics.get('MAE', 0),
        'Precision@10': user_metrics.get('Precision@10', 0),
        'NDCG@10': user_metrics.get('NDCG@10', 0),
        'Coverage': user_metrics.get('Coverage', 0),
        'n_users': n_users,
        'n_items': n_items,
        'n_ratings': n_ratings
    }
    
    # Generate visualization
    viz_path = create_interactive_visualization(
        user_cf, train_df,
        sample_users=25,
        sample_items=40,
        output_path="../visualizations/user_collaborative_filtering.html",
        algorithm_variant="user_collaborative_filtering",
        metrics=viz_metrics
    )
    
    # Generate report
    generate_statistical_report(
        user_cf, train_df, test_df, user_metrics,
        {'k_neighbors': 20, 'min_common_items': 3, 'similarity_metric': 'cosine'},
        output_dir="../reports/collaborative_filtering/user_based"
    )
    
    # Train Item-based CF
    print("\n" + "=" * 70)
    print("TRAINING ITEM-BASED COLLABORATIVE FILTERING")
    print("=" * 70)
    
    item_cf = CollaborativeFilter(
        method='item',
        k_neighbors=20,
        similarity_metric='cosine'
    )
    item_cf.fit(train_df)
    
    # Evaluate
    item_metrics = item_cf.evaluate(test_df)
    print("\n📊 Item-CF Metrics:")
    for metric, value in item_metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    # Prepare metrics for visualization
    viz_metrics_item = {
        'RMSE': item_metrics.get('RMSE', 0),
        'MAE': item_metrics.get('MAE', 0),
        'Precision@10': item_metrics.get('Precision@10', 0),
        'NDCG@10': item_metrics.get('NDCG@10', 0),
        'Coverage': item_metrics.get('Coverage', 0),
        'n_users': n_users,
        'n_items': n_items,
        'n_ratings': n_ratings
    }
    
    # Generate visualization
    create_interactive_visualization(
        item_cf, train_df,
        sample_users=25,
        sample_items=40,
        output_path="../visualizations/item_collaborative_filtering.html",
        algorithm_variant="item_collaborative_filtering",
        metrics=viz_metrics_item
    )
    
    # Generate report
    generate_statistical_report(
        item_cf, train_df, test_df, item_metrics,
        {'k_neighbors': 20, 'similarity_metric': 'cosine'},
        output_dir="../reports/collaborative_filtering/item_based"
    )
    
    # Run hyperparameter experiments (commented out for faster execution)
    # Uncomment to run full experiments:
    # run_hyperparameter_experiments(train_df, test_df)
    
    print("\n" + "=" * 70)
    print("✅ COLLABORATIVE FILTERING COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Outputs:")
    print(f"   - Visualizations: ../visualizations/")
    print(f"   - Reports: ../reports/collaborative_filtering/")
    

if __name__ == "__main__":
    main()
