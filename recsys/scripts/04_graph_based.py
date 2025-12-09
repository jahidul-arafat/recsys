"""
04_graph_based.py
==================
Graph-Based Recommendation System

Implements:
- Bipartite Graph construction (Users ↔ Items)
- Personalized PageRank
- Random Walk embeddings (simplified Node2Vec)
- Graph Neural Network-style propagation

Features:
- Interactive graph visualization with Pyvis
- Comprehensive statistical reports
- Path analysis for explainability
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import normalize
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from pyvis.network import Network
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Any, Set
import warnings
warnings.filterwarnings('ignore')

# Create output directories
os.makedirs("../outputs/graph_based", exist_ok=True)
os.makedirs("../visualizations", exist_ok=True)
os.makedirs("../reports/graph_based", exist_ok=True)


class GraphBasedRecommender:
    """
    Graph-Based Recommendation System
    
    Models user-item interactions as a bipartite graph and uses
    graph algorithms for recommendations:
    - Personalized PageRank
    - Random Walk with Restart
    - Node embeddings via random walks
    """
    
    def __init__(
        self,
        method: str = 'pagerank',  # 'pagerank', 'random_walk', 'embedding'
        damping_factor: float = 0.85,
        n_walks: int = 10,
        walk_length: int = 40,
        embedding_dim: int = 64,
        n_iterations: int = 20,
        random_state: int = 42
    ):
        self.method = method
        self.damping_factor = damping_factor
        self.n_walks = n_walks
        self.walk_length = walk_length
        self.embedding_dim = embedding_dim
        self.n_iterations = n_iterations
        self.random_state = random_state
        
        # Graph
        self.graph = None
        self.user_item_graph = None  # Bipartite graph
        
        # Node mappings
        self.user_nodes = set()
        self.item_nodes = set()
        self.node_mapping = {}
        self.reverse_node_mapping = {}
        
        # Embeddings
        self.node_embeddings = None
        
        # User ratings for baseline
        self.user_ratings = defaultdict(dict)
        self.item_ratings = defaultdict(list)
        self.global_mean = 3.0
        
        # Statistics
        self.stats = {
            'n_users': 0,
            'n_items': 0,
            'n_edges': 0,
            'avg_degree': 0,
            'graph_density': 0
        }
        
        np.random.seed(random_state)
    
    def fit(self, ratings_df: pd.DataFrame) -> 'GraphBasedRecommender':
        """
        Build the user-item graph and compute node properties
        
        Args:
            ratings_df: DataFrame with columns [user_id, item_id, rating]
        """
        print(f"\n🔧 Building Graph-Based Recommender ({self.method})...")
        print(f"   damping_factor: {self.damping_factor}")
        
        # Build bipartite graph
        self.graph = nx.Graph()
        
        # Add nodes with types
        users = ratings_df['user_id'].unique()
        items = ratings_df['item_id'].unique()
        
        for u in users:
            node_id = f"U_{u}"
            self.graph.add_node(node_id, bipartite=0, type='user', original_id=u)
            self.user_nodes.add(node_id)
        
        for i in items:
            node_id = f"I_{i}"
            self.graph.add_node(node_id, bipartite=1, type='item', original_id=i)
            self.item_nodes.add(node_id)
        
        # Add edges with weights
        for _, row in ratings_df.iterrows():
            user_node = f"U_{row['user_id']}"
            item_node = f"I_{row['item_id']}"
            weight = row['rating'] / 5.0  # Normalize weight
            
            self.graph.add_edge(user_node, item_node, weight=weight, rating=row['rating'])
            
            # Store ratings
            self.user_ratings[row['user_id']][row['item_id']] = row['rating']
            self.item_ratings[row['item_id']].append(row['rating'])
        
        # Global mean
        self.global_mean = ratings_df['rating'].mean()
        
        # Node mappings
        all_nodes = list(self.graph.nodes())
        self.node_mapping = {node: i for i, node in enumerate(all_nodes)}
        self.reverse_node_mapping = {i: node for node, i in self.node_mapping.items()}
        
        # Statistics
        self.stats['n_users'] = len(users)
        self.stats['n_items'] = len(items)
        self.stats['n_edges'] = self.graph.number_of_edges()
        self.stats['avg_degree'] = 2 * self.stats['n_edges'] / self.graph.number_of_nodes()
        self.stats['graph_density'] = nx.density(self.graph)
        
        print(f"   ✅ Graph built: {self.stats['n_users']} users, {self.stats['n_items']} items")
        print(f"   📊 Edges: {self.stats['n_edges']}, Avg Degree: {self.stats['avg_degree']:.2f}")
        
        # Compute embeddings based on method
        if self.method == 'embedding':
            self._compute_random_walk_embeddings()
        
        return self
    
    def _compute_random_walk_embeddings(self):
        """Compute node embeddings using random walks (simplified Node2Vec)"""
        print("   Computing random walk embeddings...")
        
        n_nodes = len(self.node_mapping)
        
        # Initialize embeddings randomly
        self.node_embeddings = np.random.randn(n_nodes, self.embedding_dim) * 0.1
        
        # Perform random walks and update embeddings
        for walk_num in range(self.n_walks):
            for start_node in self.graph.nodes():
                walk = self._random_walk(start_node, self.walk_length)
                self._update_embeddings_from_walk(walk)
            
            if (walk_num + 1) % 5 == 0:
                print(f"      Completed {walk_num + 1}/{self.n_walks} walks")
        
        # Normalize embeddings
        self.node_embeddings = normalize(self.node_embeddings)
    
    def _random_walk(self, start_node: str, length: int) -> List[str]:
        """Perform a random walk from start_node"""
        walk = [start_node]
        current = start_node
        
        for _ in range(length - 1):
            neighbors = list(self.graph.neighbors(current))
            if not neighbors:
                break
            
            # Weight by edge strength
            weights = [self.graph[current][n].get('weight', 1.0) for n in neighbors]
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            current = np.random.choice(neighbors, p=weights)
            walk.append(current)
        
        return walk
    
    def _update_embeddings_from_walk(self, walk: List[str], window_size: int = 5):
        """Update embeddings based on co-occurrence in random walk"""
        learning_rate = 0.01
        
        for i, node in enumerate(walk):
            node_idx = self.node_mapping[node]
            
            # Context window
            start = max(0, i - window_size)
            end = min(len(walk), i + window_size + 1)
            
            for j in range(start, end):
                if i != j:
                    context_node = walk[j]
                    context_idx = self.node_mapping[context_node]
                    
                    # Simple skip-gram style update
                    diff = self.node_embeddings[context_idx] - self.node_embeddings[node_idx]
                    self.node_embeddings[node_idx] += learning_rate * diff
    
    def personalized_pagerank(
        self,
        user_id: Any,
        alpha: float = None,
        max_iter: int = 100
    ) -> Dict[str, float]:
        """
        Compute Personalized PageRank from a user node
        
        Returns dict of item_node -> score
        """
        if alpha is None:
            alpha = 1 - self.damping_factor
        
        user_node = f"U_{user_id}"
        
        if user_node not in self.graph:
            return {}
        
        # Personalization vector: probability 1 for user node
        personalization = {node: 0.0 for node in self.graph.nodes()}
        personalization[user_node] = 1.0
        
        # Compute PageRank
        try:
            pagerank_scores = nx.pagerank(
                self.graph,
                alpha=self.damping_factor,
                personalization=personalization,
                max_iter=max_iter,
                weight='weight'
            )
        except nx.PowerIterationFailedConvergence:
            # Fall back to simpler computation
            pagerank_scores = nx.pagerank(
                self.graph,
                alpha=self.damping_factor,
                personalization=personalization,
                max_iter=max_iter
            )
        
        # Filter to item nodes only
        item_scores = {
            node: score for node, score in pagerank_scores.items()
            if node in self.item_nodes
        }
        
        return item_scores
    
    def random_walk_with_restart(
        self,
        user_id: Any,
        n_walks: int = 100,
        walk_length: int = 20,
        restart_prob: float = 0.15
    ) -> Dict[str, float]:
        """
        Compute item relevance using Random Walk with Restart
        """
        user_node = f"U_{user_id}"
        
        if user_node not in self.graph:
            return {}
        
        visit_counts = defaultdict(int)
        
        for _ in range(n_walks):
            current = user_node
            
            for _ in range(walk_length):
                # Restart?
                if np.random.random() < restart_prob:
                    current = user_node
                else:
                    neighbors = list(self.graph.neighbors(current))
                    if neighbors:
                        current = np.random.choice(neighbors)
                
                if current in self.item_nodes:
                    visit_counts[current] += 1
        
        # Normalize
        total_visits = sum(visit_counts.values())
        if total_visits > 0:
            return {node: count / total_visits for node, count in visit_counts.items()}
        return {}
    
    def embedding_similarity(self, user_id: Any) -> Dict[str, float]:
        """
        Compute item scores based on embedding similarity
        """
        if self.node_embeddings is None:
            return {}
        
        user_node = f"U_{user_id}"
        if user_node not in self.node_mapping:
            return {}
        
        user_idx = self.node_mapping[user_node]
        user_emb = self.node_embeddings[user_idx]
        
        item_scores = {}
        for item_node in self.item_nodes:
            item_idx = self.node_mapping[item_node]
            item_emb = self.node_embeddings[item_idx]
            
            # Cosine similarity
            similarity = np.dot(user_emb, item_emb)
            item_scores[item_node] = similarity
        
        return item_scores
    
    def predict(self, user_id: Any, item_id: Any) -> float:
        """Predict rating for user-item pair"""
        # Use graph-based score + baseline
        if self.method == 'pagerank':
            scores = self.personalized_pagerank(user_id)
        elif self.method == 'random_walk':
            scores = self.random_walk_with_restart(user_id)
        else:
            scores = self.embedding_similarity(user_id)
        
        item_node = f"I_{item_id}"
        
        if item_node in scores:
            # Convert graph score to rating
            graph_score = scores[item_node]
            
            # Get baseline
            if item_id in self.item_ratings:
                baseline = np.mean(self.item_ratings[item_id])
            else:
                baseline = self.global_mean
            
            # Blend graph score with baseline
            # Higher graph score -> higher predicted rating
            max_score = max(scores.values()) if scores else 1
            normalized_score = graph_score / max_score if max_score > 0 else 0
            
            predicted = baseline + normalized_score * 1.5
            return np.clip(predicted, 1, 5)
        
        # Fall back to item average
        if item_id in self.item_ratings:
            return np.mean(self.item_ratings[item_id])
        return self.global_mean
    
    def recommend(
        self,
        user_id: Any,
        n_recommendations: int = 10,
        exclude_rated: bool = True
    ) -> List[Tuple[Any, float]]:
        """Generate top-N recommendations for a user"""
        # Get graph-based scores
        if self.method == 'pagerank':
            scores = self.personalized_pagerank(user_id)
        elif self.method == 'random_walk':
            scores = self.random_walk_with_restart(user_id)
        else:
            scores = self.embedding_similarity(user_id)
        
        # Get rated items to exclude
        rated_items = set(self.user_ratings[user_id].keys()) if exclude_rated else set()
        
        # Convert to item_id and predicted rating
        recommendations = []
        for item_node, score in scores.items():
            item_id = self.graph.nodes[item_node]['original_id']
            
            if item_id in rated_items:
                continue
            
            # Convert score to predicted rating
            if item_id in self.item_ratings:
                baseline = np.mean(self.item_ratings[item_id])
            else:
                baseline = self.global_mean
            
            max_score = max(scores.values()) if scores else 1
            normalized_score = score / max_score if max_score > 0 else 0
            predicted_rating = np.clip(baseline + normalized_score * 1.5, 1, 5)
            
            recommendations.append((item_id, predicted_rating, score))
        
        # Sort by graph score (original ranking)
        recommendations.sort(key=lambda x: x[2], reverse=True)
        
        return [(item_id, pred) for item_id, pred, _ in recommendations[:n_recommendations]]
    
    def get_path_explanation(
        self,
        user_id: Any,
        item_id: Any,
        max_paths: int = 3
    ) -> List[List[str]]:
        """
        Find paths from user to item for explainability
        """
        user_node = f"U_{user_id}"
        item_node = f"I_{item_id}"
        
        if user_node not in self.graph or item_node not in self.graph:
            return []
        
        try:
            # Find shortest paths
            paths = list(nx.all_shortest_paths(self.graph, user_node, item_node))
            
            # Convert to readable format
            readable_paths = []
            for path in paths[:max_paths]:
                readable = []
                for node in path:
                    node_data = self.graph.nodes[node]
                    readable.append(f"{node_data['type'].title()} {node_data['original_id']}")
                readable_paths.append(readable)
            
            return readable_paths
        except nx.NetworkXNoPath:
            return []
    
    def get_node_importance(self, top_n: int = 20) -> Dict[str, List[Tuple[Any, float]]]:
        """Get most important nodes by various centrality measures"""
        # Degree centrality
        degree_cent = nx.degree_centrality(self.graph)
        
        # Separate users and items
        user_importance = sorted(
            [(self.graph.nodes[n]['original_id'], score) 
             for n, score in degree_cent.items() if n in self.user_nodes],
            key=lambda x: x[1], reverse=True
        )[:top_n]
        
        item_importance = sorted(
            [(self.graph.nodes[n]['original_id'], score) 
             for n, score in degree_cent.items() if n in self.item_nodes],
            key=lambda x: x[1], reverse=True
        )[:top_n]
        
        return {
            'top_users': user_importance,
            'top_items': item_importance
        }
    
    def evaluate(
        self,
        test_df: pd.DataFrame,
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """Evaluate model on test set"""
        print("\n📊 Evaluating model...")
        
        # Rating prediction
        y_true, y_pred = [], []
        
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
                
                recommendations = self.recommend(user_id, n_recommendations=k)
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
        sample_users = list(self.user_ratings.keys())[:100]
        for user_id in sample_users:
            recs = self.recommend(user_id, n_recommendations=10)
            all_recommended.update([item for item, _ in recs])
        
        metrics['Coverage'] = len(all_recommended) / self.stats['n_items']
        
        return metrics


def create_interactive_visualization(
    graph_model: GraphBasedRecommender,
    sample_users: int = 20,
    sample_items: int = 40,
    output_path: str = "../visualizations/graph_based.html",
    metrics: dict = None
):
    """Create enhanced interactive graph visualization with Pyvis"""
    print("\n🎨 Creating enhanced interactive visualization...")
    
    # Try to import enhanced utilities
    try:
        from visualization_utils import generate_enhanced_visualization
        use_enhanced = True
    except ImportError:
        use_enhanced = False
    
    net = Network(
        height="700px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#333333"
    )
    
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
            "font": {"size": 10, "align": "middle"},
            "smooth": {"type": "continuous"}
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -80,
                "springLength": 200,
                "springConstant": 0.03
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
    
    # Sample high-degree nodes
    degree_dict = dict(graph_model.graph.degree())
    edge_count = 0
    
    user_by_degree = sorted(
        [(n, d) for n, d in degree_dict.items() if n in graph_model.user_nodes],
        key=lambda x: x[1], reverse=True
    )[:sample_users]
    
    item_by_degree = sorted(
        [(n, d) for n, d in degree_dict.items() if n in graph_model.item_nodes],
        key=lambda x: x[1], reverse=True
    )[:sample_items]
    
    sampled_users = set([n for n, _ in user_by_degree])
    sampled_items = set([n for n, _ in item_by_degree])
    
    # Add user nodes with LARGER sizes
    for node, degree in user_by_degree:
        original_id = graph_model.graph.nodes[node]['original_id']
        net.add_node(
            node,
            label=f"U{original_id}",
            color="#4285f4",
            size=22 + min(degree * 0.8, 18),  # Larger base size
            title=f"<div style='padding:10px;'>"
                  f"<strong style='font-size:14px;'>👤 User {original_id}</strong><br><br>"
                  f"<b>Degree:</b> {degree}<br>"
                  f"<b>Ratings:</b> {len(graph_model.user_ratings[original_id])}"
                  f"</div>",
            group="user",
            shape="dot"
        )
    
    # Add item nodes with LARGER sizes
    for node, degree in item_by_degree:
        original_id = graph_model.graph.nodes[node]['original_id']
        avg_rating = np.mean(graph_model.item_ratings[original_id]) if graph_model.item_ratings[original_id] else 0
        
        net.add_node(
            node,
            label=f"I{original_id}",
            color="#34a853",
            size=20 + min(degree * 0.6, 15),  # Larger base size
            title=f"<div style='padding:10px;'>"
                  f"<strong style='font-size:14px;'>📦 Item {original_id}</strong><br><br>"
                  f"<b>Degree:</b> {degree}<br>"
                  f"<b>Avg Rating:</b> {avg_rating:.2f}"
                  f"</div>",
            group="item",
            shape="dot"
        )
    
    # Add edges with labels
    for user_node in sampled_users:
        for item_node in graph_model.graph.neighbors(user_node):
            if item_node in sampled_items:
                edge_data = graph_model.graph[user_node][item_node]
                rating = edge_data.get('rating', 3)
                
                # Color by rating
                if rating >= 4:
                    color = "#34a853"  # Green for high
                    quality = "Good"
                elif rating <= 2:
                    color = "#ea4335"  # Red for low
                    quality = "Poor"
                else:
                    color = "#fbbc04"  # Yellow for medium
                    quality = "Average"
                
                label_text = f"rated {rating:.0f}/5"
                net.add_edge(
                    user_node,
                    item_node,
                    color=color,
                    width=1.5 + rating / 2,
                    label=label_text,
                    title=f"<div style='padding:8px;'><b>📝 Rating Edge</b><br><br>"
                          f"<b>Rating:</b> {rating}/5<br>"
                          f"<b>Quality:</b> {quality}<br>"
                          f"<b>Edge Weight:</b> Used in random walk probability</div>",
                    font={"size": 9, "color": color, "background": "rgba(255,255,255,0.8)"}
                )
                edge_count += 1
    
    # Add recommendation edges for first user
    if user_by_degree:
        first_user_node = user_by_degree[0][0]
        first_user_id = graph_model.graph.nodes[first_user_node]['original_id']
        recs = graph_model.recommend(first_user_id, n_recommendations=5)
        
        for item_id, score in recs:
            item_node = f"I_{item_id}"
            if item_node in sampled_items:
                label_text = f"PPR {score:.2f}"
                net.add_edge(
                    first_user_node,
                    item_node,
                    color="#9c27b0",
                    width=3,
                    dashes=True,
                    label=label_text,
                    title=f"<div style='padding:8px;'><b>🎯 Recommendation</b><br><br>"
                          f"<b>PPR Score:</b> {score:.4f}<br>"
                          f"<b>Method:</b> Personalized PageRank<br>"
                          f"<b>Meaning:</b> High random walk visitation probability</div>",
                    font={"size": 9, "color": "#9c27b0", "background": "rgba(255,255,255,0.8)"}
                )
                edge_count += 1
    
    # Graph stats
    graph_stats = {
        "Users": len(sampled_users),
        "Items": len(sampled_items),
        "Edges": edge_count,
        "Total Nodes": graph_model.graph.number_of_nodes()
    }
    
    node_types = [
        {"color": "#4285f4", "label": "Users", "description": "User nodes (size=degree)"},
        {"color": "#34a853", "label": "Items", "description": "Item nodes (size=degree)"}
    ]
    
    edge_types = [
        {"color": "#34a853", "label": "High Rating (4-5)", "style": "solid"},
        {"color": "#fbbc04", "label": "Medium (3)", "style": "solid"},
        {"color": "#ea4335", "label": "Low (1-2)", "style": "solid"},
        {"color": "#9c27b0", "label": "Recommendation", "style": "dashed"}
    ]
    
    if use_enhanced:
        generate_enhanced_visualization(
            net=net,
            algorithm_key="graph_based",
            output_path=output_path,
            metrics=metrics,
            graph_stats=graph_stats,
            node_types=node_types,
            edge_types=edge_types
        )
    else:
        # Fallback legend
        legend_html = """
        <div style="position: absolute; top: 10px; right: 10px; background: white; 
                    padding: 20px; border-radius: 10px; box-shadow: 0 2px 15px rgba(0,0,0,0.15);
                    font-family: Arial, sans-serif; font-size: 13px; max-width: 250px;">
            <h3 style="margin: 0 0 15px 0; font-size: 16px; color: #333; border-bottom: 2px solid #e94560; padding-bottom: 10px;">
                Graph-Based Recommender
            </h3>
            <p style="font-size: 12px; color: #666; margin: 0 0 10px 0;">
                Random Walk / PageRank on bipartite graph
            </p>
            <div style="margin: 8px 0;">
                <span style="display: inline-block; width: 18px; height: 18px; 
                            background: #4285f4; border-radius: 50%; margin-right: 10px; vertical-align: middle;"></span>
                <strong>Users</strong>
            </div>
            <div style="margin: 8px 0;">
                <span style="display: inline-block; width: 18px; height: 18px; 
                            background: #34a853; border-radius: 50%; margin-right: 10px; vertical-align: middle;"></span>
                <strong>Items</strong>
            </div>
            <hr style="margin: 12px 0; border: none; border-top: 1px solid #ddd;">
            <div style="margin: 8px 0;">
                <span style="display: inline-block; width: 30px; height: 3px; 
                            background: #34a853; margin-right: 10px; vertical-align: middle;"></span>
                High Rating (4-5)
            </div>
            <div style="margin: 8px 0;">
                <span style="display: inline-block; width: 30px; height: 3px; 
                            background: #fbbc04; margin-right: 10px; vertical-align: middle;"></span>
                Medium Rating (3)
            </div>
            <div style="margin: 8px 0;">
                <span style="display: inline-block; width: 30px; height: 3px; 
                            background: #ea4335; margin-right: 10px; vertical-align: middle;"></span>
                Low Rating (1-2)
            </div>
            <div style="margin: 8px 0;">
                <span style="display: inline-block; width: 30px; height: 3px; 
                            background: #9c27b0; margin-right: 10px; vertical-align: middle;
                            border-style: dashed; border-width: 2px 0 0 0; background: none; border-color: #9c27b0;"></span>
                Recommendation
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
    graph_model: GraphBasedRecommender,
    metrics: Dict[str, float],
    output_dir: str = "../reports/graph_based"
):
    """Generate comprehensive statistical report"""
    print("\n📈 Generating statistical report...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figures
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Graph-Based Recommendation Analysis ({graph_model.method})', fontsize=14)
    
    # 1. Degree distribution
    ax1 = axes[0, 0]
    degrees = [d for n, d in graph_model.graph.degree()]
    ax1.hist(degrees, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    ax1.set_xlabel('Degree')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Node Degree Distribution')
    ax1.set_xlim(0, np.percentile(degrees, 95))
    
    # 2. User vs Item degree comparison
    ax2 = axes[0, 1]
    user_degrees = [d for n, d in graph_model.graph.degree() if n in graph_model.user_nodes]
    item_degrees = [d for n, d in graph_model.graph.degree() if n in graph_model.item_nodes]
    ax2.boxplot([user_degrees, item_degrees], labels=['Users', 'Items'])
    ax2.set_ylabel('Degree')
    ax2.set_title('Degree by Node Type')
    
    # 3. Rating distribution on edges
    ax3 = axes[0, 2]
    edge_ratings = [d['rating'] for u, v, d in graph_model.graph.edges(data=True)]
    ax3.hist(edge_ratings, bins=5, color='coral', edgecolor='white', alpha=0.7)
    ax3.set_xlabel('Rating')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Edge Rating Distribution')
    
    # 4. Connected components
    ax4 = axes[1, 0]
    components = list(nx.connected_components(graph_model.graph))
    component_sizes = [len(c) for c in components]
    ax4.bar(range(min(10, len(component_sizes))), sorted(component_sizes, reverse=True)[:10], 
            color='seagreen')
    ax4.set_xlabel('Component Rank')
    ax4.set_ylabel('Size')
    ax4.set_title(f'Largest Connected Components ({len(components)} total)')
    
    # 5. Evaluation metrics
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
    
    # 6. Path length distribution (sample)
    ax6 = axes[1, 2]
    sample_users = list(graph_model.user_ratings.keys())[:20]
    sample_items = list(graph_model.item_ratings.keys())[:20]
    path_lengths = []
    for u in sample_users:
        for i in sample_items:
            try:
                length = nx.shortest_path_length(graph_model.graph, f"U_{u}", f"I_{i}")
                path_lengths.append(length)
            except nx.NetworkXNoPath:
                pass
    
    if path_lengths:
        ax6.hist(path_lengths, bins=range(1, max(path_lengths) + 2), 
                color='purple', edgecolor='white', alpha=0.7)
        ax6.set_xlabel('Path Length')
        ax6.set_ylabel('Frequency')
        ax6.set_title('User-Item Path Lengths (sample)')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/analysis_figures.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Text report
    report = f"""
================================================================================
GRAPH-BASED RECOMMENDATION - STATISTICAL REPORT
================================================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CONFIGURATION
-------------
Method: {graph_model.method}
Damping Factor: {graph_model.damping_factor}
N Walks: {graph_model.n_walks}
Walk Length: {graph_model.walk_length}
Embedding Dim: {graph_model.embedding_dim}

GRAPH STATISTICS
----------------
Number of Users: {graph_model.stats['n_users']:,}
Number of Items: {graph_model.stats['n_items']:,}
Number of Edges: {graph_model.stats['n_edges']:,}
Average Degree: {graph_model.stats['avg_degree']:.2f}
Graph Density: {graph_model.stats['graph_density']:.6f}
Connected Components: {nx.number_connected_components(graph_model.graph)}

DEGREE STATISTICS
-----------------
Mean User Degree: {np.mean(user_degrees):.2f}
Max User Degree: {max(user_degrees)}
Mean Item Degree: {np.mean(item_degrees):.2f}
Max Item Degree: {max(item_degrees)}

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

================================================================================
"""
    
    with open(f"{output_dir}/statistical_report.txt", 'w') as f:
        f.write(report)
    
    with open(f"{output_dir}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"   ✅ Report saved to: {output_dir}/")
    
    return report


def main():
    """Main execution function"""
    print("=" * 70)
    print("GRAPH-BASED RECOMMENDER SYSTEM")
    print("=" * 70)
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    from data_loader import DatasetGenerator
    
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
    
    # Test different methods
    methods = ['pagerank', 'random_walk']
    
    for method in methods:
        print("\n" + "=" * 70)
        print(f"TRAINING {method.upper()} MODEL")
        print("=" * 70)
        
        model = GraphBasedRecommender(
            method=method,
            damping_factor=0.85,
            n_walks=10,
            walk_length=20
        )
        model.fit(train_df)
        
        metrics = model.evaluate(test_df)
        print(f"\n📊 {method.upper()} Metrics:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        create_interactive_visualization(
            model,
            sample_users=25,
            sample_items=50,
            output_path=f"../visualizations/graph_{method}.html"
        )
        
        generate_statistical_report(
            model, metrics,
            output_dir=f"../reports/graph_based/{method}"
        )
        
        # Show path explanation example
        print("\n📝 Path Explanation Example:")
        sample_user = list(model.user_ratings.keys())[0]
        recs = model.recommend(sample_user, n_recommendations=1)
        if recs:
            paths = model.get_path_explanation(sample_user, recs[0][0])
            print(f"   User {sample_user} → Item {recs[0][0]}:")
            for path in paths[:2]:
                print(f"   Path: {' → '.join(path)}")
    
    print("\n" + "=" * 70)
    print("✅ GRAPH-BASED RECOMMENDATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
