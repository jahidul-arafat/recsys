"""
03_content_based.py
====================
Content-Based Recommendation System

Implements:
- TF-IDF based similarity
- Feature-based similarity (categorical/numerical)
- Hybrid content features

Features:
- Interactive graph visualization
- Comprehensive statistical reports
- Hyperparameter exploration
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Create output directories
os.makedirs("../outputs/content_based", exist_ok=True)
os.makedirs("../visualizations", exist_ok=True)
os.makedirs("../reports/content_based", exist_ok=True)


class ContentBasedRecommender:
    """
    Content-Based Recommendation System
    
    Recommends items based on:
    1. Item content similarity (TF-IDF on descriptions)
    2. Item feature similarity (genres, categories, etc.)
    3. User profile matching
    """
    
    def __init__(
        self,
        tfidf_max_features: int = 500,
        tfidf_ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        use_features: bool = True
    ):
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range = tfidf_ngram_range
        self.min_df = min_df
        self.use_features = use_features
        
        # Models
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.feature_matrix = None
        self.combined_similarity = None
        
        # Data
        self.items_df = None
        self.item_mapping = {}
        self.reverse_item_mapping = {}
        
        # User profiles
        self.user_profiles = {}  # user_id -> weighted item vectors
        self.user_ratings = defaultdict(dict)  # user_id -> {item_id: rating}
        
        # Statistics
        self.stats = {
            'n_items': 0,
            'n_users': 0,
            'n_ratings': 0,
            'tfidf_vocab_size': 0,
            'n_features': 0
        }
    
    def fit(
        self,
        items_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        text_column: str = 'description',
        feature_columns: List[str] = None
    ) -> 'ContentBasedRecommender':
        """
        Fit the content-based recommender
        
        Args:
            items_df: Item metadata DataFrame
            ratings_df: User-item ratings DataFrame
            text_column: Column containing item text/description
            feature_columns: List of categorical/numerical feature columns
        """
        print("\n🔧 Fitting Content-Based Recommender...")
        
        self.items_df = items_df.copy()
        
        # Create item mapping
        self.item_mapping = {item: i for i, item in enumerate(items_df['item_id'])}
        self.reverse_item_mapping = {i: item for item, i in self.item_mapping.items()}
        
        n_items = len(items_df)
        self.stats['n_items'] = n_items
        
        # Build TF-IDF matrix from text
        if text_column in items_df.columns:
            print(f"   Building TF-IDF matrix from '{text_column}'...")
            self._build_tfidf_matrix(items_df[text_column].fillna(''))
        else:
            print(f"   No text column '{text_column}', creating from titles/genres...")
            # Create pseudo-descriptions from available columns
            text_data = self._create_text_features(items_df)
            self._build_tfidf_matrix(text_data)
        
        # Build feature matrix
        if self.use_features and feature_columns:
            print(f"   Building feature matrix from {feature_columns}...")
            self._build_feature_matrix(items_df, feature_columns)
        elif self.use_features:
            # Auto-detect feature columns
            feature_columns = self._auto_detect_features(items_df)
            if feature_columns:
                print(f"   Auto-detected features: {feature_columns}")
                self._build_feature_matrix(items_df, feature_columns)
        
        # Combine similarities
        self._compute_combined_similarity()
        
        # Build user profiles from ratings
        print("   Building user profiles...")
        self._build_user_profiles(ratings_df)
        
        print(f"   ✅ Model fitted: {n_items} items")
        print(f"   📊 TF-IDF vocabulary: {self.stats['tfidf_vocab_size']} terms")
        
        return self
    
    def _create_text_features(self, items_df: pd.DataFrame) -> pd.Series:
        """Create text features from available columns"""
        text_parts = []
        
        for col in ['title', 'genres', 'category', 'brand', 'author', 'publisher']:
            if col in items_df.columns:
                text_parts.append(items_df[col].fillna('').astype(str))
        
        if text_parts:
            return pd.concat(text_parts, axis=1).apply(lambda x: ' '.join(x), axis=1)
        else:
            return pd.Series([''] * len(items_df))
    
    def _auto_detect_features(self, items_df: pd.DataFrame) -> List[str]:
        """Auto-detect categorical feature columns"""
        feature_cols = []
        
        for col in items_df.columns:
            if col in ['item_id', 'title', 'description']:
                continue
            
            dtype = items_df[col].dtype
            n_unique = items_df[col].nunique()
            
            # Categorical if string or few unique values
            if dtype == 'object' or (n_unique < 50 and n_unique > 1):
                feature_cols.append(col)
        
        return feature_cols[:5]  # Limit to 5 features
    
    def _build_tfidf_matrix(self, text_data: pd.Series):
        """Build TF-IDF matrix from text data"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            ngram_range=self.tfidf_ngram_range,
            min_df=self.min_df,
            stop_words='english'
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_data)
        self.stats['tfidf_vocab_size'] = len(self.tfidf_vectorizer.vocabulary_)
    
    def _build_feature_matrix(self, items_df: pd.DataFrame, feature_columns: List[str]):
        """Build feature matrix from categorical/numerical columns"""
        feature_matrices = []
        
        for col in feature_columns:
            if items_df[col].dtype == 'object':
                # Handle pipe-separated values (like genres)
                if items_df[col].str.contains('|', regex=False).any():
                    # Multi-label encoding
                    mlb_matrix = self._multi_label_encode(items_df[col])
                    feature_matrices.append(mlb_matrix)
                else:
                    # One-hot encoding
                    encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
                    encoded = encoder.fit_transform(items_df[[col]])
                    feature_matrices.append(encoded.toarray())
            else:
                # Numerical - normalize
                scaler = StandardScaler()
                scaled = scaler.fit_transform(items_df[[col]].fillna(0))
                feature_matrices.append(scaled)
        
        if feature_matrices:
            self.feature_matrix = np.hstack(feature_matrices)
            self.stats['n_features'] = self.feature_matrix.shape[1]
    
    def _multi_label_encode(self, series: pd.Series) -> np.ndarray:
        """Encode multi-label column (pipe-separated values)"""
        # Get all unique labels
        all_labels = set()
        for val in series.dropna():
            all_labels.update(val.split('|'))
        
        all_labels = sorted(all_labels)
        label_to_idx = {label: i for i, label in enumerate(all_labels)}
        
        # Create binary matrix
        matrix = np.zeros((len(series), len(all_labels)))
        
        for i, val in enumerate(series):
            if pd.notna(val):
                for label in val.split('|'):
                    if label in label_to_idx:
                        matrix[i, label_to_idx[label]] = 1
        
        return matrix
    
    def _compute_combined_similarity(self):
        """Compute combined similarity matrix from TF-IDF and features"""
        # TF-IDF similarity
        tfidf_sim = cosine_similarity(self.tfidf_matrix)
        
        if self.feature_matrix is not None:
            # Feature similarity
            feature_sim = cosine_similarity(self.feature_matrix)
            # Combine (weighted average)
            self.combined_similarity = 0.6 * tfidf_sim + 0.4 * feature_sim
        else:
            self.combined_similarity = tfidf_sim
        
        # Set diagonal to 0
        np.fill_diagonal(self.combined_similarity, 0)
    
    def _build_user_profiles(self, ratings_df: pd.DataFrame):
        """Build user profiles based on their rating history"""
        self.stats['n_users'] = ratings_df['user_id'].nunique()
        self.stats['n_ratings'] = len(ratings_df)
        
        # Store raw ratings
        for _, row in ratings_df.iterrows():
            self.user_ratings[row['user_id']][row['item_id']] = row['rating']
        
        # Build weighted profile vectors
        for user_id, item_ratings in self.user_ratings.items():
            profile_vector = np.zeros(self.tfidf_matrix.shape[1])
            total_weight = 0
            
            for item_id, rating in item_ratings.items():
                if item_id in self.item_mapping:
                    idx = self.item_mapping[item_id]
                    # Weight by normalized rating (higher ratings = stronger preference)
                    weight = (rating - 2.5) / 2.5  # Normalize to [-1, 1]
                    profile_vector += weight * self.tfidf_matrix[idx].toarray().flatten()
                    total_weight += abs(weight)
            
            if total_weight > 0:
                profile_vector /= total_weight
            
            self.user_profiles[user_id] = profile_vector
    
    def get_similar_items(self, item_id: Any, n: int = 10) -> List[Tuple[Any, float]]:
        """Get most similar items based on content"""
        if item_id not in self.item_mapping:
            return []
        
        idx = self.item_mapping[item_id]
        similarities = self.combined_similarity[idx]
        
        top_idx = np.argsort(similarities)[-n:][::-1]
        
        return [(self.reverse_item_mapping[i], similarities[i]) for i in top_idx]
    
    def predict(self, user_id: Any, item_id: Any) -> float:
        """Predict user rating for an item"""
        if item_id not in self.item_mapping:
            return 3.0
        
        idx = self.item_mapping[item_id]
        item_vector = self.tfidf_matrix[idx].toarray().flatten()
        
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            # Cosine similarity between user profile and item
            norm_profile = np.linalg.norm(profile)
            norm_item = np.linalg.norm(item_vector)
            
            if norm_profile > 0 and norm_item > 0:
                similarity = np.dot(profile, item_vector) / (norm_profile * norm_item)
                # Convert to rating scale [1, 5]
                rating = 3 + similarity * 2
                return np.clip(rating, 1, 5)
        
        # Fall back to average rating for this item
        item_ratings = [r for uid, ratings in self.user_ratings.items() 
                       for iid, r in ratings.items() if iid == item_id]
        if item_ratings:
            return np.mean(item_ratings)
        return 3.0
    
    def recommend(
        self,
        user_id: Any,
        n_recommendations: int = 10,
        exclude_rated: bool = True
    ) -> List[Tuple[Any, float]]:
        """Generate recommendations for a user"""
        if user_id not in self.user_profiles:
            return self._get_popular_items(n_recommendations)
        
        profile = self.user_profiles[user_id]
        rated_items = set(self.user_ratings[user_id].keys())
        
        scores = []
        
        for idx in range(self.stats['n_items']):
            item_id = self.reverse_item_mapping[idx]
            
            if exclude_rated and item_id in rated_items:
                continue
            
            item_vector = self.tfidf_matrix[idx].toarray().flatten()
            
            # Calculate similarity to user profile
            norm_profile = np.linalg.norm(profile)
            norm_item = np.linalg.norm(item_vector)
            
            if norm_profile > 0 and norm_item > 0:
                similarity = np.dot(profile, item_vector) / (norm_profile * norm_item)
            else:
                similarity = 0
            
            # Convert to predicted rating
            pred_rating = np.clip(3 + similarity * 2, 1, 5)
            scores.append((item_id, pred_rating))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n_recommendations]
    
    def _get_popular_items(self, n: int) -> List[Tuple[Any, float]]:
        """Return most rated items (cold start)"""
        item_counts = defaultdict(int)
        item_ratings = defaultdict(list)
        
        for uid, ratings in self.user_ratings.items():
            for iid, r in ratings.items():
                item_counts[iid] += 1
                item_ratings[iid].append(r)
        
        popular = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:n]
        return [(iid, np.mean(item_ratings[iid])) for iid, _ in popular]
    
    def explain_recommendation(self, user_id: Any, item_id: Any) -> Dict[str, Any]:
        """Explain why an item was recommended"""
        if item_id not in self.item_mapping:
            return {'error': 'Item not found'}
        
        idx = self.item_mapping[item_id]
        item = self.items_df.iloc[idx]
        
        explanation = {
            'item_id': item_id,
            'item_info': item.to_dict(),
            'predicted_rating': self.predict(user_id, item_id)
        }
        
        # Find similar items user has rated highly
        if user_id in self.user_ratings:
            rated_items = self.user_ratings[user_id]
            high_rated = {iid: r for iid, r in rated_items.items() if r >= 4}
            
            similar_rated = []
            for rated_id, rating in high_rated.items():
                if rated_id in self.item_mapping:
                    rated_idx = self.item_mapping[rated_id]
                    sim = self.combined_similarity[idx, rated_idx]
                    if sim > 0.1:
                        similar_rated.append({
                            'item_id': rated_id,
                            'user_rating': rating,
                            'similarity': sim
                        })
            
            similar_rated.sort(key=lambda x: x['similarity'], reverse=True)
            explanation['similar_to_liked'] = similar_rated[:5]
        
        # Top TF-IDF terms for this item
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = self.tfidf_matrix[idx].toarray().flatten()
        top_term_idx = np.argsort(tfidf_scores)[-10:][::-1]
        explanation['top_terms'] = [(feature_names[i], tfidf_scores[i]) for i in top_term_idx]
        
        return explanation
    
    def evaluate(
        self,
        test_df: pd.DataFrame,
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """Evaluate model on test set"""
        print("\n📊 Evaluating model...")
        
        # Rating prediction metrics
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
        sample_users = list(self.user_profiles.keys())[:100]
        for user_id in sample_users:
            recs = self.recommend(user_id, n_recommendations=10)
            all_recommended.update([item for item, _ in recs])
        
        metrics['Coverage'] = len(all_recommended) / self.stats['n_items']
        
        return metrics


def create_interactive_visualization(
    cb_model: ContentBasedRecommender,
    sample_items: int = 80,
    output_path: str = "../visualizations/content_based.html",
    metrics: dict = None
):
    """Create enhanced interactive graph showing item content similarities"""
    print("\n🎨 Creating enhanced interactive visualization...")
    
    # Try to import enhanced utilities
    try:
        from visualization_utils import (
            generate_enhanced_visualization
        )
        use_enhanced = True
    except ImportError:
        use_enhanced = False
    
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
            "font": {"size": 10, "align": "middle"},
            "smooth": {"type": "continuous"}
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -60,
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
    
    # Sample items
    item_ids = list(cb_model.item_mapping.keys())[:sample_items]
    edge_count = 0
    genre_counts = {}
    
    # Add item nodes with LARGER sizes
    for item_id in item_ids:
        idx = cb_model.item_mapping[item_id]
        item_data = cb_model.items_df.iloc[idx]
        
        # Get item info
        title = item_data.get('title', f'Item {item_id}')
        genres = item_data.get('genres', item_data.get('category', 'Unknown'))
        
        # Color by first genre/category
        if pd.notna(genres):
            first_genre = str(genres).split('|')[0] if '|' in str(genres) else str(genres)
        else:
            first_genre = 'Unknown'
        
        # Track genres for stats
        genre_counts[first_genre] = genre_counts.get(first_genre, 0) + 1
        
        # Hash genre to color
        color_hash = hash(first_genre) % 360
        color = f"hsl({color_hash}, 70%, 50%)"
        
        # Calculate node size based on connections (will be updated)
        base_size = 22
        
        net.add_node(
            f"I_{item_id}",
            label=str(title)[:12] if len(str(title)) > 12 else str(title),
            color=color,
            size=base_size,
            title=f"<div style='padding:10px;'>"
                  f"<strong style='font-size:14px;'>📦 {title}</strong><br><br>"
                  f"<b>Genre/Category:</b> {genres}<br>"
                  f"<b>Item ID:</b> {item_id}<br>"
                  f"<b>Group:</b> {first_genre}"
                  f"</div>",
            group=first_genre,
            shape="dot"
        )
    
    # Add similarity edges with descriptive labels
    for item_id in item_ids:
        similar = cb_model.get_similar_items(item_id, n=3)
        for sim_item_id, sim_score in similar:
            if f"I_{sim_item_id}" in [n['id'] for n in net.nodes] and sim_score > 0.1:
                label_text = f"sim {sim_score:.2f}"
                net.add_edge(
                    f"I_{item_id}",
                    f"I_{sim_item_id}",
                    value=sim_score * 4,
                    width=1.5 + sim_score * 3,
                    label=label_text,
                    title=f"<div style='padding:8px;'><b>🔗 Content Similarity</b><br><br>"
                          f"<b>Items:</b> {item_id} ↔ {sim_item_id}<br>"
                          f"<b>TF-IDF Cosine:</b> {sim_score:.3f}<br>"
                          f"<b>Meaning:</b> Items share similar content features (genres, keywords)</div>",
                    color={'color': '#888888', 'opacity': 0.7},
                    font={"size": 9, "color": "#666666", "background": "rgba(255,255,255,0.8)"}
                )
                edge_count += 1
    
    # Graph stats
    graph_stats = {
        "Items": len(item_ids),
        "Edges": edge_count,
        "Categories": len(genre_counts),
        "TF-IDF Features": cb_model.tfidf_matrix.shape[1] if hasattr(cb_model, 'tfidf_matrix') else 'N/A'
    }
    
    # Node types (top genres)
    top_genres = sorted(genre_counts.items(), key=lambda x: -x[1])[:5]
    node_types = []
    for genre, count in top_genres:
        color_hash = hash(genre) % 360
        node_types.append({
            "color": f"hsl({color_hash}, 70%, 50%)",
            "label": genre,
            "description": f"{count} items"
        })
    
    edge_types = [
        {"color": "#888888", "label": "Content Similarity", "style": "solid", "description": "TF-IDF cosine similarity"}
    ]
    
    if use_enhanced:
        generate_enhanced_visualization(
            net=net,
            algorithm_key="content_based",
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
                Content-Based Filtering
            </h3>
            <p style="font-size: 12px; color: #666; margin: 0 0 15px 0;">
                Items colored by genre/category.<br>
                Edges show TF-IDF content similarity.
            </p>
            <div style="margin: 8px 0;">
                <span style="display: inline-block; width: 30px; height: 3px; 
                            background: #888; margin-right: 10px; vertical-align: middle;"></span>
                Content Similarity
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
    cb_model: ContentBasedRecommender,
    metrics: Dict[str, float],
    output_dir: str = "../reports/content_based"
):
    """Generate comprehensive statistical report"""
    print("\n📈 Generating statistical report...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figures
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Content-Based Filtering Analysis', fontsize=14)
    
    # 1. TF-IDF term frequency
    ax1 = axes[0, 0]
    tfidf_sum = np.array(cb_model.tfidf_matrix.sum(axis=0)).flatten()
    top_idx = np.argsort(tfidf_sum)[-20:]
    feature_names = cb_model.tfidf_vectorizer.get_feature_names_out()
    ax1.barh(range(20), tfidf_sum[top_idx], color='steelblue')
    ax1.set_yticks(range(20))
    ax1.set_yticklabels([feature_names[i] for i in top_idx], fontsize=8)
    ax1.set_xlabel('TF-IDF Sum')
    ax1.set_title('Top 20 TF-IDF Terms')
    
    # 2. Similarity distribution
    ax2 = axes[0, 1]
    sim_values = cb_model.combined_similarity.flatten()
    sim_values = sim_values[sim_values > 0]
    if len(sim_values) > 10000:
        sim_values = np.random.choice(sim_values, 10000)
    ax2.hist(sim_values, bins=50, color='coral', edgecolor='white', alpha=0.7)
    ax2.set_xlabel('Similarity Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Item Similarity Distribution')
    
    # 3. User profile norms
    ax3 = axes[0, 2]
    profile_norms = [np.linalg.norm(p) for p in cb_model.user_profiles.values()]
    ax3.hist(profile_norms, bins=30, color='seagreen', edgecolor='white', alpha=0.7)
    ax3.set_xlabel('Profile Vector Norm')
    ax3.set_ylabel('Number of Users')
    ax3.set_title('User Profile Strength Distribution')
    
    # 4. Items per user
    ax4 = axes[1, 0]
    items_per_user = [len(ratings) for ratings in cb_model.user_ratings.values()]
    ax4.hist(items_per_user, bins=30, color='purple', edgecolor='white', alpha=0.7)
    ax4.set_xlabel('Number of Rated Items')
    ax4.set_ylabel('Number of Users')
    ax4.set_title('Items Rated per User')
    
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
    
    # 6. TF-IDF matrix sparsity visualization
    ax6 = axes[1, 2]
    sample_tfidf = cb_model.tfidf_matrix[:50, :50].toarray()
    ax6.imshow(sample_tfidf, cmap='Blues', aspect='auto')
    ax6.set_xlabel('Terms (sample)')
    ax6.set_ylabel('Items (sample)')
    ax6.set_title('TF-IDF Matrix (50x50 sample)')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/analysis_figures.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Text report
    report = f"""
================================================================================
CONTENT-BASED FILTERING - STATISTICAL REPORT
================================================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CONFIGURATION
-------------
TF-IDF Max Features: {cb_model.tfidf_max_features}
TF-IDF N-gram Range: {cb_model.tfidf_ngram_range}
Min Document Frequency: {cb_model.min_df}
Use Additional Features: {cb_model.use_features}

DATASET STATISTICS
------------------
Number of Items: {cb_model.stats['n_items']:,}
Number of Users: {cb_model.stats['n_users']:,}
Number of Ratings: {cb_model.stats['n_ratings']:,}

TF-IDF STATISTICS
-----------------
Vocabulary Size: {cb_model.stats['tfidf_vocab_size']:,}
Matrix Shape: {cb_model.tfidf_matrix.shape}
Matrix Sparsity: {1 - cb_model.tfidf_matrix.nnz / (cb_model.tfidf_matrix.shape[0] * cb_model.tfidf_matrix.shape[1]):.4%}

SIMILARITY STATISTICS
---------------------
Mean Similarity: {cb_model.combined_similarity.mean():.4f}
Max Similarity: {cb_model.combined_similarity.max():.4f}
Median Similarity: {np.median(cb_model.combined_similarity):.4f}

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
    print("CONTENT-BASED RECOMMENDER SYSTEM")
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
    items_df = dataset['items']
    
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
    print(f"   Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    print("\n" + "=" * 70)
    print("TRAINING CONTENT-BASED MODEL")
    print("=" * 70)
    
    model = ContentBasedRecommender(
        tfidf_max_features=500,
        tfidf_ngram_range=(1, 2),
        use_features=True
    )
    model.fit(items_df, train_df, feature_columns=['genres'])
    
    metrics = model.evaluate(test_df)
    print("\n📊 Content-Based Metrics:")
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    create_interactive_visualization(
        model,
        sample_items=80,
        output_path="../visualizations/content_based.html"
    )
    
    generate_statistical_report(model, metrics)
    
    # Show explanation example
    print("\n📝 Example Recommendation Explanation:")
    sample_user = list(model.user_profiles.keys())[0]
    recs = model.recommend(sample_user, n_recommendations=3)
    if recs:
        explanation = model.explain_recommendation(sample_user, recs[0][0])
        print(f"   For User {sample_user}, Item {recs[0][0]}:")
        print(f"   Predicted Rating: {explanation['predicted_rating']:.2f}")
        if explanation.get('top_terms'):
            print(f"   Top Terms: {[t[0] for t in explanation['top_terms'][:5]]}")
    
    print("\n" + "=" * 70)
    print("✅ CONTENT-BASED FILTERING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
