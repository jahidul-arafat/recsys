"""
06_hybrid.py
=============
Hybrid Recommendation System

Implements:
- Weighted Ensemble (combining multiple recommenders)
- Switching Hybrid (context-based algorithm selection)
- Feature Combination (unified feature space)

Features:
- Interactive visualization showing contribution of each algorithm
- Comprehensive statistical reports
- Explainability of hybrid decisions
"""

import numpy as np
import pandas as pd
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
os.makedirs("../outputs/hybrid", exist_ok=True)
os.makedirs("../visualizations", exist_ok=True)
os.makedirs("../reports/hybrid", exist_ok=True)


class SimpleCollaborativeFilter:
    """Simplified CF for hybrid use"""
    
    def __init__(self, k_neighbors=20):
        self.k_neighbors = k_neighbors
        self.user_item_matrix = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.global_mean = 3.0
        self.item_means = {}
        
    def fit(self, ratings_df):
        unique_users = ratings_df['user_id'].unique()
        unique_items = ratings_df['item_id'].unique()
        
        self.user_mapping = {u: i for i, u in enumerate(unique_users)}
        self.item_mapping = {it: i for i, it in enumerate(unique_items)}
        
        n_users, n_items = len(unique_users), len(unique_items)
        self.user_item_matrix = np.zeros((n_users, n_items))
        
        for _, row in ratings_df.iterrows():
            u_idx = self.user_mapping[row['user_id']]
            i_idx = self.item_mapping[row['item_id']]
            self.user_item_matrix[u_idx, i_idx] = row['rating']
        
        self.global_mean = ratings_df['rating'].mean()
        
        for i_idx in range(n_items):
            ratings = self.user_item_matrix[:, i_idx]
            mask = ratings > 0
            self.item_means[i_idx] = ratings[mask].mean() if mask.sum() > 0 else self.global_mean
        
        return self
    
    def predict(self, user_id, item_id):
        if user_id not in self.user_mapping or item_id not in self.item_mapping:
            return self.global_mean
        
        i_idx = self.item_mapping[item_id]
        return self.item_means.get(i_idx, self.global_mean)


class SimpleContentBased:
    """Simplified content-based for hybrid use"""
    
    def __init__(self):
        self.item_features = {}
        self.user_profiles = defaultdict(lambda: defaultdict(float))
        self.global_mean = 3.0
        
    def fit(self, ratings_df, items_df=None):
        self.global_mean = ratings_df['rating'].mean()
        
        # Build simple user profiles based on rated items
        for _, row in ratings_df.iterrows():
            self.user_profiles[row['user_id']][row['item_id']] = row['rating']
        
        return self
    
    def predict(self, user_id, item_id):
        if user_id not in self.user_profiles:
            return self.global_mean
        
        profile = self.user_profiles[user_id]
        if not profile:
            return self.global_mean
        
        # Simple prediction based on average of similar items
        return np.mean(list(profile.values()))


class PopularityBased:
    """Popularity-based baseline"""
    
    def __init__(self):
        self.item_popularity = {}
        self.item_avg_rating = {}
        self.global_mean = 3.0
        
    def fit(self, ratings_df):
        self.global_mean = ratings_df['rating'].mean()
        
        item_stats = ratings_df.groupby('item_id').agg({
            'rating': ['count', 'mean']
        }).reset_index()
        item_stats.columns = ['item_id', 'count', 'mean']
        
        for _, row in item_stats.iterrows():
            self.item_popularity[row['item_id']] = row['count']
            self.item_avg_rating[row['item_id']] = row['mean']
        
        return self
    
    def predict(self, user_id, item_id):
        return self.item_avg_rating.get(item_id, self.global_mean)


class HybridRecommender:
    """
    Hybrid Recommendation System
    
    Combines multiple recommendation algorithms using various strategies:
    - weighted: Weighted average of predictions
    - switching: Select algorithm based on context
    - cascade: Use one algorithm to filter, another to rank
    """
    
    def __init__(
        self,
        strategy: str = 'weighted',  # 'weighted', 'switching', 'cascade'
        weights: Dict[str, float] = None,
        cold_start_threshold: int = 5
    ):
        self.strategy = strategy
        self.weights = weights or {
            'collaborative': 0.4,
            'content': 0.3,
            'popularity': 0.3
        }
        self.cold_start_threshold = cold_start_threshold
        
        # Component models
        self.models = {
            'collaborative': SimpleCollaborativeFilter(),
            'content': SimpleContentBased(),
            'popularity': PopularityBased()
        }
        
        # User data
        self.user_ratings = defaultdict(dict)
        self.global_mean = 3.0
        
        # Statistics
        self.stats = {
            'n_users': 0,
            'n_items': 0,
            'n_ratings': 0,
            'cold_start_users': 0,
            'cold_start_items': 0
        }
        
        # Prediction breakdown for explainability
        self.last_prediction_breakdown = {}
    
    def fit(self, ratings_df: pd.DataFrame, items_df: pd.DataFrame = None) -> 'HybridRecommender':
        """Fit all component models"""
        print(f"\n🔧 Fitting Hybrid Recommender ({self.strategy})...")
        print(f"   Weights: {self.weights}")
        
        # Store user ratings
        for _, row in ratings_df.iterrows():
            self.user_ratings[row['user_id']][row['item_id']] = row['rating']
        
        self.global_mean = ratings_df['rating'].mean()
        
        # Fit each component
        print("   Fitting Collaborative Filter...")
        self.models['collaborative'].fit(ratings_df)
        
        print("   Fitting Content-Based...")
        self.models['content'].fit(ratings_df, items_df)
        
        print("   Fitting Popularity Model...")
        self.models['popularity'].fit(ratings_df)
        
        # Statistics
        self.stats['n_users'] = ratings_df['user_id'].nunique()
        self.stats['n_items'] = ratings_df['item_id'].nunique()
        self.stats['n_ratings'] = len(ratings_df)
        
        # Count cold start cases
        user_counts = ratings_df.groupby('user_id').size()
        item_counts = ratings_df.groupby('item_id').size()
        self.stats['cold_start_users'] = (user_counts < self.cold_start_threshold).sum()
        self.stats['cold_start_items'] = (item_counts < self.cold_start_threshold).sum()
        
        print(f"   ✅ Hybrid model fitted")
        print(f"   📊 Cold-start users: {self.stats['cold_start_users']}, items: {self.stats['cold_start_items']}")
        
        return self
    
    def predict(self, user_id: Any, item_id: Any, return_breakdown: bool = False) -> float:
        """Predict rating using hybrid strategy"""
        breakdown = {}
        
        if self.strategy == 'weighted':
            prediction = self._weighted_predict(user_id, item_id, breakdown)
        elif self.strategy == 'switching':
            prediction = self._switching_predict(user_id, item_id, breakdown)
        else:  # cascade
            prediction = self._cascade_predict(user_id, item_id, breakdown)
        
        self.last_prediction_breakdown = breakdown
        
        if return_breakdown:
            return prediction, breakdown
        return prediction
    
    def _weighted_predict(self, user_id: Any, item_id: Any, breakdown: dict) -> float:
        """Weighted average of all models"""
        total = 0
        total_weight = 0
        
        for model_name, model in self.models.items():
            weight = self.weights.get(model_name, 0)
            if weight > 0:
                pred = model.predict(user_id, item_id)
                breakdown[model_name] = {'prediction': pred, 'weight': weight}
                total += weight * pred
                total_weight += weight
        
        prediction = total / total_weight if total_weight > 0 else self.global_mean
        breakdown['final'] = prediction
        breakdown['strategy'] = 'weighted'
        
        return np.clip(prediction, 1, 5)
    
    def _switching_predict(self, user_id: Any, item_id: Any, breakdown: dict) -> float:
        """Switch between models based on context"""
        n_user_ratings = len(self.user_ratings.get(user_id, {}))
        
        breakdown['n_user_ratings'] = n_user_ratings
        breakdown['strategy'] = 'switching'
        
        if n_user_ratings >= self.cold_start_threshold:
            # Use collaborative for warm users
            pred = self.models['collaborative'].predict(user_id, item_id)
            breakdown['selected_model'] = 'collaborative'
            breakdown['reason'] = f'User has {n_user_ratings} ratings (warm)'
        elif n_user_ratings > 0:
            # Use content for users with some history
            pred = self.models['content'].predict(user_id, item_id)
            breakdown['selected_model'] = 'content'
            breakdown['reason'] = f'User has {n_user_ratings} ratings (limited)'
        else:
            # Use popularity for cold start
            pred = self.models['popularity'].predict(user_id, item_id)
            breakdown['selected_model'] = 'popularity'
            breakdown['reason'] = 'Cold start user'
        
        breakdown['prediction'] = pred
        breakdown['final'] = pred
        
        return np.clip(pred, 1, 5)
    
    def _cascade_predict(self, user_id: Any, item_id: Any, breakdown: dict) -> float:
        """Cascade: popularity filters, CF refines"""
        breakdown['strategy'] = 'cascade'
        
        # Stage 1: Popularity baseline
        pop_pred = self.models['popularity'].predict(user_id, item_id)
        breakdown['stage1_popularity'] = pop_pred
        
        # Stage 2: CF refinement
        cf_pred = self.models['collaborative'].predict(user_id, item_id)
        breakdown['stage2_collaborative'] = cf_pred
        
        # Combine: use CF but anchor toward popularity
        final = 0.3 * pop_pred + 0.7 * cf_pred
        breakdown['final'] = final
        
        return np.clip(final, 1, 5)
    
    def recommend(
        self,
        user_id: Any,
        n_recommendations: int = 10,
        exclude_rated: bool = True
    ) -> List[Tuple[Any, float, Dict]]:
        """Generate recommendations with explanations"""
        rated_items = set(self.user_ratings[user_id].keys()) if exclude_rated else set()
        
        # Get all items
        all_items = set(self.models['popularity'].item_avg_rating.keys())
        candidate_items = all_items - rated_items
        
        # Score all candidates
        scores = []
        for item_id in candidate_items:
            pred, breakdown = self.predict(user_id, item_id, return_breakdown=True)
            scores.append((item_id, pred, breakdown.copy()))
        
        # Sort and return top N
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n_recommendations]
    
    def explain_recommendation(self, user_id: Any, item_id: Any) -> Dict:
        """Get detailed explanation for a recommendation"""
        pred, breakdown = self.predict(user_id, item_id, return_breakdown=True)
        
        explanation = {
            'user_id': user_id,
            'item_id': item_id,
            'final_prediction': pred,
            'strategy': self.strategy,
            'breakdown': breakdown,
            'user_history_size': len(self.user_ratings.get(user_id, {})),
            'is_cold_start_user': len(self.user_ratings.get(user_id, {})) < self.cold_start_threshold
        }
        
        return explanation
    
    def evaluate(
        self,
        test_df: pd.DataFrame,
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """Evaluate hybrid model"""
        print("\n📊 Evaluating model...")
        
        y_true, y_pred = [], []
        strategy_counts = defaultdict(int)
        
        for _, row in test_df.iterrows():
            y_true.append(row['rating'])
            pred, breakdown = self.predict(row['user_id'], row['item_id'], return_breakdown=True)
            y_pred.append(pred)
            
            if 'selected_model' in breakdown:
                strategy_counts[breakdown['selected_model']] += 1
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        metrics = {'RMSE': rmse, 'MAE': mae}
        
        # Add strategy usage stats
        total = sum(strategy_counts.values())
        if total > 0:
            for model, count in strategy_counts.items():
                metrics[f'{model}_usage'] = count / total
        
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
                recommended_items = [item for item, _, _ in recommendations]
                
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
            all_recommended.update([item for item, _, _ in recs])
        
        metrics['Coverage'] = len(all_recommended) / self.stats['n_items']
        
        return metrics


def create_interactive_visualization(
    hybrid_model: HybridRecommender,
    sample_users: int = 20,
    output_path: str = "../visualizations/hybrid.html",
    metrics: Dict = None
):
    """Create interactive visualization showing hybrid recommendations with enhanced template"""
    print("\n🎨 Creating enhanced interactive visualization...")
    
    # Try to import enhanced visualization utilities
    try:
        from visualization_utils import generate_enhanced_visualization
        use_enhanced = True
    except ImportError:
        use_enhanced = False
        print("   ⚠️ Enhanced visualization not available, using basic template")
    
    net = Network(
        height="700px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#333333"
    )
    
    net.set_options("""
    {
        "nodes": {"font": {"size": 12, "strokeWidth": 2, "strokeColor": "#ffffff"}, "borderWidth": 2, "shadow": true},
        "edges": {"font": {"size": 9, "align": "middle"}, "smooth": {"type": "continuous"}},
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -80,
                "springLength": 150
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 100}
        },
        "interaction": {"hover": true, "tooltipDelay": 100, "navigationButtons": true, "keyboard": true}
    }
    """)
    
    # Sample users
    user_ids = list(hybrid_model.user_ratings.keys())[:sample_users]
    recommended_items = set()
    all_recommendations = []
    
    # Add user nodes with cold-start indication
    for user_id in user_ids:
        n_ratings = len(hybrid_model.user_ratings[user_id])
        is_cold = n_ratings < hybrid_model.cold_start_threshold
        
        color = "#ff9800" if is_cold else "#4285f4"
        
        net.add_node(
            f"U_{user_id}",
            label=f"U{user_id}",
            color=color,
            size=24 + min(n_ratings * 0.5, 15),
            shape="dot",
            title=f"<div style='padding:8px;'><strong>👤 User {user_id}</strong><br><br>"
                  f"<b>Ratings:</b> {n_ratings}<br>"
                  f"<b>Cold Start:</b> {'Yes' if is_cold else 'No'}<br>"
                  f"<b>Threshold:</b> {hybrid_model.cold_start_threshold}</div>",
            group="user"
        )
        
        # Get recommendations
        recs = hybrid_model.recommend(user_id, n_recommendations=5)
        for item_id, score, breakdown in recs:
            strategy = breakdown.get('strategy', 'unknown')
            recommended_items.add((item_id, score, strategy))
            reason = f"Hybrid {strategy} recommendation (score={score:.2f})"
            all_recommendations.append((item_id, score, reason))
    
    # Add item nodes
    for item_id, score, strategy in recommended_items:
        # Color by dominant strategy
        if strategy == 'weighted':
            color = "#9c27b0"
        elif 'collaborative' in str(strategy):
            color = "#2196f3"
        elif 'content' in str(strategy):
            color = "#4caf50"
        else:
            color = "#ff5722"
        
        net.add_node(
            f"I_{item_id}",
            label=f"I{item_id}",
            color=color,
            size=20,
            shape="dot",
            title=f"<div style='padding:8px;'><strong>📦 Item {item_id}</strong><br><br>"
                  f"<b>Strategy:</b> {strategy}<br>"
                  f"<b>Score:</b> {score:.2f}</div>",
            group="item"
        )
    
    # Add recommendation edges with descriptive labels
    edge_count = 0
    for user_id in user_ids:
        recs = hybrid_model.recommend(user_id, n_recommendations=5)
        for item_id, score, breakdown in recs:
            strategy = breakdown.get('strategy', 'unknown')
            
            if strategy == 'weighted':
                color = "#9c27b0"
            elif breakdown.get('selected_model') == 'collaborative':
                color = "#2196f3"
            elif breakdown.get('selected_model') == 'content':
                color = "#4caf50"
            else:
                color = "#ff5722"
            
            label_text = f"{strategy[:3]} {score:.1f}"
            net.add_edge(
                f"U_{user_id}",
                f"I_{item_id}",
                color=color,
                width=2 + score * 0.3,
                label=label_text,
                title=f"<div style='padding:8px;'><b>🎯 Hybrid Recommendation</b><br><br>"
                      f"<b>User:</b> {user_id}<br>"
                      f"<b>Item:</b> {item_id}<br>"
                      f"<b>Score:</b> {score:.2f}<br>"
                      f"<b>Strategy:</b> {strategy}</div>",
                font={"size": 9, "color": color, "background": "rgba(255,255,255,0.8)"}
            )
            edge_count += 1
    
    # Prepare recommendation data for tables
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
        "Items": len(recommended_items),
        "Edges": edge_count,
        "Strategy": hybrid_model.strategy.capitalize(),
        "Cold Threshold": hybrid_model.cold_start_threshold
    }
    
    # Node and edge types for legend
    node_types = [
        {"color": "#4285f4", "label": "Warm Users", "description": "Users with sufficient rating history"},
        {"color": "#ff9800", "label": "Cold Start Users", "description": "Users with limited ratings"},
        {"color": "#9c27b0", "label": "Weighted Items", "description": "Items from weighted combination"},
        {"color": "#2196f3", "label": "Collaborative Items", "description": "Items from collaborative filtering"},
        {"color": "#4caf50", "label": "Content Items", "description": "Items from content-based filtering"}
    ]
    
    edge_types = [
        {"color": "#9c27b0", "label": "Weighted", "style": "solid", "description": "Weighted combination of models"},
        {"color": "#2196f3", "label": "Collaborative", "style": "solid", "description": "User-User similarity based"},
        {"color": "#4caf50", "label": "Content", "style": "solid", "description": "Content similarity based"},
        {"color": "#ff5722", "label": "Popularity", "style": "solid", "description": "Popular items fallback"}
    ]
    
    # Generate enhanced or basic visualization
    if use_enhanced:
        generate_enhanced_visualization(
            net=net,
            algorithm_key="hybrid",
            output_path=output_path,
            metrics=metrics,
            graph_stats=graph_stats,
            node_types=node_types,
            edge_types=edge_types,
            recommendations_data=recommendations_data
        )
    else:
        # Fallback: basic visualization
        legend_html = """
        <div style="position: absolute; top: 10px; right: 10px; background: white; 
                    padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    font-family: Arial, sans-serif; font-size: 12px; max-width: 220px;">
            <h3 style="margin: 0 0 10px 0; font-size: 14px;">Hybrid Recommender</h3>
            <div style="margin: 5px 0;">
                <span style="display: inline-block; width: 15px; height: 15px; 
                            background: #4285f4; border-radius: 50%; margin-right: 8px;"></span>
                Warm Users
            </div>
            <div style="margin: 5px 0;">
                <span style="display: inline-block; width: 15px; height: 15px; 
                            background: #ff9800; border-radius: 50%; margin-right: 8px;"></span>
                Cold Start Users
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
    hybrid_model: HybridRecommender,
    metrics: Dict[str, float],
    output_dir: str = "../reports/hybrid"
):
    """Generate comprehensive statistical report"""
    print("\n📈 Generating statistical report...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Hybrid Recommender Analysis ({hybrid_model.strategy})', fontsize=14)
    
    # 1. Weight distribution
    ax1 = axes[0, 0]
    weights = hybrid_model.weights
    ax1.bar(weights.keys(), weights.values(), color=['#2196f3', '#4caf50', '#ff5722'])
    ax1.set_ylabel('Weight')
    ax1.set_title('Model Weights')
    ax1.set_ylim(0, 1)
    
    # 2. User ratings distribution
    ax2 = axes[0, 1]
    user_rating_counts = [len(r) for r in hybrid_model.user_ratings.values()]
    ax2.hist(user_rating_counts, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
    ax2.axvline(x=hybrid_model.cold_start_threshold, color='red', linestyle='--', 
                label=f'Cold Start Threshold ({hybrid_model.cold_start_threshold})')
    ax2.set_xlabel('Number of Ratings')
    ax2.set_ylabel('Number of Users')
    ax2.set_title('User Activity Distribution')
    ax2.legend()
    
    # 3. Cold start analysis
    ax3 = axes[0, 2]
    cold_users = hybrid_model.stats['cold_start_users']
    warm_users = hybrid_model.stats['n_users'] - cold_users
    ax3.pie([warm_users, cold_users], labels=['Warm Users', 'Cold Start Users'],
            colors=['#4caf50', '#ff9800'], autopct='%1.1f%%', startangle=90)
    ax3.set_title('User Cold Start Distribution')
    
    # 4. Strategy usage (if switching)
    ax4 = axes[1, 0]
    usage_metrics = {k.replace('_usage', ''): v for k, v in metrics.items() if '_usage' in k}
    if usage_metrics:
        ax4.bar(usage_metrics.keys(), usage_metrics.values(), 
                color=['#2196f3', '#4caf50', '#ff5722'][:len(usage_metrics)])
        ax4.set_ylabel('Usage Proportion')
        ax4.set_title('Model Usage Distribution')
    else:
        ax4.text(0.5, 0.5, f'Strategy: {hybrid_model.strategy}\n(No switching stats)', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Model Selection')
    
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
    
    # 6. Configuration summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    config_text = f"""
    Strategy: {hybrid_model.strategy}
    
    Weights:
    - Collaborative: {hybrid_model.weights.get('collaborative', 0):.2f}
    - Content: {hybrid_model.weights.get('content', 0):.2f}
    - Popularity: {hybrid_model.weights.get('popularity', 0):.2f}
    
    Cold Start Threshold: {hybrid_model.cold_start_threshold}
    
    Dataset:
    - Users: {hybrid_model.stats['n_users']:,}
    - Items: {hybrid_model.stats['n_items']:,}
    - Ratings: {hybrid_model.stats['n_ratings']:,}
    """
    ax6.text(0.1, 0.9, config_text, transform=ax6.transAxes, fontsize=20,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax6.set_title('Configuration')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/analysis_figures.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Text report
    report = f"""
================================================================================
HYBRID RECOMMENDER - STATISTICAL REPORT
================================================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CONFIGURATION
-------------
Strategy: {hybrid_model.strategy}
Weights: {hybrid_model.weights}
Cold Start Threshold: {hybrid_model.cold_start_threshold}

DATASET STATISTICS
------------------
Number of Users: {hybrid_model.stats['n_users']:,}
Number of Items: {hybrid_model.stats['n_items']:,}
Number of Ratings: {hybrid_model.stats['n_ratings']:,}
Cold Start Users: {hybrid_model.stats['cold_start_users']:,} ({100*hybrid_model.stats['cold_start_users']/hybrid_model.stats['n_users']:.1f}%)
Cold Start Items: {hybrid_model.stats['cold_start_items']:,}

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
    print("HYBRID RECOMMENDER SYSTEM")
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
    
    # Test different strategies
    strategies = ['weighted', 'switching', 'cascade']
    
    for strategy in strategies:
        print("\n" + "=" * 70)
        print(f"TRAINING {strategy.upper()} HYBRID MODEL")
        print("=" * 70)
        
        model = HybridRecommender(
            strategy=strategy,
            weights={'collaborative': 0.4, 'content': 0.3, 'popularity': 0.3},
            cold_start_threshold=5
        )
        model.fit(train_df, items_df)
        
        metrics = model.evaluate(test_df)
        print(f"\n📊 {strategy.upper()} Metrics:")
        for metric, value in metrics.items():
            if not metric.endswith('_usage'):
                print(f"   {metric}: {value:.4f}")
        
        create_interactive_visualization(
            model,
            sample_users=25,
            output_path=f"../visualizations/hybrid_{strategy}.html",
            metrics=metrics
        )
        
        generate_statistical_report(
            model, metrics,
            output_dir=f"../reports/hybrid/{strategy}"
        )
        
        # Show explanation example
        print("\n📝 Recommendation Explanation Example:")
        sample_user = list(model.user_ratings.keys())[0]
        recs = model.recommend(sample_user, n_recommendations=1)
        if recs:
            explanation = model.explain_recommendation(sample_user, recs[0][0])
            print(f"   User {sample_user} → Item {recs[0][0]}")
            print(f"   Strategy: {explanation['strategy']}")
            print(f"   Cold Start: {explanation['is_cold_start_user']}")
            print(f"   Predicted: {explanation['final_prediction']:.2f}")
    
    print("\n" + "=" * 70)
    print("✅ HYBRID RECOMMENDER COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
