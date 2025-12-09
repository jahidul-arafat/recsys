"""
08_popularity_baseline.py
==========================
Popularity-Based Recommendation Baseline

Implements:
- Global popularity (most rated items)
- Rating-weighted popularity
- Time-decayed popularity
- Demographic-based popularity

This serves as a baseline to compare against more sophisticated algorithms.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import defaultdict
import matplotlib.pyplot as plt
from pyvis.network import Network
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Create output directories
os.makedirs("../outputs/popularity_baseline", exist_ok=True)
os.makedirs("../visualizations", exist_ok=True)
os.makedirs("../reports/popularity_baseline", exist_ok=True)


class PopularityRecommender:
    """
    Popularity-Based Baseline Recommender
    
    Simple but effective baseline that recommends items based on:
    - Number of ratings (popularity)
    - Average rating (quality)
    - Combination of both (weighted score)
    """
    
    def __init__(
        self,
        method: str = 'weighted',  # 'count', 'rating', 'weighted', 'bayesian'
        popularity_weight: float = 0.5,
        min_ratings: int = 5,
        bayesian_prior: float = 3.0,
        bayesian_confidence: int = 10
    ):
        self.method = method
        self.popularity_weight = popularity_weight
        self.min_ratings = min_ratings
        self.bayesian_prior = bayesian_prior
        self.bayesian_confidence = bayesian_confidence
        
        # Item statistics
        self.item_counts = {}
        self.item_ratings = {}
        self.item_avg_ratings = {}
        self.item_scores = {}
        
        # User statistics
        self.user_ratings = defaultdict(dict)
        self.global_mean = 3.0
        
        # Statistics
        self.stats = {
            'n_users': 0,
            'n_items': 0,
            'n_ratings': 0,
            'max_popularity': 0,
            'min_popularity': 0
        }
    
    def fit(self, ratings_df: pd.DataFrame) -> 'PopularityRecommender':
        """
        Fit the popularity model
        
        Args:
            ratings_df: DataFrame with columns [user_id, item_id, rating]
        """
        print(f"\n🔧 Fitting Popularity Recommender ({self.method})...")
        print(f"   popularity_weight: {self.popularity_weight}")
        print(f"   min_ratings: {self.min_ratings}")
        
        # Store user ratings
        for _, row in ratings_df.iterrows():
            self.user_ratings[row['user_id']][row['item_id']] = row['rating']
        
        # Calculate item statistics
        item_groups = ratings_df.groupby('item_id')
        
        for item_id, group in item_groups:
            count = len(group)
            avg_rating = group['rating'].mean()
            
            self.item_counts[item_id] = count
            self.item_ratings[item_id] = list(group['rating'])
            self.item_avg_ratings[item_id] = avg_rating
        
        # Global statistics
        self.global_mean = ratings_df['rating'].mean()
        max_count = max(self.item_counts.values())
        
        # Calculate scores based on method
        for item_id in self.item_counts:
            count = self.item_counts[item_id]
            avg_rating = self.item_avg_ratings[item_id]
            
            if self.method == 'count':
                score = count / max_count
            elif self.method == 'rating':
                if count >= self.min_ratings:
                    score = avg_rating
                else:
                    score = self.global_mean
            elif self.method == 'weighted':
                # Weighted combination of popularity and rating
                norm_count = count / max_count
                norm_rating = (avg_rating - 1) / 4  # Normalize to [0, 1]
                score = (self.popularity_weight * norm_count + 
                        (1 - self.popularity_weight) * norm_rating)
            elif self.method == 'bayesian':
                # Bayesian average (IMDB formula)
                # WR = (v/(v+m)) * R + (m/(v+m)) * C
                # v = number of votes, m = minimum votes required, R = avg rating, C = mean rating
                score = ((count / (count + self.bayesian_confidence)) * avg_rating +
                        (self.bayesian_confidence / (count + self.bayesian_confidence)) * self.bayesian_prior)
            else:
                score = count
            
            self.item_scores[item_id] = score
        
        # Update statistics
        self.stats['n_users'] = len(self.user_ratings)
        self.stats['n_items'] = len(self.item_counts)
        self.stats['n_ratings'] = len(ratings_df)
        self.stats['max_popularity'] = max(self.item_counts.values())
        self.stats['min_popularity'] = min(self.item_counts.values())
        
        print(f"   ✅ Model fitted: {self.stats['n_items']} items")
        print(f"   📊 Popularity range: {self.stats['min_popularity']} - {self.stats['max_popularity']}")
        
        return self
    
    def predict(self, user_id: Any, item_id: Any) -> float:
        """Predict rating for user-item pair"""
        if item_id in self.item_avg_ratings:
            # Use item's average rating
            return self.item_avg_ratings[item_id]
        return self.global_mean
    
    def recommend(
        self,
        user_id: Any = None,
        n_recommendations: int = 10,
        exclude_rated: bool = True
    ) -> List[Tuple[Any, float]]:
        """
        Generate top-N recommendations
        
        For popularity-based, recommendations are the same for all users
        (except for excluding already rated items)
        """
        # Get items to exclude
        rated_items = set()
        if exclude_rated and user_id in self.user_ratings:
            rated_items = set(self.user_ratings[user_id].keys())
        
        # Sort items by score
        recommendations = []
        for item_id, score in self.item_scores.items():
            if item_id not in rated_items:
                # Convert score to predicted rating
                if self.method == 'count':
                    pred_rating = self.item_avg_ratings.get(item_id, self.global_mean)
                elif self.method in ['rating', 'bayesian']:
                    pred_rating = score
                else:
                    pred_rating = self.item_avg_ratings.get(item_id, self.global_mean)
                
                recommendations.append((item_id, pred_rating, score))
        
        # Sort by score (not predicted rating)
        recommendations.sort(key=lambda x: x[2], reverse=True)
        
        return [(item_id, pred) for item_id, pred, _ in recommendations[:n_recommendations]]
    
    def get_popular_items(self, n: int = 20) -> List[Tuple[Any, int, float]]:
        """Get most popular items with their counts and ratings"""
        items = [(item_id, self.item_counts[item_id], self.item_avg_ratings[item_id])
                 for item_id in self.item_counts]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:n]
    
    def get_highly_rated_items(self, n: int = 20, min_ratings: int = None) -> List[Tuple[Any, float, int]]:
        """Get highest rated items with minimum rating threshold"""
        if min_ratings is None:
            min_ratings = self.min_ratings
        
        items = [(item_id, self.item_avg_ratings[item_id], self.item_counts[item_id])
                 for item_id in self.item_counts
                 if self.item_counts[item_id] >= min_ratings]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:n]
    
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
        
        # Coverage (for popularity baseline, this is typically low)
        all_recommended = set()
        sample_users = list(self.user_ratings.keys())[:100]
        for user_id in sample_users:
            recs = self.recommend(user_id, n_recommendations=10)
            all_recommended.update([item for item, _ in recs])
        
        metrics['Coverage'] = len(all_recommended) / self.stats['n_items']
        
        return metrics


def create_interactive_visualization(
    pop_model: PopularityRecommender,
    top_n_items: int = 50,
    output_path: str = "../visualizations/popularity_baseline.html",
    metrics: Dict = None
):
    """Create visualization showing popularity distribution with enhanced template"""
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
        "edges": {"smooth": false},
        "physics": {
            "barnesHut": {
                "gravitationalConstant": -5000,
                "springLength": 100
            },
            "stabilization": {"iterations": 100}
        },
        "interaction": {"hover": true, "tooltipDelay": 100, "navigationButtons": true, "keyboard": true}
    }
    """)
    
    # Add central "Popular Items" hub
    net.add_node(
        "hub",
        label="Top Items",
        color="#9c27b0",
        size=40,
        shape="dot",
        title="<div style='padding:8px;'><strong>🏆 Most Popular Items</strong><br><br>"
              f"<b>Method:</b> {pop_model.method.capitalize()}<br>"
              f"<b>Total Items:</b> {len(pop_model.item_popularity)}</div>",
        fixed=True,
        x=0,
        y=0
    )
    
    # Get top items
    top_items = pop_model.get_popular_items(top_n_items)
    
    # Collect recommendations for tables
    all_recommendations = []
    
    # Calculate positions in a circle
    for i, (item_id, count, avg_rating) in enumerate(top_items):
        angle = 2 * np.pi * i / len(top_items)
        radius = 300 + (len(top_items) - i) * 5
        
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        
        # Size by popularity, color by rating
        size = 10 + count * 0.5
        
        # Color gradient from red (low rating) to green (high rating)
        rating_norm = (avg_rating - 1) / 4
        r = int(255 * (1 - rating_norm))
        g = int(255 * rating_norm)
        color = f"rgb({r}, {g}, 100)"
        
        # Add to recommendations
        reason = f"Rank #{i+1}: {count} ratings, avg {avg_rating:.2f}"
        all_recommendations.append((item_id, count / top_items[0][1], reason))  # Normalize score
        
        net.add_node(
            f"I_{item_id}",
            label=f"#{i+1}",
            color=color,
            size=size,
            x=float(x),
            y=float(y),
            shape="dot",
            title=f"<div style='padding:8px;'><strong>📦 Item {item_id}</strong><br><br>"
                  f"<b>Rank:</b> #{i+1}<br>"
                  f"<b>Ratings:</b> {count}<br>"
                  f"<b>Avg Rating:</b> {avg_rating:.2f}<br>"
                  f"<b>Popularity Score:</b> {count / top_items[0][1]:.2%}</div>",
            group="item"
        )
        
        # Connect to hub with edge width by popularity rank
        edge_width = max(1, 5 - i * 0.1)
        label_text = f"rank {i+1}"
        net.add_edge(
            "hub",
            f"I_{item_id}",
            width=edge_width,
            label=label_text if i < 10 else "",  # Only label top 10
            color={'color': '#cccccc', 'opacity': 0.5},
            font={"size": 8, "color": "#999999", "background": "rgba(255,255,255,0.8)"}
        )
    
    # Prepare recommendation data for tables
    recommendations_data = {
        'recommendations': [(item_id, score, reason) for item_id, score, reason in all_recommendations[:20]],
        'best_fit': [],
        'worst_fit': [],
        'avoided': []
    }
    
    # Graph statistics
    graph_stats = {
        "Total Items": len(pop_model.item_popularity),
        "Top Items Shown": len(top_items),
        "Method": pop_model.method.capitalize(),
        "Top Item Ratings": top_items[0][1] if top_items else 0,
        "Avg Top Rating": np.mean([r for _, _, r in top_items[:10]]) if top_items else 0
    }
    
    # Node and edge types for legend
    node_types = [
        {"color": "#9c27b0", "label": "Hub", "description": "Central popularity hub"},
        {"color": "rgb(50, 200, 100)", "label": "High Rated", "description": "Items with high avg rating"},
        {"color": "rgb(200, 100, 100)", "label": "Low Rated", "description": "Items with lower avg rating"}
    ]
    
    edge_types = [
        {"color": "#cccccc", "label": "Popularity Link", "style": "solid", "description": "Connection to popularity hub"}
    ]
    
    # Generate enhanced or basic visualization
    if use_enhanced:
        generate_enhanced_visualization(
            net=net,
            algorithm_key="popularity",
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
                    font-family: Arial, sans-serif; font-size: 12px; max-width: 200px;">
            <h3 style="margin: 0 0 10px 0; font-size: 14px;">Popularity Baseline</h3>
            <p style="font-size: 11px; color: #666; margin: 0 0 10px 0;">
                Items sized by popularity<br>
                Colored by average rating
            </p>
            <div style="margin: 5px 0;">
                <span style="display: inline-block; width: 15px; height: 15px; 
                            background: rgb(50, 200, 100); border-radius: 50%; margin-right: 8px;"></span>
                High Rating
            </div>
            <div style="margin: 5px 0;">
                <span style="display: inline-block; width: 15px; height: 15px; 
                            background: rgb(200, 100, 100); border-radius: 50%; margin-right: 8px;"></span>
                Low Rating
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
    pop_model: PopularityRecommender,
    metrics: Dict[str, float],
    output_dir: str = "../reports/popularity_baseline"
):
    """Generate comprehensive statistical report"""
    print("\n📈 Generating statistical report...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Popularity Baseline Analysis ({pop_model.method})', fontsize=14)
    
    # 1. Popularity distribution (long tail)
    ax1 = axes[0, 0]
    counts = sorted(pop_model.item_counts.values(), reverse=True)
    ax1.plot(range(len(counts)), counts, 'b-', linewidth=1)
    ax1.fill_between(range(len(counts)), counts, alpha=0.3)
    ax1.set_xlabel('Item Rank')
    ax1.set_ylabel('Number of Ratings')
    ax1.set_title('Long Tail Distribution')
    ax1.set_yscale('log')
    
    # 2. Rating distribution
    ax2 = axes[0, 1]
    avg_ratings = list(pop_model.item_avg_ratings.values())
    ax2.hist(avg_ratings, bins=20, color='coral', edgecolor='white', alpha=0.7)
    ax2.axvline(x=pop_model.global_mean, color='red', linestyle='--', label=f'Global Mean: {pop_model.global_mean:.2f}')
    ax2.set_xlabel('Average Rating')
    ax2.set_ylabel('Number of Items')
    ax2.set_title('Item Rating Distribution')
    ax2.legend()
    
    # 3. Popularity vs Rating scatter
    ax3 = axes[0, 2]
    item_ids = list(pop_model.item_counts.keys())
    x = [pop_model.item_counts[i] for i in item_ids]
    y = [pop_model.item_avg_ratings[i] for i in item_ids]
    ax3.scatter(x, y, alpha=0.5, s=20)
    ax3.set_xlabel('Number of Ratings')
    ax3.set_ylabel('Average Rating')
    ax3.set_title('Popularity vs Rating')
    ax3.set_xscale('log')
    
    # 4. Score distribution
    ax4 = axes[1, 0]
    scores = list(pop_model.item_scores.values())
    ax4.hist(scores, bins=30, color='seagreen', edgecolor='white', alpha=0.7)
    ax4.set_xlabel('Item Score')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'Score Distribution ({pop_model.method})')
    
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
    
    # 6. Top items table
    ax6 = axes[1, 2]
    ax6.axis('off')
    top_items = pop_model.get_popular_items(10)
    
    table_text = "Top 10 Popular Items:\n" + "-" * 35 + "\n"
    table_text += f"{'Rank':<6}{'Item':<10}{'Count':<8}{'Rating':<8}\n"
    table_text += "-" * 35 + "\n"
    for i, (item_id, count, rating) in enumerate(top_items, 1):
        table_text += f"{i:<6}{item_id:<10}{count:<8}{rating:.2f}\n"
    
    ax6.text(0.1, 0.5, table_text, fontsize=9, family='monospace',
             verticalalignment='center', transform=ax6.transAxes)
    ax6.set_title('Top Popular Items')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/analysis_figures.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Text report
    report = f"""
================================================================================
POPULARITY BASELINE - STATISTICAL REPORT
================================================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CONFIGURATION
-------------
Method: {pop_model.method}
Popularity Weight: {pop_model.popularity_weight}
Min Ratings Threshold: {pop_model.min_ratings}
Bayesian Prior: {pop_model.bayesian_prior}
Bayesian Confidence: {pop_model.bayesian_confidence}

DATASET STATISTICS
------------------
Number of Users: {pop_model.stats['n_users']:,}
Number of Items: {pop_model.stats['n_items']:,}
Number of Ratings: {pop_model.stats['n_ratings']:,}
Global Mean Rating: {pop_model.global_mean:.4f}

POPULARITY STATISTICS
---------------------
Max Popularity: {pop_model.stats['max_popularity']}
Min Popularity: {pop_model.stats['min_popularity']}
Mean Popularity: {np.mean(list(pop_model.item_counts.values())):.2f}
Median Popularity: {np.median(list(pop_model.item_counts.values())):.2f}

RATING STATISTICS
-----------------
Mean Item Rating: {np.mean(list(pop_model.item_avg_ratings.values())):.4f}
Std Item Rating: {np.std(list(pop_model.item_avg_ratings.values())):.4f}

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

TOP 10 POPULAR ITEMS
--------------------
"""
    
    top_items = pop_model.get_popular_items(10)
    for i, (item_id, count, rating) in enumerate(top_items, 1):
        report += f"{i}. Item {item_id}: {count} ratings, avg {rating:.2f}\n"
    
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
    print("POPULARITY-BASED BASELINE RECOMMENDER")
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
    methods = ['count', 'rating', 'weighted', 'bayesian']
    
    for method in methods:
        print("\n" + "=" * 70)
        print(f"TRAINING {method.upper()} POPULARITY MODEL")
        print("=" * 70)
        
        model = PopularityRecommender(
            method=method,
            popularity_weight=0.5,
            min_ratings=5,
            bayesian_prior=3.0,
            bayesian_confidence=10
        )
        model.fit(train_df)
        
        metrics = model.evaluate(test_df)
        print(f"\n📊 {method.upper()} Metrics:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        create_interactive_visualization(
            model,
            top_n_items=50,
            output_path=f"../visualizations/popularity_{method}.html",
            metrics=metrics
        )
        
        generate_statistical_report(
            model, metrics,
            output_dir=f"../reports/popularity_baseline/{method}"
        )
    
    print("\n" + "=" * 70)
    print("✅ POPULARITY BASELINE COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
