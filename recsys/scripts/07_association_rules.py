"""
07_association_rules.py
========================
Association Rules Recommendation System

Implements:
- Apriori Algorithm
- FP-Growth (simplified)
- Market Basket Analysis for recommendations

Features:
- Interactive visualization of item associations
- Rule-based explainability
- Comprehensive statistical reports
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from pyvis.network import Network
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Any, Set, FrozenSet
import warnings
warnings.filterwarnings('ignore')

# Create output directories
os.makedirs("../outputs/association_rules", exist_ok=True)
os.makedirs("../visualizations", exist_ok=True)
os.makedirs("../reports/association_rules", exist_ok=True)


class AssociationRulesRecommender:
    """
    Association Rules Recommendation System
    
    Uses market basket analysis to find item associations and
    recommend items based on what similar users have liked.
    """
    
    def __init__(
        self,
        min_support: float = 0.01,
        min_confidence: float = 0.1,
        min_lift: float = 1.0,
        max_itemset_size: int = 3,
        rating_threshold: float = 3.5
    ):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.max_itemset_size = max_itemset_size
        self.rating_threshold = rating_threshold
        
        # Data
        self.transactions = []  # List of sets (user baskets)
        self.user_baskets = {}  # user_id -> set of liked items
        self.item_support = {}  # item -> support
        self.frequent_itemsets = {}  # frozenset -> support
        self.rules = []  # List of (antecedent, consequent, confidence, lift, support)
        
        # For rating prediction
        self.item_avg_rating = {}
        self.global_mean = 3.0
        
        # Statistics
        self.stats = {
            'n_users': 0,
            'n_items': 0,
            'n_transactions': 0,
            'n_frequent_itemsets': 0,
            'n_rules': 0,
            'avg_basket_size': 0
        }
    
    def fit(self, ratings_df: pd.DataFrame) -> 'AssociationRulesRecommender':
        """
        Fit association rules model
        
        Args:
            ratings_df: DataFrame with columns [user_id, item_id, rating]
        """
        print(f"\n🔧 Fitting Association Rules Recommender...")
        print(f"   min_support: {self.min_support}")
        print(f"   min_confidence: {self.min_confidence}")
        print(f"   min_lift: {self.min_lift}")
        print(f"   rating_threshold: {self.rating_threshold}")
        
        # Calculate item average ratings
        item_stats = ratings_df.groupby('item_id')['rating'].agg(['mean', 'count'])
        self.item_avg_rating = item_stats['mean'].to_dict()
        self.global_mean = ratings_df['rating'].mean()
        
        # Create user baskets (items with rating >= threshold)
        liked_items = ratings_df[ratings_df['rating'] >= self.rating_threshold]
        
        for user_id in liked_items['user_id'].unique():
            user_items = liked_items[liked_items['user_id'] == user_id]['item_id'].tolist()
            if user_items:
                self.user_baskets[user_id] = set(user_items)
                self.transactions.append(frozenset(user_items))
        
        # Statistics
        self.stats['n_users'] = len(self.user_baskets)
        self.stats['n_items'] = ratings_df['item_id'].nunique()
        self.stats['n_transactions'] = len(self.transactions)
        self.stats['avg_basket_size'] = np.mean([len(t) for t in self.transactions])
        
        print(f"   Transactions: {self.stats['n_transactions']}")
        print(f"   Avg basket size: {self.stats['avg_basket_size']:.2f}")
        
        # Find frequent itemsets using Apriori
        print("   Mining frequent itemsets...")
        self._find_frequent_itemsets()
        
        # Generate association rules
        print("   Generating association rules...")
        self._generate_rules()
        
        print(f"   ✅ Found {self.stats['n_frequent_itemsets']} frequent itemsets")
        print(f"   ✅ Generated {self.stats['n_rules']} rules")
        
        return self
    
    def _find_frequent_itemsets(self):
        """Find frequent itemsets using Apriori algorithm"""
        n_transactions = len(self.transactions)
        
        # Count single items
        item_counts = defaultdict(int)
        for transaction in self.transactions:
            for item in transaction:
                item_counts[item] += 1
        
        # Filter by minimum support
        self.item_support = {
            item: count / n_transactions
            for item, count in item_counts.items()
            if count / n_transactions >= self.min_support
        }
        
        # Initialize frequent itemsets with singletons
        self.frequent_itemsets = {
            frozenset([item]): support
            for item, support in self.item_support.items()
        }
        
        # Generate larger itemsets
        current_itemsets = list(self.frequent_itemsets.keys())
        
        for k in range(2, self.max_itemset_size + 1):
            # Generate candidates
            candidates = self._generate_candidates(current_itemsets, k)
            
            if not candidates:
                break
            
            # Count support for candidates
            candidate_counts = defaultdict(int)
            for transaction in self.transactions:
                for candidate in candidates:
                    if candidate.issubset(transaction):
                        candidate_counts[candidate] += 1
            
            # Filter by minimum support
            new_frequent = {
                itemset: count / n_transactions
                for itemset, count in candidate_counts.items()
                if count / n_transactions >= self.min_support
            }
            
            if not new_frequent:
                break
            
            self.frequent_itemsets.update(new_frequent)
            current_itemsets = list(new_frequent.keys())
        
        self.stats['n_frequent_itemsets'] = len(self.frequent_itemsets)
    
    def _generate_candidates(self, itemsets: List[FrozenSet], k: int) -> List[FrozenSet]:
        """Generate candidate itemsets of size k"""
        candidates = set()
        itemsets_list = list(itemsets)
        
        for i, itemset1 in enumerate(itemsets_list):
            for itemset2 in itemsets_list[i+1:]:
                union = itemset1 | itemset2
                if len(union) == k:
                    # Check if all k-1 subsets are frequent (Apriori pruning)
                    is_valid = True
                    for item in union:
                        subset = union - {item}
                        if subset not in self.frequent_itemsets:
                            is_valid = False
                            break
                    
                    if is_valid:
                        candidates.add(union)
        
        return list(candidates)
    
    def _generate_rules(self):
        """Generate association rules from frequent itemsets"""
        n_transactions = len(self.transactions)
        
        for itemset, support in self.frequent_itemsets.items():
            if len(itemset) < 2:
                continue
            
            # Generate all possible rules from this itemset
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    
                    if antecedent in self.frequent_itemsets:
                        antecedent_support = self.frequent_itemsets[antecedent]
                        confidence = support / antecedent_support
                        
                        if confidence >= self.min_confidence:
                            # Calculate lift
                            consequent_support = self.frequent_itemsets.get(consequent, 
                                sum(1 for t in self.transactions if consequent.issubset(t)) / n_transactions)
                            
                            lift = confidence / consequent_support if consequent_support > 0 else 0
                            
                            if lift >= self.min_lift:
                                self.rules.append({
                                    'antecedent': antecedent,
                                    'consequent': consequent,
                                    'confidence': confidence,
                                    'lift': lift,
                                    'support': support
                                })
        
        # Sort rules by lift
        self.rules.sort(key=lambda x: x['lift'], reverse=True)
        self.stats['n_rules'] = len(self.rules)
    
    def predict(self, user_id: Any, item_id: Any) -> float:
        """Predict rating based on association rules"""
        user_basket = self.user_baskets.get(user_id, set())
        
        if not user_basket:
            return self.item_avg_rating.get(item_id, self.global_mean)
        
        # Find rules where antecedent is subset of user's basket
        # and consequent contains the target item
        total_confidence = 0
        rule_count = 0
        
        for rule in self.rules:
            if rule['antecedent'].issubset(user_basket) and item_id in rule['consequent']:
                total_confidence += rule['confidence'] * rule['lift']
                rule_count += 1
        
        if rule_count > 0:
            # Convert rule strength to rating
            avg_strength = total_confidence / rule_count
            # Scale to rating range (high confidence/lift -> high rating)
            base_rating = self.item_avg_rating.get(item_id, self.global_mean)
            predicted = base_rating + min(avg_strength, 1.0) * (5 - base_rating) * 0.5
            return np.clip(predicted, 1, 5)
        
        return self.item_avg_rating.get(item_id, self.global_mean)
    
    def recommend(
        self,
        user_id: Any,
        n_recommendations: int = 10,
        exclude_rated: bool = True
    ) -> List[Tuple[Any, float, List[Dict]]]:
        """Generate recommendations based on association rules"""
        user_basket = self.user_baskets.get(user_id, set())
        
        if not user_basket:
            # Cold start: return popular items
            return self._get_popular_items(n_recommendations)
        
        # Find all applicable rules
        item_scores = defaultdict(lambda: {'score': 0, 'rules': []})
        
        for rule in self.rules:
            if rule['antecedent'].issubset(user_basket):
                for item in rule['consequent']:
                    if not exclude_rated or item not in user_basket:
                        score = rule['confidence'] * rule['lift']
                        item_scores[item]['score'] += score
                        item_scores[item]['rules'].append(rule)
        
        # Sort by score
        recommendations = []
        for item_id, data in item_scores.items():
            base_rating = self.item_avg_rating.get(item_id, self.global_mean)
            predicted = base_rating + min(data['score'], 2) * 0.5
            predicted = np.clip(predicted, 1, 5)
            recommendations.append((item_id, predicted, data['rules'][:3]))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:n_recommendations]
    
    def _get_popular_items(self, n: int) -> List[Tuple[Any, float, List]]:
        """Return popular items for cold start"""
        # Sort by support
        popular = sorted(self.item_support.items(), key=lambda x: x[1], reverse=True)[:n]
        return [(item, self.item_avg_rating.get(item, self.global_mean), []) for item, _ in popular]
    
    def get_rules_for_item(self, item_id: Any) -> List[Dict]:
        """Get all rules where item appears in consequent"""
        return [rule for rule in self.rules if item_id in rule['consequent']]
    
    def explain_recommendation(self, user_id: Any, item_id: Any) -> Dict:
        """Explain why an item was recommended"""
        user_basket = self.user_baskets.get(user_id, set())
        
        applicable_rules = []
        for rule in self.rules:
            if rule['antecedent'].issubset(user_basket) and item_id in rule['consequent']:
                applicable_rules.append({
                    'antecedent': list(rule['antecedent']),
                    'confidence': rule['confidence'],
                    'lift': rule['lift'],
                    'support': rule['support']
                })
        
        return {
            'user_id': user_id,
            'item_id': item_id,
            'user_basket_size': len(user_basket),
            'predicted_rating': self.predict(user_id, item_id),
            'applicable_rules': applicable_rules[:5],
            'explanation': f"Based on {len(applicable_rules)} association rules"
        }
    
    def evaluate(
        self,
        test_df: pd.DataFrame,
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """Evaluate the model"""
        print("\n📊 Evaluating model...")
        
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
        sample_users = list(self.user_baskets.keys())[:100]
        for user_id in sample_users:
            recs = self.recommend(user_id, n_recommendations=10)
            all_recommended.update([item for item, _, _ in recs])
        
        metrics['Coverage'] = len(all_recommended) / self.stats['n_items'] if self.stats['n_items'] > 0 else 0
        
        return metrics


def create_interactive_visualization(
    ar_model: AssociationRulesRecommender,
    top_rules: int = 50,
    output_path: str = "../visualizations/association_rules.html",
    metrics: Dict = None
):
    """Create interactive visualization of association rules with enhanced template"""
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
        font_color="#333333",
        directed=True
    )
    
    net.set_options("""
    {
        "nodes": {"font": {"size": 12, "strokeWidth": 2, "strokeColor": "#ffffff"}, "borderWidth": 2, "shadow": true},
        "edges": {
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
            "font": {"size": 9, "align": "middle"},
            "smooth": {"type": "curvedCW"}
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -80,
                "springLength": 200
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 100}
        },
        "interaction": {"hover": true, "tooltipDelay": 100, "navigationButtons": true, "keyboard": true}
    }
    """)
    
    # Get top rules
    rules = ar_model.rules[:top_rules]
    
    # Collect all items
    all_items = set()
    for rule in rules:
        all_items.update(rule['antecedent'])
        all_items.update(rule['consequent'])
    
    # Add item nodes with enhanced tooltips
    for item_id in all_items:
        support = ar_model.item_support.get(item_id, 0)
        avg_rating = ar_model.item_avg_rating.get(item_id, 3.0)
        
        net.add_node(
            f"I_{item_id}",
            label=f"I{item_id}",
            color="#4285f4",
            size=22 + min(support * 150, 20),
            shape="dot",
            title=f"<div style='padding:8px;'><strong>📦 Item {item_id}</strong><br><br>"
                  f"<b>Support:</b> {support:.3f}<br>"
                  f"<b>Avg Rating:</b> {avg_rating:.2f}<br>"
                  f"<b>In Rules:</b> {sum(1 for r in rules if item_id in r['antecedent'] or item_id in r['consequent'])}</div>"
        )
    
    # Collect recommendations from rules
    all_recommendations = []
    
    # Add rule edges with descriptive labels
    for rule in rules:
        for ant_item in rule['antecedent']:
            for con_item in rule['consequent']:
                # Add to recommendations
                reason = f"Association rule (lift={rule['lift']:.2f}, conf={rule['confidence']:.2f})"
                all_recommendations.append((con_item, rule['confidence'], reason))
                
                # Color by lift
                if rule['lift'] > 2:
                    color = "#4caf50"
                    lift_label = "high"
                elif rule['lift'] > 1.5:
                    color = "#ff9800"
                    lift_label = "med"
                else:
                    color = "#f44336"
                    lift_label = "low"
                
                label_text = f"lift {rule['lift']:.1f}"
                net.add_edge(
                    f"I_{ant_item}",
                    f"I_{con_item}",
                    color=color,
                    width=rule['confidence'] * 3,
                    label=label_text,
                    title=f"<div style='padding:8px;'><b>🔗 Association Rule</b><br><br>"
                          f"<b>Antecedent:</b> Item {ant_item}<br>"
                          f"<b>Consequent:</b> Item {con_item}<br>"
                          f"<b>Confidence:</b> {rule['confidence']:.3f}<br>"
                          f"<b>Lift:</b> {rule['lift']:.3f}<br>"
                          f"<b>Support:</b> {rule['support']:.3f}</div>",
                    font={"size": 9, "color": color, "background": "rgba(255,255,255,0.8)"}
                )
    
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
        "Items": len(all_items),
        "Rules": len(rules),
        "Min Support": ar_model.min_support,
        "Min Confidence": ar_model.min_confidence,
        "Algorithm": ar_model.algorithm.upper()
    }
    
    # Node and edge types for legend
    node_types = [
        {"color": "#4285f4", "label": "Items", "description": "Product items (size = support)"}
    ]
    
    edge_types = [
        {"color": "#4caf50", "label": "High Lift (>2)", "style": "solid", "description": "Strong positive association"},
        {"color": "#ff9800", "label": "Medium Lift (1.5-2)", "style": "solid", "description": "Moderate association"},
        {"color": "#f44336", "label": "Low Lift (<1.5)", "style": "solid", "description": "Weak association"}
    ]
    
    # Generate enhanced or basic visualization
    if use_enhanced:
        generate_enhanced_visualization(
            net=net,
            algorithm_key="association_rules",
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
            <h3 style="margin: 0 0 10px 0; font-size: 14px;">Association Rules</h3>
            <p style="font-size: 11px; color: #666;">Edge color = Lift strength</p>
            <div style="margin: 5px 0;">
                <span style="display: inline-block; width: 30px; height: 3px; 
                            background: #4caf50; margin-right: 8px;"></span>
                High Lift (>2)
            </div>
            <div style="margin: 5px 0;">
                <span style="display: inline-block; width: 30px; height: 3px; 
                            background: #ff9800; margin-right: 8px;"></span>
                Medium Lift (1.5-2)
            </div>
            <div style="margin: 5px 0;">
                <span style="display: inline-block; width: 30px; height: 3px; 
                            background: #f44336; margin-right: 8px;"></span>
                Low Lift (<1.5)
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
    ar_model: AssociationRulesRecommender,
    metrics: Dict[str, float],
    output_dir: str = "../reports/association_rules"
):
    """Generate comprehensive statistical report"""
    print("\n📈 Generating statistical report...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Association Rules Analysis', fontsize=14)
    
    # 1. Support distribution
    ax1 = axes[0, 0]
    supports = [r['support'] for r in ar_model.rules]
    if supports:
        ax1.hist(supports, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
    ax1.set_xlabel('Support')
    ax1.set_ylabel('Number of Rules')
    ax1.set_title('Rule Support Distribution')
    
    # 2. Confidence distribution
    ax2 = axes[0, 1]
    confidences = [r['confidence'] for r in ar_model.rules]
    if confidences:
        ax2.hist(confidences, bins=30, color='coral', edgecolor='white', alpha=0.7)
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Number of Rules')
    ax2.set_title('Rule Confidence Distribution')
    
    # 3. Lift distribution
    ax3 = axes[0, 2]
    lifts = [r['lift'] for r in ar_model.rules]
    if lifts:
        ax3.hist(lifts, bins=30, color='seagreen', edgecolor='white', alpha=0.7)
        ax3.axvline(x=1, color='red', linestyle='--', label='Lift = 1')
        ax3.legend()
    ax3.set_xlabel('Lift')
    ax3.set_ylabel('Number of Rules')
    ax3.set_title('Rule Lift Distribution')
    
    # 4. Basket size distribution
    ax4 = axes[1, 0]
    basket_sizes = [len(b) for b in ar_model.user_baskets.values()]
    if basket_sizes:
        ax4.hist(basket_sizes, bins=30, color='purple', edgecolor='white', alpha=0.7)
    ax4.set_xlabel('Basket Size')
    ax4.set_ylabel('Number of Users')
    ax4.set_title('User Basket Size Distribution')
    
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
    
    # 6. Top rules table
    ax6 = axes[1, 2]
    ax6.axis('off')
    if ar_model.rules:
        top_rules_text = "Top 5 Rules by Lift:\n\n"
        for i, rule in enumerate(ar_model.rules[:5]):
            ant = list(rule['antecedent'])[:2]
            con = list(rule['consequent'])[:2]
            top_rules_text += f"{i+1}. {ant} → {con}\n"
            top_rules_text += f"   Conf: {rule['confidence']:.2f}, Lift: {rule['lift']:.2f}\n"
        ax6.text(0.1, 0.9, top_rules_text, transform=ax6.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace')
    ax6.set_title('Top Rules')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/analysis_figures.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Text report
    report = f"""
================================================================================
ASSOCIATION RULES - STATISTICAL REPORT
================================================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CONFIGURATION
-------------
Min Support: {ar_model.min_support}
Min Confidence: {ar_model.min_confidence}
Min Lift: {ar_model.min_lift}
Max Itemset Size: {ar_model.max_itemset_size}
Rating Threshold: {ar_model.rating_threshold}

MINING STATISTICS
-----------------
Number of Transactions: {ar_model.stats['n_transactions']:,}
Number of Items: {ar_model.stats['n_items']:,}
Average Basket Size: {ar_model.stats['avg_basket_size']:.2f}
Frequent Itemsets Found: {ar_model.stats['n_frequent_itemsets']:,}
Association Rules Generated: {ar_model.stats['n_rules']:,}

RULE STATISTICS
---------------
Avg Confidence: {np.mean([r['confidence'] for r in ar_model.rules]):.4f if ar_model.rules else 0}
Avg Lift: {np.mean([r['lift'] for r in ar_model.rules]):.4f if ar_model.rules else 0}
Avg Support: {np.mean([r['support'] for r in ar_model.rules]):.4f if ar_model.rules else 0}

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
    print("ASSOCIATION RULES RECOMMENDER")
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
    
    print("\n" + "=" * 70)
    print("TRAINING ASSOCIATION RULES MODEL")
    print("=" * 70)
    
    model = AssociationRulesRecommender(
        min_support=0.01,
        min_confidence=0.1,
        min_lift=1.0,
        max_itemset_size=3,
        rating_threshold=3.5
    )
    model.fit(train_df)
    
    metrics = model.evaluate(test_df)
    print("\n📊 Association Rules Metrics:")
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    create_interactive_visualization(
        model,
        top_rules=50,
        output_path="../visualizations/association_rules.html",
        metrics=metrics
    )
    
    generate_statistical_report(model, metrics)
    
    # Show explanation example
    print("\n📝 Recommendation Explanation Example:")
    sample_user = list(model.user_baskets.keys())[0]
    recs = model.recommend(sample_user, n_recommendations=3)
    print(f"   User {sample_user} (basket size: {len(model.user_baskets[sample_user])})")
    for item_id, score, rules in recs[:3]:
        print(f"   → Item {item_id}: {score:.2f} ({len(rules)} rules)")
    
    print("\n" + "=" * 70)
    print("✅ ASSOCIATION RULES COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
