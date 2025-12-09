#!/usr/bin/env python3
"""
visualization_utils.py
======================
Enhanced Visualization Utilities for Academic Research Presentations

Features:
- Comprehensive headers with algorithm descriptions
- Performance metrics tables
- Expandable recommendation tables (Best Fit, Worst Fit, Avoided)
- Interactive node highlighting
- Full-screen mode
- Consistent node styling (all circles, proper colors)
- Descriptive edge labels

Author: Recommendation Systems Research Project
Version: 3.0
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False


# =============================================================================
# ALGORITHM METADATA FOR ACADEMIC PRESENTATIONS
# =============================================================================

ALGORITHM_INFO = {
    'collaborative_filtering': {
        'title': 'Collaborative Filtering Recommendation Network',
        'subtitle': 'Memory-Based Neighborhood Approach',
        'description': '''
            Collaborative Filtering (CF) is a foundational recommendation technique that predicts 
            user preferences by collecting preferences from many users (collaborating). This visualization 
            shows the user-item interaction network where recommendations are generated based on 
            similar users (User-CF) or similar items (Item-CF).
        ''',
        'methodology': '''
            <strong>User-User CF:</strong> Finds users with similar rating patterns using cosine similarity 
            or Pearson correlation, then recommends items liked by similar users.<br><br>
            <strong>Item-Item CF:</strong> Computes item similarities based on co-rating patterns, 
            recommending items similar to those the user has already rated highly.
        ''',
        'formula': 'sim(u,v) = cos(r_u, r_v) = (r_u · r_v) / (||r_u|| × ||r_v||)',
        'complexity': 'O(n²) for similarity matrix, O(k) for k-nearest neighbors lookup',
        'technology': ['Python 3.x', 'NumPy', 'SciPy', 'Pandas', 'Pyvis', 'NetworkX'],
        'references': [
            'Resnick et al. (1994) - GroupLens: An Open Architecture for Collaborative Filtering',
            'Sarwar et al. (2001) - Item-Based Collaborative Filtering Recommendation Algorithms'
        ]
    },
    'user_collaborative_filtering': {
        'title': 'User-Based Collaborative Filtering Network',
        'subtitle': 'Finding Similar Users for Recommendations',
        'description': '''
            User-Based CF identifies users with similar tastes and recommends items that similar 
            users have liked. This graph shows user neighborhoods where connected users share 
            similar preferences, enabling preference propagation.
        ''',
        'methodology': '''
            For each target user, we compute similarity with all other users based on their 
            rating vectors. The k most similar users form the neighborhood. Predictions are 
            weighted averages of neighbor ratings, adjusted for rating biases.
        ''',
        'formula': 'r̂(u,i) = r̄_u + Σ(sim(u,v) × (r_vi - r̄_v)) / Σ|sim(u,v)|',
        'complexity': 'O(|U|²×|I|) for full similarity computation',
        'technology': ['Python 3.x', 'NumPy', 'SciPy Sparse', 'Pyvis'],
        'references': [
            'Herlocker et al. (1999) - An Algorithmic Framework for Performing Collaborative Filtering'
        ]
    },
    'item_collaborative_filtering': {
        'title': 'Item-Based Collaborative Filtering Network',
        'subtitle': 'Finding Similar Items for Recommendations',
        'description': '''
            Item-Based CF computes similarities between items based on user co-ratings. 
            When a user rates an item, similar items are recommended. This approach is 
            more scalable and stable than user-based CF for large catalogs.
        ''',
        'methodology': '''
            Item similarities are pre-computed based on the users who rated both items. 
            For prediction, we look at items the user has rated and weight similar items 
            by both similarity score and the user's rating of the source item.
        ''',
        'formula': 'r̂(u,i) = Σ(sim(i,j) × r_uj) / Σ|sim(i,j)| for j ∈ rated_by_u',
        'complexity': 'O(|I|²×|U|) for similarity, but |I| often << |U|',
        'technology': ['Python 3.x', 'NumPy', 'SciPy Sparse', 'Pyvis'],
        'references': [
            'Sarwar et al. (2001) - Item-Based Collaborative Filtering Recommendation Algorithms',
            'Linden et al. (2003) - Amazon.com Recommendations: Item-to-Item Collaborative Filtering'
        ]
    },
    'matrix_factorization': {
        'title': 'Matrix Factorization Latent Space Visualization',
        'subtitle': 'SVD / ALS / Funk SVD Decomposition',
        'description': '''
            Matrix Factorization decomposes the sparse user-item rating matrix into two dense 
            low-rank matrices representing latent factors. Users and items are embedded in a 
            shared latent space where proximity indicates affinity.
        ''',
        'methodology': '''
            <strong>SVD:</strong> Direct decomposition R = UΣV^T, truncated to k factors.<br>
            <strong>ALS:</strong> Alternating Least Squares iteratively optimizes U and V.<br>
            <strong>Funk SVD:</strong> Stochastic Gradient Descent minimizes regularized MSE.
        ''',
        'formula': 'R ≈ U × V^T, minimize ||R - UV^T||² + λ(||U||² + ||V||²)',
        'complexity': 'O(k × iterations × nnz) where k=factors, nnz=non-zero ratings',
        'technology': ['Python 3.x', 'NumPy', 'SciPy SVD', 'Pyvis'],
        'references': [
            'Koren et al. (2009) - Matrix Factorization Techniques for Recommender Systems',
            'Funk (2006) - Netflix Update: Try This at Home'
        ]
    },
    'content_based': {
        'title': 'Content-Based Filtering Feature Network',
        'subtitle': 'TF-IDF and Feature Vector Similarities',
        'description': '''
            Content-Based Filtering recommends items similar to those a user has liked, 
            based on item features (genres, descriptions, attributes). User profiles are 
            built from features of liked items, enabling recommendations without collaborative data.
        ''',
        'methodology': '''
            <strong>TF-IDF:</strong> Text features are weighted by term frequency-inverse document frequency.<br>
            <strong>Feature Vectors:</strong> Categorical attributes are one-hot encoded.<br>
            <strong>User Profile:</strong> Weighted centroid of liked item features.
        ''',
        'formula': 'TF-IDF(t,d) = tf(t,d) × log(N/df(t)), sim = cos(user_profile, item_features)',
        'complexity': 'O(|I| × |F|) for feature matrix, O(|F|) for similarity',
        'technology': ['Python 3.x', 'Scikit-learn TfidfVectorizer', 'NumPy', 'Pyvis'],
        'references': [
            'Pazzani & Billsus (2007) - Content-Based Recommendation Systems',
            'Lops et al. (2011) - Content-based Recommender Systems: State of the Art and Trends'
        ]
    },
    'graph_based': {
        'title': 'Graph-Based Recommendation Network',
        'subtitle': 'Personalized PageRank & Random Walk with Restart',
        'description': '''
            Graph-Based methods model the recommendation problem as random walks on a 
            user-item bipartite graph. Node importance is computed through iterative 
            propagation, capturing transitive relationships.
        ''',
        'methodology': '''
            <strong>Personalized PageRank (PPR):</strong> Modified PageRank with teleportation 
            back to the source user with probability (1-α).<br>
            <strong>Random Walk with Restart:</strong> Simulates a random surfer who restarts 
            at the user node, measuring steady-state item visitation probabilities.
        ''',
        'formula': 'PPR(v) = α × Σ(PPR(u)/out_degree(u)) + (1-α) × personalization(v)',
        'complexity': 'O(|E| × iterations) for power iteration convergence',
        'technology': ['Python 3.x', 'NetworkX', 'NumPy', 'Pyvis'],
        'references': [
            'Page et al. (1999) - The PageRank Citation Ranking',
            'Gori & Pucci (2007) - ItemRank: A Random-Walk Based Scoring Algorithm'
        ]
    },
    'neural_cf': {
        'title': 'Neural Collaborative Filtering Architecture',
        'subtitle': 'GMF, MLP, and NeuMF Deep Learning Models',
        'description': '''
            Neural Collaborative Filtering replaces the dot product in matrix factorization 
            with neural networks, learning complex non-linear user-item interactions from 
            embedding representations.
        ''',
        'methodology': '''
            <strong>GMF:</strong> Generalized Matrix Factorization with element-wise product.<br>
            <strong>MLP:</strong> Multi-Layer Perceptron learns from concatenated embeddings.<br>
            <strong>NeuMF:</strong> Combines GMF and MLP for linear + non-linear modeling.
        ''',
        'formula': 'ŷ = σ(h^T × (p_u ⊙ q_i)) for GMF, MLP uses f(W×[p_u;q_i] + b)',
        'complexity': 'O(batch_size × embedding_dim × hidden_layers) per iteration',
        'technology': ['Python 3.x', 'PyTorch', 'NumPy', 'Pyvis'],
        'references': [
            'He et al. (2017) - Neural Collaborative Filtering (WWW)',
            'Rendle et al. (2020) - Neural Collaborative Filtering vs. Matrix Factorization Revisited'
        ]
    },
    'hybrid': {
        'title': 'Hybrid Recommendation System Architecture',
        'subtitle': 'Weighted, Switching, and Cascade Ensemble Methods',
        'description': '''
            Hybrid Recommenders combine multiple algorithms to leverage complementary 
            strengths. This addresses limitations of individual methods, particularly 
            the cold-start problem and sparsity issues.
        ''',
        'methodology': '''
            <strong>Weighted:</strong> Linear combination of scores from multiple models.<br>
            <strong>Switching:</strong> Context-aware selection (e.g., cold-start → content-based).<br>
            <strong>Cascade:</strong> Coarse ranking followed by fine-grained reranking.
        ''',
        'formula': 'Score_hybrid = Σ(w_i × Score_i), where Σw_i = 1',
        'complexity': 'O(Σ complexity of individual models)',
        'technology': ['Python 3.x', 'Multiple Base Algorithms', 'Pyvis'],
        'references': [
            'Burke (2002) - Hybrid Recommender Systems: Survey and Experiments',
            'Adomavicius & Tuzhilin (2005) - Toward the Next Generation of Recommender Systems'
        ]
    },
    'association_rules': {
        'title': 'Association Rules Mining Network',
        'subtitle': 'Apriori Algorithm - Market Basket Analysis',
        'description': '''
            Association Rules mining discovers frequently co-occurring item sets in 
            transaction data. Rules of the form {A} → {B} indicate that users who 
            interact with A are likely to also interact with B.
        ''',
        'methodology': '''
            <strong>Apriori:</strong> Bottom-up level-wise candidate generation with pruning.<br>
            <strong>Support:</strong> Frequency of itemset in transactions.<br>
            <strong>Confidence:</strong> Conditional probability P(B|A).<br>
            <strong>Lift:</strong> Ratio of observed to expected co-occurrence.
        ''',
        'formula': 'Confidence(A→B) = P(B|A) = Support(A∪B)/Support(A), Lift = P(B|A)/P(B)',
        'complexity': 'O(2^|I|) worst case, pruned by minimum support threshold',
        'technology': ['Python 3.x', 'mlxtend', 'Pandas', 'Pyvis'],
        'references': [
            'Agrawal et al. (1994) - Fast Algorithms for Mining Association Rules',
            'Han et al. (2000) - Mining Frequent Patterns without Candidate Generation'
        ]
    },
    'popularity_baseline': {
        'title': 'Popularity-Based Baseline Recommendations',
        'subtitle': 'Non-Personalized Global Ranking Methods',
        'description': '''
            Popularity-based methods recommend globally popular items to all users. 
            While simple, they serve as essential baselines and are surprisingly 
            effective, especially for cold-start users.
        ''',
        'methodology': '''
            <strong>Count-Based:</strong> Rank by number of interactions.<br>
            <strong>Rating-Based:</strong> Rank by average rating.<br>
            <strong>Bayesian Average:</strong> IMDB-style weighted mean toward global average.<br>
            <strong>Time-Weighted:</strong> Recent interactions weighted higher.
        ''',
        'formula': 'Bayesian = (v×R + m×C)/(v+m) where v=votes, R=avg, m=threshold, C=global_avg',
        'complexity': 'O(|I|) for ranking, O(1) for lookup',
        'technology': ['Python 3.x', 'Pandas', 'NumPy', 'Pyvis'],
        'references': [
            'Cremonesi et al. (2010) - Performance of Recommender Algorithms on Top-N Recommendation Tasks'
        ]
    },
    # Alias for popularity
    'popularity': {
        'title': 'Popularity-Based Baseline Recommendations',
        'subtitle': 'Non-Personalized Global Ranking Methods',
        'description': '''
            Popularity-based methods recommend globally popular items to all users. 
            While simple, they serve as essential baselines and are surprisingly 
            effective, especially for cold-start users.
        ''',
        'methodology': '''
            <strong>Count-Based:</strong> Rank by number of interactions.<br>
            <strong>Rating-Based:</strong> Rank by average rating.<br>
            <strong>Bayesian Average:</strong> IMDB-style weighted mean toward global average.<br>
            <strong>Time-Weighted:</strong> Recent interactions weighted higher.
        ''',
        'formula': 'Bayesian = (v×R + m×C)/(v+m) where v=votes, R=avg, m=threshold, C=global_avg',
        'complexity': 'O(|I|) for ranking, O(1) for lookup',
        'technology': ['Python 3.x', 'Pandas', 'NumPy', 'Pyvis'],
        'references': [
            'Cremonesi et al. (2010) - Performance of Recommender Algorithms on Top-N Recommendation Tasks'
        ]
    }
}


# =============================================================================
# RECOMMENDATION TABLE DATA STRUCTURE
# =============================================================================

def format_recommendation_tables(
    recommendations: List[Tuple] = None,
    best_fit: List[Tuple] = None,
    worst_fit: List[Tuple] = None,
    avoided: List[Tuple] = None
) -> str:
    """Generate HTML for expandable recommendation tables."""
    
    tables_html = """
    <div class="recommendation-tables">
        <h3>📋 Recommendation Analysis</h3>
    """
    
    if recommendations:
        tables_html += f"""
        <div class="table-section">
            <div class="table-header" onclick="toggleTable('rec-table')">
                <span class="toggle-icon" id="rec-table-icon">▶</span>
                <span class="table-title">🎯 Top Recommendations</span>
                <span class="table-count">{len(recommendations)} items</span>
            </div>
            <div class="table-content" id="rec-table" style="display:none;">
                <table class="data-table">
                    <thead><tr><th>Rank</th><th>Item ID</th><th>Predicted Score</th><th>Confidence</th><th>Reason</th></tr></thead>
                    <tbody>
        """
        for i, item in enumerate(recommendations[:20], 1):
            item_id = item[0] if len(item) > 0 else 'N/A'
            score = item[1] if len(item) > 1 else 0
            reason = item[2] if len(item) > 2 else 'High predicted preference based on similar user/item patterns'
            confidence = 'High' if score > 4 else 'Medium' if score > 3 else 'Low'
            conf_class = 'high' if score > 4 else 'medium' if score > 3 else 'low'
            tables_html += f'<tr><td>{i}</td><td><strong>Item {item_id}</strong></td><td>{score:.3f}</td><td><span class="confidence {conf_class}">{confidence}</span></td><td>{reason}</td></tr>'
        tables_html += "</tbody></table></div></div>"
    
    if best_fit:
        tables_html += f"""
        <div class="table-section">
            <div class="table-header" onclick="toggleTable('best-table')">
                <span class="toggle-icon" id="best-table-icon">▶</span>
                <span class="table-title">✅ Best Fit Items (Highest Rated)</span>
                <span class="table-count">{len(best_fit)} items</span>
            </div>
            <div class="table-content" id="best-table" style="display:none;">
                <table class="data-table">
                    <thead><tr><th>Rank</th><th>Item ID</th><th>User Rating</th><th>Popularity</th><th>Match Quality</th></tr></thead>
                    <tbody>
        """
        for i, item in enumerate(best_fit[:15], 1):
            item_id = item[0] if len(item) > 0 else 'N/A'
            rating = item[1] if len(item) > 1 else 0
            popularity = item[2] if len(item) > 2 else 'N/A'
            tables_html += f'<tr><td>{i}</td><td><strong>Item {item_id}</strong></td><td>⭐ {rating:.1f}</td><td>{popularity}</td><td><span class="match excellent">Excellent</span></td></tr>'
        tables_html += "</tbody></table></div></div>"
    
    if worst_fit:
        tables_html += f"""
        <div class="table-section">
            <div class="table-header" onclick="toggleTable('worst-table')">
                <span class="toggle-icon" id="worst-table-icon">▶</span>
                <span class="table-title">❌ Worst Fit Items (Lowest Rated)</span>
                <span class="table-count">{len(worst_fit)} items</span>
            </div>
            <div class="table-content" id="worst-table" style="display:none;">
                <table class="data-table">
                    <thead><tr><th>Rank</th><th>Item ID</th><th>User Rating</th><th>Global Avg</th><th>Mismatch</th></tr></thead>
                    <tbody>
        """
        for i, item in enumerate(worst_fit[:10], 1):
            item_id = item[0] if len(item) > 0 else 'N/A'
            rating = item[1] if len(item) > 1 else 0
            global_avg = item[2] if len(item) > 2 else 'N/A'
            tables_html += f'<tr><td>{i}</td><td><strong>Item {item_id}</strong></td><td>⭐ {rating:.1f}</td><td>{global_avg}</td><td><span class="match poor">Poor</span></td></tr>'
        tables_html += "</tbody></table></div></div>"
    
    if avoided:
        tables_html += f"""
        <div class="table-section">
            <div class="table-header" onclick="toggleTable('avoided-table')">
                <span class="toggle-icon" id="avoided-table-icon">▶</span>
                <span class="table-title">🚫 Avoided Items (Not Recommended)</span>
                <span class="table-count">{len(avoided)} items</span>
            </div>
            <div class="table-content" id="avoided-table" style="display:none;">
                <table class="data-table">
                    <thead><tr><th>Item ID</th><th>Predicted Score</th><th>Reason Not Recommended</th></tr></thead>
                    <tbody>
        """
        for item in avoided[:10]:
            item_id = item[0] if len(item) > 0 else 'N/A'
            score = item[1] if len(item) > 1 else 0
            reason = item[2] if len(item) > 2 else 'Low predicted preference / Already rated / Insufficient data'
            tables_html += f'<tr><td><strong>Item {item_id}</strong></td><td>{score:.3f}</td><td>{reason}</td></tr>'
        tables_html += "</tbody></table></div></div>"
    
    tables_html += "</div>"
    return tables_html


# =============================================================================
# ENHANCED HTML TEMPLATE
# =============================================================================

def get_enhanced_html_template(
    algorithm_key: str,
    metrics: Dict[str, Any] = None,
    graph_stats: Dict[str, Any] = None,
    recommendations_data: Dict[str, List] = None,
    custom_title: str = None,
    custom_description: str = None
) -> str:
    """Generate enhanced HTML wrapper with academic presentation features."""
    
    info = ALGORITHM_INFO.get(algorithm_key, ALGORITHM_INFO.get('collaborative_filtering'))
    title = custom_title or info['title']
    description = custom_description or info['description']
    
    # Format metrics table
    metrics_html = ""
    if metrics:
        metrics_html = '<div class="metrics-section"><h3>📊 Performance Metrics</h3><table class="metrics-table"><thead><tr><th>Metric</th><th>Value</th><th>Description</th></tr></thead><tbody>'
        metric_descriptions = {
            'RMSE': 'Root Mean Square Error - Lower is better',
            'MAE': 'Mean Absolute Error - Lower is better',
            'Precision@10': 'Precision at 10 - Higher is better',
            'Recall@10': 'Recall at 10 - Higher is better',
            'NDCG@10': 'Normalized DCG at 10 - Higher is better',
            'Coverage': 'Catalog coverage percentage',
            'n_users': 'Number of users in dataset',
            'n_items': 'Number of items in dataset',
            'n_ratings': 'Number of ratings/interactions'
        }
        for key, value in metrics.items():
            desc = metric_descriptions.get(key, '')
            value_str = f"{value:.4f}" if isinstance(value, float) else str(value)
            metrics_html += f'<tr><td><strong>{key}</strong></td><td>{value_str}</td><td class="metric-desc">{desc}</td></tr>'
        metrics_html += '</tbody></table></div>'
    
    # Graph stats
    graph_stats_html = ""
    if graph_stats:
        graph_stats_html = '<div class="graph-stats"><h4>📈 Graph Statistics</h4><div class="stats-grid">'
        for key, value in graph_stats.items():
            graph_stats_html += f'<div class="stat-item"><span class="stat-value">{value}</span><span class="stat-label">{key}</span></div>'
        graph_stats_html += '</div></div>'
    
    tech_html = ""
    if 'technology' in info:
        tech_html = "<div class='tech-stack'><strong>Technology:</strong> " + " • ".join(info['technology']) + "</div>"
    
    refs_html = ""
    if 'references' in info:
        refs_html = '<div class="references"><h4>📚 Key References</h4><ul>'
        for ref in info['references']:
            refs_html += f"<li>{ref}</li>"
        refs_html += "</ul></div>"
    
    rec_tables_html = ""
    if recommendations_data:
        rec_tables_html = format_recommendation_tables(
            recommendations=recommendations_data.get('recommendations'),
            best_fit=recommendations_data.get('best_fit'),
            worst_fit=recommendations_data.get('worst_fit'),
            avoided=recommendations_data.get('avoided')
        )
    
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: #e0e0e0; min-height: 100vh; }}
        .header {{ background: linear-gradient(135deg, #0f3460 0%, #16213e 100%); padding: 30px 40px; border-bottom: 3px solid #e94560; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }}
        .header h1 {{ font-size: 2.2em; color: #ffffff; margin-bottom: 8px; }}
        .header .subtitle {{ font-size: 1.2em; color: #e94560; font-weight: 500; margin-bottom: 15px; }}
        .header .description {{ font-size: 1em; color: #b0b0b0; line-height: 1.6; max-width: 1200px; }}
        .timestamp {{ font-size: 0.85em; color: #888; margin-top: 10px; }}
        .main-container {{ display: flex; flex-direction: column; padding: 20px; gap: 20px; }}
        .info-panel {{ background: rgba(255,255,255,0.05); border-radius: 12px; padding: 25px; border: 1px solid rgba(255,255,255,0.1); }}
        .info-panel h3 {{ color: #e94560; margin-bottom: 15px; font-size: 1.3em; }}
        .info-panel h4 {{ color: #4fc3f7; margin: 20px 0 10px 0; font-size: 1.1em; }}
        .methodology-box {{ background: rgba(0,0,0,0.2); padding: 20px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #e94560; }}
        .formula {{ font-family: 'Courier New', monospace; background: rgba(233, 69, 96, 0.1); padding: 12px 20px; border-radius: 6px; margin: 15px 0; font-size: 1.1em; color: #4fc3f7; }}
        .complexity {{ color: #ffd700; font-size: 0.95em; }}
        .tech-stack {{ margin-top: 15px; padding: 10px 15px; background: rgba(79, 195, 247, 0.1); border-radius: 6px; font-size: 0.9em; color: #4fc3f7; }}
        .metrics-section {{ margin-top: 20px; }}
        .metrics-table {{ width: 100%; border-collapse: collapse; margin-top: 15px; background: rgba(0,0,0,0.2); border-radius: 8px; overflow: hidden; }}
        .metrics-table th {{ background: rgba(233, 69, 96, 0.3); padding: 12px 15px; text-align: left; font-weight: 600; color: #fff; }}
        .metrics-table td {{ padding: 10px 15px; border-bottom: 1px solid rgba(255,255,255,0.05); }}
        .metrics-table tr:hover {{ background: rgba(255,255,255,0.05); }}
        .metric-desc {{ color: #888; font-size: 0.85em; }}
        .graph-stats {{ margin-top: 20px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 15px; margin-top: 15px; }}
        .stat-item {{ background: rgba(0,0,0,0.3); padding: 15px; border-radius: 8px; text-align: center; }}
        .stat-value {{ display: block; font-size: 1.8em; font-weight: bold; color: #e94560; }}
        .stat-label {{ display: block; font-size: 0.85em; color: #888; margin-top: 5px; }}
        .references {{ margin-top: 20px; padding: 15px; background: rgba(0,0,0,0.2); border-radius: 8px; }}
        .references ul {{ margin-left: 20px; margin-top: 10px; }}
        .references li {{ margin: 8px 0; color: #b0b0b0; font-size: 0.9em; }}
        .recommendation-tables {{ margin-top: 25px; }}
        .table-section {{ margin: 15px 0; border-radius: 8px; overflow: hidden; background: rgba(0,0,0,0.2); }}
        .table-header {{ display: flex; align-items: center; padding: 15px 20px; background: rgba(233, 69, 96, 0.2); cursor: pointer; transition: background 0.3s; }}
        .table-header:hover {{ background: rgba(233, 69, 96, 0.3); }}
        .toggle-icon {{ margin-right: 10px; font-size: 0.9em; transition: transform 0.3s; }}
        .toggle-icon.open {{ transform: rotate(90deg); }}
        .table-title {{ flex: 1; font-weight: 600; color: #fff; }}
        .table-count {{ color: #888; font-size: 0.9em; }}
        .table-content {{ max-height: 400px; overflow-y: auto; }}
        .data-table {{ width: 100%; border-collapse: collapse; }}
        .data-table th {{ background: rgba(0,0,0,0.3); padding: 12px 15px; text-align: left; font-weight: 600; color: #4fc3f7; position: sticky; top: 0; }}
        .data-table td {{ padding: 10px 15px; border-bottom: 1px solid rgba(255,255,255,0.05); }}
        .data-table tr:hover {{ background: rgba(255,255,255,0.05); }}
        .confidence, .match {{ padding: 3px 8px; border-radius: 4px; font-size: 0.85em; }}
        .confidence.high, .match.excellent {{ background: rgba(52, 168, 83, 0.3); color: #34a853; }}
        .confidence.medium {{ background: rgba(251, 188, 4, 0.3); color: #fbbc04; }}
        .confidence.low, .match.poor {{ background: rgba(234, 67, 53, 0.3); color: #ea4335; }}
        .graph-section {{ position: relative; }}
        .graph-header {{ display: flex; justify-content: space-between; align-items: center; padding: 15px 20px; background: rgba(255,255,255,0.05); border-radius: 12px 12px 0 0; }}
        .graph-header h3 {{ color: #4fc3f7; }}
        .graph-controls {{ display: flex; gap: 10px; }}
        .control-btn {{ background: rgba(233, 69, 96, 0.2); border: 1px solid #e94560; color: #e94560; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 0.9em; transition: all 0.3s ease; }}
        .control-btn:hover {{ background: #e94560; color: #fff; }}
        .graph-container {{ background: #ffffff; border-radius: 0 0 12px 12px; overflow: hidden; position: relative; }}
        .graph-container.fullscreen {{ position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: 9999; border-radius: 0; }}
        .graph-container.fullscreen .exit-fullscreen {{ position: absolute; top: 20px; right: 20px; z-index: 10000; background: #e94560; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; }}
        #mynetwork {{ width: 100%; height: 700px; border: none; }}
        .fullscreen #mynetwork {{ height: 100vh; }}
        .legend {{ position: absolute; top: 20px; right: 20px; background: rgba(255,255,255,0.95); padding: 20px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); font-size: 13px; color: #333; max-width: 280px; z-index: 100; }}
        .legend h4 {{ margin-bottom: 12px; color: #0f3460; font-size: 1.1em; border-bottom: 2px solid #e94560; padding-bottom: 8px; }}
        .legend-item {{ display: flex; align-items: center; margin: 10px 0; }}
        .legend-color {{ width: 20px; height: 20px; border-radius: 50%; margin-right: 12px; flex-shrink: 0; border: 2px solid rgba(0,0,0,0.1); }}
        .legend-line {{ width: 35px; height: 4px; margin-right: 12px; flex-shrink: 0; border-radius: 2px; }}
        .legend-line.dashed {{ background: repeating-linear-gradient(90deg, currentColor, currentColor 5px, transparent 5px, transparent 10px); height: 3px; }}
        .legend-text {{ line-height: 1.3; }}
        .legend-text strong {{ display: block; color: #333; }}
        .legend-text small {{ color: #666; font-size: 0.85em; }}
        .instructions {{ background: rgba(79, 195, 247, 0.1); padding: 15px 20px; border-radius: 8px; margin-top: 15px; border-left: 4px solid #4fc3f7; }}
        .instructions h4 {{ color: #4fc3f7; margin-bottom: 10px; }}
        .instructions ul {{ margin-left: 20px; }}
        .instructions li {{ margin: 5px 0; color: #b0b0b0; }}
        .selection-info {{ position: absolute; bottom: 20px; left: 20px; background: rgba(0,0,0,0.9); color: white; padding: 15px 20px; border-radius: 8px; font-size: 0.9em; max-width: 350px; display: none; z-index: 100; border: 1px solid #e94560; }}
        .selection-info.visible {{ display: block; }}
        .selection-info h5 {{ color: #e94560; margin-bottom: 8px; }}
        .footer {{ text-align: center; padding: 20px; color: #666; font-size: 0.85em; border-top: 1px solid rgba(255,255,255,0.1); margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 {title}</h1>
        <div class="subtitle">{info.get('subtitle', '')}</div>
        <div class="description">{description.strip()}</div>
        <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>
    <div class="main-container">
        <div class="info-panel">
            <h3>📋 Methodology</h3>
            <div class="methodology-box">{info.get('methodology', '')}</div>
            <h4>Mathematical Formulation</h4>
            <div class="formula">{info.get('formula', 'N/A')}</div>
            <div class="complexity"><strong>Computational Complexity:</strong> {info.get('complexity', 'N/A')}</div>
            {tech_html}
            {metrics_html}
            {graph_stats_html}
            {rec_tables_html}
            {refs_html}
            <div class="instructions">
                <h4>🖱️ Interaction Guide</h4>
                <ul>
                    <li><strong>Click</strong> a node to highlight its connections</li>
                    <li><strong>Double-click</strong> to focus and zoom on a node</li>
                    <li><strong>Scroll</strong> to zoom in/out</li>
                    <li><strong>Drag</strong> nodes to rearrange the layout</li>
                    <li><strong>Hover</strong> over nodes/edges for detailed information</li>
                </ul>
            </div>
        </div>
        <div class="graph-section">
            <div class="graph-header">
                <h3>🔗 Interactive Network Graph</h3>
                <div class="graph-controls">
                    <button class="control-btn" onclick="resetView()">↺ Reset View</button>
                    <button class="control-btn" onclick="togglePhysics()">⚡ Toggle Physics</button>
                    <button class="control-btn" onclick="toggleFullscreen()">⛶ Full Screen</button>
                </div>
            </div>
            <div class="graph-container" id="graphContainer">
                <button class="exit-fullscreen" onclick="toggleFullscreen()" style="display:none;">✕ Exit Full Screen</button>
                <div id="mynetwork"></div>
                <div class="selection-info" id="selectionInfo"><h5>Selected Node</h5><div id="selectionContent">Click a node to see details</div></div>
                <!-- LEGEND_PLACEHOLDER -->
            </div>
        </div>
    </div>
    <div class="footer"><p>Recommendation System Research | Interactive Network Visualization</p></div>
    <script>
        var physicsEnabled = true;
        var network = null;
        function toggleTable(tableId) {{
            var content = document.getElementById(tableId);
            var icon = document.getElementById(tableId + '-icon');
            if (content.style.display === 'none') {{ content.style.display = 'block'; icon.classList.add('open'); }}
            else {{ content.style.display = 'none'; icon.classList.remove('open'); }}
        }}
        function toggleFullscreen() {{
            var container = document.getElementById('graphContainer');
            var exitBtn = container.querySelector('.exit-fullscreen');
            if (container.classList.contains('fullscreen')) {{
                container.classList.remove('fullscreen'); exitBtn.style.display = 'none'; document.body.style.overflow = 'auto';
            }} else {{
                container.classList.add('fullscreen'); exitBtn.style.display = 'block'; document.body.style.overflow = 'hidden';
            }}
            if (network) {{ setTimeout(function() {{ network.fit(); }}, 100); }}
        }}
        function resetView() {{ if (network) {{ network.fit(); }} }}
        function togglePhysics() {{ if (network) {{ physicsEnabled = !physicsEnabled; network.setOptions({{ physics: {{ enabled: physicsEnabled }} }}); }} }}
        document.addEventListener('keydown', function(e) {{ if (e.key === 'Escape') {{ var container = document.getElementById('graphContainer'); if (container.classList.contains('fullscreen')) {{ toggleFullscreen(); }} }} }});
        function showSelectionInfo(nodeData) {{
            var infoPanel = document.getElementById('selectionInfo');
            var content = document.getElementById('selectionContent');
            if (nodeData) {{ content.innerHTML = '<strong>' + nodeData.label + '</strong><br>' + (nodeData.title || 'No additional info'); infoPanel.classList.add('visible'); }}
            else {{ infoPanel.classList.remove('visible'); }}
        }}
    </script>
    <!-- PYVIS_CONTENT -->
</body>
</html>'''


def create_enhanced_legend(node_types: List[Dict], edge_types: List[Dict]) -> str:
    """Create enhanced legend HTML with descriptions."""
    legend_html = '<div class="legend"><h4>📍 Legend</h4><div class="legend-section"><strong style="display:block; margin-bottom:10px; color:#0f3460;">Node Types</strong>'
    for node in node_types:
        desc = node.get('description', '')
        legend_html += f'<div class="legend-item"><div class="legend-color" style="background:{node["color"]};"></div><div class="legend-text"><strong>{node["label"]}</strong>{f"<small>{desc}</small>" if desc else ""}</div></div>'
    legend_html += '</div><hr style="margin: 15px 0; border: none; border-top: 1px solid #ddd;"><div class="legend-section"><strong style="display:block; margin-bottom:10px; color:#0f3460;">Edge Types</strong>'
    for edge in edge_types:
        desc = edge.get('description', '')
        style_class = 'dashed' if edge.get('style') == 'dashed' else ''
        legend_html += f'<div class="legend-item"><div class="legend-line {style_class}" style="background:{edge["color"]}; color:{edge["color"]};"></div><div class="legend-text"><strong>{edge["label"]}</strong>{f"<small>{desc}</small>" if desc else ""}</div></div>'
    legend_html += '</div></div>'
    return legend_html


def add_neighbor_highlight_script() -> str:
    """JavaScript to highlight connected nodes when a node is selected."""
    return '''
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            var checkNetwork = setInterval(function() {
                if (typeof network !== 'undefined' && network !== null) { clearInterval(checkNetwork); setupHighlighting(); }
            }, 100);
        });
        function setupHighlighting() {
            var highlightActive = false;
            network.on("click", function(params) {
                if (params.nodes.length > 0) { highlightConnectedNodes(params.nodes[0]); } else { resetHighlight(); }
            });
            function highlightConnectedNodes(selectedNode) {
                var connectedNodes = network.getConnectedNodes(selectedNode);
                var connectedEdges = network.getConnectedEdges(selectedNode);
                connectedNodes.push(selectedNode);
                var updateNodes = [];
                var nodes = network.body.data.nodes;
                nodes.forEach(function(node) {
                    var nodeId = node.id;
                    if (connectedNodes.indexOf(nodeId) !== -1) { updateNodes.push({ id: nodeId, opacity: 1, shadow: { enabled: true, color: 'rgba(233,69,96,0.5)', size: 20 } }); }
                    else { updateNodes.push({ id: nodeId, opacity: 0.15, shadow: { enabled: false } }); }
                });
                nodes.update(updateNodes);
                var updateEdges = [];
                var edges = network.body.data.edges;
                edges.forEach(function(edge) {
                    var edgeId = edge.id;
                    if (connectedEdges.indexOf(edgeId) !== -1) { updateEdges.push({ id: edgeId, width: 5, shadow: { enabled: true } }); }
                    else { updateEdges.push({ id: edgeId, width: 0.3, shadow: { enabled: false } }); }
                });
                edges.update(updateEdges);
                highlightActive = true;
                var nodeData = nodes.get(selectedNode);
                showSelectionInfo(nodeData);
            }
            function resetHighlight() {
                if (!highlightActive) return;
                var nodes = network.body.data.nodes;
                var edges = network.body.data.edges;
                var updateNodes = [];
                nodes.forEach(function(node) { updateNodes.push({ id: node.id, opacity: 1, shadow: { enabled: true, size: 10 } }); });
                nodes.update(updateNodes);
                var updateEdges = [];
                edges.forEach(function(edge) { updateEdges.push({ id: edge.id, width: 2, shadow: { enabled: false } }); });
                edges.update(updateEdges);
                highlightActive = false;
                showSelectionInfo(null);
            }
        }
    </script>
    '''


def generate_enhanced_visualization(
    net: 'Network',
    algorithm_key: str,
    output_path: str,
    metrics: Dict[str, Any] = None,
    graph_stats: Dict[str, Any] = None,
    node_types: List[Dict] = None,
    edge_types: List[Dict] = None,
    recommendations_data: Dict[str, List] = None,
    custom_title: str = None,
    custom_description: str = None
) -> str:
    """Generate enhanced visualization with all academic features."""
    
    if not PYVIS_AVAILABLE:
        print("Warning: Pyvis not available")
        return None
    
    # Save basic network first
    net.save_graph(output_path)
    
    # Read the generated HTML to extract vis.js data
    with open(output_path, 'r', encoding='utf-8') as f:
        pyvis_html = f.read()
    
    import re
    draw_match = re.search(r'function drawGraph\(\)\s*\{(.*?)return network;', pyvis_html, re.DOTALL)
    
    if draw_match:
        # Don't redeclare 'network' - it's already declared in the template
        vis_script = f"""
        <script type="text/javascript">
            function drawGraph() {{
                var container = document.getElementById('mynetwork');
                {draw_match.group(1)}
                return network;
            }}
            drawGraph();
        </script>
        """
    else:
        scripts = re.findall(r'<script[^>]*>.*?</script>', pyvis_html, re.DOTALL)
        vis_script = '\n'.join(scripts)
    
    template = get_enhanced_html_template(
        algorithm_key=algorithm_key,
        metrics=metrics,
        graph_stats=graph_stats,
        recommendations_data=recommendations_data,
        custom_title=custom_title,
        custom_description=custom_description
    )
    
    if node_types is None:
        node_types = [
            {"color": "#4285f4", "label": "Users", "description": "User nodes in the network"},
            {"color": "#34a853", "label": "Items", "description": "Item/product nodes"},
            {"color": "#fbbc04", "label": "Recommended", "description": "Recommended items"}
        ]
    
    if edge_types is None:
        edge_types = [
            {"color": "#888888", "label": "Interaction", "style": "solid", "description": "Existing rating or interaction"},
            {"color": "#ea4335", "label": "Recommendation", "style": "dashed", "description": "Predicted preference"}
        ]
    
    legend_html = create_enhanced_legend(node_types, edge_types)
    final_html = template.replace('<!-- LEGEND_PLACEHOLDER -->', legend_html)
    final_html = final_html.replace('<!-- PYVIS_CONTENT -->', vis_script + add_neighbor_highlight_script())
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_html)
    
    print(f"   ✅ Enhanced visualization saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    print("Visualization Utilities Module v3.0")
    print("=" * 50)
    print("\nFeatures: Academic headers, Expandable tables, Node highlighting, Full-screen mode")
