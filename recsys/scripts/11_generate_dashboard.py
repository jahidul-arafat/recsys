"""
11_generate_dashboard.py
=========================
Enhanced Dashboard with Algorithm Explanations, Visuals, and Multi-Dataset Results
"""

import os
import sys
import json
import base64
from datetime import datetime
import pandas as pd

os.makedirs("../visualizations", exist_ok=True)


def encode_image_to_base64(image_path: str) -> str:
    try:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except:
        return ""


def load_json(path: str) -> dict:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return {}


def generate_hyperparameter_tab_content(hp_data):
    """Generate comprehensive HTML content for hyperparameter tab"""
    
    if hp_data['results'] is None:
        return """
        <div class="alert alert-warning">
            <h4>⚠️ Hyperparameter tuning not yet run</h4>
            <p>Run <code>python 09_hyperparameter_tuning.py</code> to generate hyperparameter analysis.</p>
        </div>
        """
    
    results_df = hp_data['results']
    summary = hp_data['summary'] or {}
    
    # Overview stats
    content = """
    <div class="hp-container">
        <h2>🔬 Hyperparameter Tuning Analysis</h2>
        <p class="subtitle">Comprehensive analysis of what hyperparameters were tested, why they matter, and what we found.</p>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{total_configs}</div>
                <div class="stat-label">Total Configurations</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{num_algorithms}</div>
                <div class="stat-label">Algorithms Tested</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{best_rmse:.4f}</div>
                <div class="stat-label">Best RMSE</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{best_ndcg:.4f}</div>
                <div class="stat-label">Best NDCG@10</div>
            </div>
        </div>
    """.format(
        total_configs=len(results_df),
        num_algorithms=results_df['algorithm'].nunique(),
        best_rmse=results_df['RMSE'].min(),
        best_ndcg=results_df['NDCG@10'].max()
    )
    
    # Methodology section
    content += """
        <div class="section methodology-section">
            <h3>📋 Tuning Methodology</h3>
            <div class="methodology-box">
                <h4>How Hyperparameters Were Tested</h4>
                <ol>
                    <li><strong>Grid Search</strong>: All combinations of parameter values were systematically tested</li>
                    <li><strong>Consistent Evaluation</strong>: Same train/test split (80/20) used across all configurations</li>
                    <li><strong>Multiple Metrics</strong>: RMSE, MAE, Precision@K, Recall@K, NDCG@K, Coverage computed for each</li>
                    <li><strong>Training Time</strong>: Recorded for each configuration to assess efficiency trade-offs</li>
                </ol>
                
                <h4>Why Hyperparameters Matter</h4>
                <p>Each hyperparameter controls a different aspect of the recommendation algorithm:</p>
                <ul>
                    <li><strong>Model Capacity</strong> (n_factors, k_neighbors): How much the model can learn - too low = underfitting, too high = overfitting</li>
                    <li><strong>Regularization</strong> (lambda, min_ratings): Prevents overfitting by constraining model complexity</li>
                    <li><strong>Algorithm Behavior</strong> (method, similarity_metric): Fundamental approach and how similarity is computed</li>
                    <li><strong>Feature Extraction</strong> (tfidf_max_features, ngram_range): How input data is transformed into features</li>
                </ul>
            </div>
        </div>
    """
    
    # Get hyperparameter info from summary
    hp_info = summary.get('hyperparameter_info', {})
    
    # If no hp_info in summary, create default explanations
    if not hp_info:
        hp_info = {
            'Collaborative Filtering': {
                'description': 'Collaborative Filtering recommends items based on user-item interaction patterns. It finds similar users (User-CF) or similar items (Item-CF) and uses their ratings to predict preferences.',
                'parameters': {
                    'method': {
                        'values': ['user', 'item'],
                        'impact': 'High',
                        'explanation': 'User-based finds similar users ("people like you liked X"), Item-based finds similar items ("you liked X, try similar Y"). Item-based is more stable and scalable.'
                    },
                    'k_neighbors': {
                        'values': [5, 10, 20, 30, 50],
                        'impact': 'Medium-High',
                        'explanation': 'Number of neighbors to consider. Small k = more personalized but noisy. Large k = more stable but diluted signal. Sweet spot: 10-30.'
                    },
                    'similarity_metric': {
                        'values': ['cosine', 'pearson'],
                        'impact': 'Medium',
                        'explanation': 'Cosine ignores rating scale (good for sparse data). Pearson accounts for user biases (good for dense data).'
                    }
                }
            },
            'Matrix Factorization': {
                'description': 'Matrix Factorization decomposes the rating matrix into user and item latent factor matrices. Each user/item is represented as a vector capturing hidden preferences.',
                'parameters': {
                    'method': {
                        'values': ['svd', 'als', 'funk_svd'],
                        'impact': 'High',
                        'explanation': 'SVD: Direct decomposition. ALS: Iterative, good for implicit feedback. Funk SVD: SGD-based, learns from observed ratings only.'
                    },
                    'n_factors': {
                        'values': [10, 20, 50, 100],
                        'impact': 'High',
                        'explanation': 'Dimensionality of latent space. More factors = more expressive but risk overfitting. Typical range: 20-100.'
                    },
                    'regularization': {
                        'values': [0.01, 0.02, 0.05, 0.1],
                        'impact': 'Medium-High',
                        'explanation': 'L2 penalty on factor magnitudes. Higher = more regularization, prevents overfitting but may underfit.'
                    }
                }
            },
            'Content-Based': {
                'description': 'Content-Based filtering recommends items similar to what users liked before, based on item features like genres, descriptions, and tags.',
                'parameters': {
                    'tfidf_max_features': {
                        'values': [100, 300, 500, 1000],
                        'impact': 'Medium',
                        'explanation': 'Maximum vocabulary size for TF-IDF. More features = richer representation but more noise and computation.'
                    },
                    'tfidf_ngram_range': {
                        'values': ['(1,1)', '(1,2)', '(1,3)'],
                        'impact': 'Medium',
                        'explanation': '(1,1): unigrams only. (1,2): adds bigrams like "romantic comedy". (1,3): captures longer phrases.'
                    },
                    'min_df': {
                        'values': [1, 2, 5, 10],
                        'impact': 'Low-Medium',
                        'explanation': 'Minimum document frequency. Higher values filter rare terms, reducing noise but may lose specificity.'
                    }
                }
            },
            'Graph-Based': {
                'description': 'Graph-Based methods model recommendations as a graph traversal problem, using PageRank or random walks to find relevant items.',
                'parameters': {
                    'method': {
                        'values': ['pagerank', 'random_walk'],
                        'impact': 'High',
                        'explanation': 'PageRank: Global importance scores. Random Walk: Local exploration from user node, more serendipitous.'
                    },
                    'damping_factor': {
                        'values': [0.7, 0.8, 0.85, 0.9, 0.95],
                        'impact': 'High',
                        'explanation': 'Probability of following edges vs jumping back. Higher = explores further, more diverse. Standard: 0.85.'
                    },
                    'n_walks': {
                        'values': [5, 10, 20, 50],
                        'impact': 'Medium',
                        'explanation': 'Number of random walks per user. More walks = more stable results but slower.'
                    }
                }
            },
            'Popularity': {
                'description': 'Popularity-based methods recommend the most popular or highest-rated items. Simple but effective baseline.',
                'parameters': {
                    'method': {
                        'values': ['count', 'rating', 'weighted', 'bayesian'],
                        'impact': 'High',
                        'explanation': 'Count: most-rated items. Rating: highest-rated. Weighted: blend both. Bayesian: IMDB-style weighted average.'
                    },
                    'popularity_weight': {
                        'values': [0.3, 0.5, 0.7],
                        'impact': 'Medium',
                        'explanation': 'Balance between popularity (count) and quality (rating). 0.5 = equal weight.'
                    },
                    'min_ratings': {
                        'values': [3, 5, 10, 20],
                        'impact': 'Medium',
                        'explanation': 'Minimum ratings required. Higher = more reliable scores but excludes long-tail items.'
                    }
                }
            }
        }
    
    # Per-algorithm detailed sections
    for algo_name, algo_info in hp_info.items():
        # Find matching results
        algo_key = algo_name.split()[0]  # Get first word for matching
        algo_results = results_df[results_df['algorithm'].str.contains(algo_key, case=False, na=False)]
        
        if len(algo_results) == 0:
            continue
        
        best_idx = algo_results['RMSE'].idxmin()
        best_row = algo_results.loc[best_idx]
        
        content += f"""
        <div class="section algorithm-section">
            <h3>🔹 {algo_name}</h3>
            <p class="algo-description">{algo_info.get('description', '').strip()}</p>
            
            <div class="results-summary">
                <div class="best-config">
                    <h4>✅ Best Configuration</h4>
                    <table class="config-table">
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>RMSE</td><td><strong>{best_row['RMSE']:.4f}</strong></td></tr>
                        <tr><td>NDCG@10</td><td><strong>{best_row['NDCG@10']:.4f}</strong></td></tr>
                        <tr><td>Coverage</td><td>{best_row['Coverage']:.2%}</td></tr>
                        <tr><td>Train Time</td><td>{best_row.get('train_time', 'N/A')}s</td></tr>
                    </table>
                </div>
                
                <div class="config-range">
                    <h4>📊 Results Range ({len(algo_results)} configurations)</h4>
                    <table class="config-table">
                        <tr><th>Metric</th><th>Min</th><th>Max</th><th>Std</th></tr>
                        <tr>
                            <td>RMSE</td>
                            <td>{algo_results['RMSE'].min():.4f}</td>
                            <td>{algo_results['RMSE'].max():.4f}</td>
                            <td>{algo_results['RMSE'].std():.4f}</td>
                        </tr>
                        <tr>
                            <td>NDCG@10</td>
                            <td>{algo_results['NDCG@10'].min():.4f}</td>
                            <td>{algo_results['NDCG@10'].max():.4f}</td>
                            <td>{algo_results['NDCG@10'].std():.4f}</td>
                        </tr>
                    </table>
                </div>
            </div>
            
            <div class="hyperparameters">
                <h4>🎛️ Hyperparameters Tested</h4>
        """
        
        # Add parameter cards
        for param_name, param_info in algo_info.get('parameters', {}).items():
            explanation = param_info.get('explanation', '').strip().replace('\n', '<br>')
            impact = param_info.get('impact', 'Medium')
            impact_class = impact.split('-')[0].lower() if impact else 'medium'
            values = param_info.get('values', [])
            
            content += f"""
                <div class="param-card">
                    <div class="param-header">
                        <span class="param-name">{param_name}</span>
                        <span class="param-impact impact-{impact_class}">{impact} Impact</span>
                    </div>
                    <div class="param-values">
                        <strong>Values tested:</strong> {values}
                    </div>
                    <div class="param-explanation">
                        {explanation}
                    </div>
                </div>
            """
        
        content += """
            </div>
        </div>
        """
    
    # Add image gallery
    content += """
        <div class="section">
            <h3>📈 Visualization Analysis</h3>
            <p>Visual analysis of hyperparameter effects on model performance.</p>
            <div class="image-gallery" id="hp-images">
    """
    
    for img_file in hp_data.get('images', []):
        img_path = f"../reports/hyperparameter_tuning/{img_file}"
        if os.path.exists(img_path):
            img_base64 = encode_image_to_base64(img_path)
            if img_base64:
                caption = img_file.replace('_', ' ').replace('.png', '').title()
                content += f"""
                <div class="image-card">
                    <img src="data:image/png;base64,{img_base64}" alt="{img_file}">
                    <div class="image-caption">{caption}</div>
                </div>
                """
    
    content += """
            </div>
        </div>
    """
    
    # Full results table
    content += """
        <div class="section">
            <h3>📋 Full Results Table</h3>
            <p>All configurations tested, sorted by RMSE (best first). Top performers highlighted.</p>
            <div class="table-container">
                <table class="results-table hp-results-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Algorithm</th>
                            <th>RMSE</th>
                            <th>MAE</th>
                            <th>NDCG@10</th>
                            <th>Precision@10</th>
                            <th>Coverage</th>
                            <th>Time (s)</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    # Sort by RMSE and show top 50
    sorted_results = results_df.sort_values('RMSE').head(50)
    
    for rank, (_, row) in enumerate(sorted_results.iterrows(), 1):
        row_class = 'best-row' if rank <= 3 else ''
        content += f"""
                        <tr class="{row_class}">
                            <td>{rank}</td>
                            <td>{row['algorithm']}</td>
                            <td>{row['RMSE']:.4f}</td>
                            <td>{row['MAE']:.4f}</td>
                            <td>{row['NDCG@10']:.4f}</td>
                            <td>{row.get('Precision@10', 0):.4f}</td>
                            <td>{row['Coverage']:.2%}</td>
                            <td>{row.get('train_time', 'N/A')}</td>
                        </tr>
        """
    
    content += """
                    </tbody>
                </table>
            </div>
            <p class="table-note">Showing top 50 of {total} configurations sorted by RMSE. Green rows indicate top 3 performers.</p>
        </div>
    </div>
    """.format(total=len(results_df))
    
    return content


def get_algorithm_explanations():
    """Detailed explanations for each algorithm"""
    return {
        'collaborative_filtering': {
            'name': 'Collaborative Filtering',
            'tagline': '"People who liked what you liked also liked..."',
            'difficulty': 'Beginner',
            'overview': '''
Collaborative Filtering is one of the oldest and most intuitive recommendation techniques. 
It works on a simple principle: if two users agreed in the past, they will agree in the future.
No knowledge about items is needed - only the interaction patterns between users and items.
            ''',
            'how_it_works': '''
<strong>User-Based CF (Find Similar Users):</strong>
1. Build a matrix of all user ratings
2. For target user, find K most similar users (neighbors)
3. Aggregate neighbors' ratings to predict unrated items
4. Recommend top-rated items user hasn't seen

<strong>Item-Based CF (Find Similar Items):</strong>
1. Build item-item similarity matrix
2. For each item user rated, find similar items
3. Predict rating = weighted average of similar items' ratings
4. Recommend highest predicted items
            ''',
            'similarity_formula': '''
<strong>Cosine Similarity:</strong>
sim(u,v) = (u · v) / (||u|| × ||v||)

<strong>Pearson Correlation:</strong>
sim(u,v) = Σ(r_ui - r̄_u)(r_vi - r̄_v) / √[Σ(r_ui - r̄_u)² × Σ(r_vi - r̄_v)²]
            ''',
            'pros': [
                'No content analysis needed',
                'Can discover unexpected recommendations',
                'Works across domains',
                'Easy to explain recommendations'
            ],
            'cons': [
                'Cold-start problem (new users/items)',
                'Scalability issues with large datasets',
                'Sparsity problem',
                'Popularity bias'
            ],
            'best_for': 'Dense rating datasets with many active users',
            'svg_diagram': '''
<svg viewBox="0 0 400 250" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arrow1" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#3498db"/>
    </marker>
  </defs>
  
  <!-- Title -->
  <text x="200" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#2c3e50">User-Based Collaborative Filtering</text>
  
  <!-- Users -->
  <circle cx="50" cy="80" r="25" fill="#3498db"/>
  <text x="50" y="85" text-anchor="middle" fill="white" font-size="12">You</text>
  
  <circle cx="50" cy="150" r="20" fill="#9b59b6"/>
  <text x="50" y="155" text-anchor="middle" fill="white" font-size="10">User A</text>
  
  <circle cx="50" cy="210" r="20" fill="#9b59b6"/>
  <text x="50" y="215" text-anchor="middle" fill="white" font-size="10">User B</text>
  
  <!-- Similarity arrows -->
  <line x1="75" y1="90" x2="75" y2="140" stroke="#3498db" stroke-width="2" stroke-dasharray="5,5"/>
  <text x="85" y="115" font-size="9" fill="#3498db">87% similar</text>
  
  <line x1="75" y1="160" x2="75" y2="200" stroke="#e74c3c" stroke-width="1" stroke-dasharray="5,5"/>
  <text x="85" y="180" font-size="9" fill="#e74c3c">23% similar</text>
  
  <!-- Items rated by similar user -->
  <rect x="150" y="60" width="60" height="40" rx="5" fill="#2ecc71"/>
  <text x="180" y="85" text-anchor="middle" fill="white" font-size="10">Movie A</text>
  <text x="180" y="70" text-anchor="middle" fill="white" font-size="8">★★★★★</text>
  
  <rect x="150" y="130" width="60" height="40" rx="5" fill="#2ecc71"/>
  <text x="180" y="155" text-anchor="middle" fill="white" font-size="10">Movie B</text>
  <text x="180" y="140" text-anchor="middle" fill="white" font-size="8">★★★★☆</text>
  
  <!-- Arrow from similar user to items -->
  <path d="M70,150 Q120,100 145,80" stroke="#9b59b6" stroke-width="2" fill="none" marker-end="url(#arrow1)"/>
  <path d="M70,150 Q120,150 145,150" stroke="#9b59b6" stroke-width="2" fill="none" marker-end="url(#arrow1)"/>
  
  <!-- Recommendation -->
  <rect x="280" y="95" width="100" height="60" rx="8" fill="#e74c3c"/>
  <text x="330" y="120" text-anchor="middle" fill="white" font-size="11" font-weight="bold">Recommend</text>
  <text x="330" y="140" text-anchor="middle" fill="white" font-size="10">Movie A to You!</text>
  
  <!-- Arrow to recommendation -->
  <path d="M215,80 L275,110" stroke="#2c3e50" stroke-width="2" marker-end="url(#arrow1)"/>
</svg>
            '''
        },
        
        'matrix_factorization': {
            'name': 'Matrix Factorization',
            'tagline': '"Discover hidden patterns in user preferences"',
            'difficulty': 'Intermediate',
            'overview': '''
Matrix Factorization decomposes the sparse user-item rating matrix into two lower-dimensional matrices.
Each user and item is represented as a vector of "latent factors" - hidden characteristics learned from data.
These factors might represent genres, mood, quality, etc., but they're discovered automatically.
            ''',
            'how_it_works': '''
<strong>The Big Idea:</strong>
Rating Matrix R (m×n) ≈ User Matrix U (m×k) × Item Matrix V (k×n)

<strong>Steps:</strong>
1. Initialize random user vectors (U) and item vectors (V)
2. For each known rating r_ui:
   - Predict: r̂_ui = u_i · v_j (dot product)
   - Calculate error: e_ui = r_ui - r̂_ui
3. Update vectors to minimize error:
   - u_i = u_i + α(e_ui × v_j - λ × u_i)
   - v_j = v_j + α(e_ui × u_i - λ × v_j)
4. Repeat until convergence
5. Predict missing ratings using r̂_ui = u_i · v_j
            ''',
            'methods': '''
<strong>SVD (Singular Value Decomposition):</strong>
Direct matrix decomposition. Fast but requires handling missing values.

<strong>ALS (Alternating Least Squares):</strong>
Alternately fix U, optimize V, then fix V, optimize U. Good for implicit feedback.

<strong>Funk SVD (SGD-based):</strong>
Stochastic gradient descent on observed ratings only. Most common in practice.
            ''',
            'pros': [
                'Handles sparsity well',
                'Captures latent patterns',
                'Scalable with SGD',
                'Better accuracy than CF'
            ],
            'cons': [
                'Cold-start problem remains',
                'Latent factors not interpretable',
                'Requires careful hyperparameter tuning',
                'Can overfit on sparse data'
            ],
            'best_for': 'Large sparse matrices with explicit ratings',
            'svg_diagram': '''
<svg viewBox="0 0 450 200" xmlns="http://www.w3.org/2000/svg">
  <text x="225" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#2c3e50">Matrix Factorization</text>
  
  <!-- Original Matrix R -->
  <rect x="10" y="40" width="100" height="80" fill="#ecf0f1" stroke="#2c3e50" stroke-width="2"/>
  <text x="60" y="55" text-anchor="middle" font-size="10" fill="#2c3e50">Rating Matrix R</text>
  <text x="60" y="75" text-anchor="middle" font-size="20" fill="#2c3e50">?</text>
  <text x="60" y="95" text-anchor="middle" font-size="8" fill="#7f8c8d">(m users × n items)</text>
  <text x="60" y="110" text-anchor="middle" font-size="8" fill="#e74c3c">95% missing!</text>
  
  <!-- Equals sign -->
  <text x="135" y="85" text-anchor="middle" font-size="24" fill="#2c3e50">≈</text>
  
  <!-- User Matrix U -->
  <rect x="160" y="40" width="50" height="80" fill="#3498db" stroke="#2c3e50" stroke-width="2"/>
  <text x="185" y="55" text-anchor="middle" font-size="10" fill="white">U</text>
  <text x="185" y="75" text-anchor="middle" font-size="8" fill="white">User</text>
  <text x="185" y="87" text-anchor="middle" font-size="8" fill="white">Factors</text>
  <text x="185" y="110" text-anchor="middle" font-size="7" fill="white">(m × k)</text>
  
  <!-- Times sign -->
  <text x="230" y="85" text-anchor="middle" font-size="24" fill="#2c3e50">×</text>
  
  <!-- Item Matrix V -->
  <rect x="250" y="50" width="80" height="50" fill="#2ecc71" stroke="#2c3e50" stroke-width="2"/>
  <text x="290" y="65" text-anchor="middle" font-size="10" fill="white">V</text>
  <text x="290" y="80" text-anchor="middle" font-size="8" fill="white">Item Factors</text>
  <text x="290" y="95" text-anchor="middle" font-size="7" fill="white">(k × n)</text>
  
  <!-- Latent factors explanation -->
  <rect x="350" y="40" width="90" height="80" fill="#f8f9fa" stroke="#9b59b6" stroke-width="2" rx="5"/>
  <text x="395" y="55" text-anchor="middle" font-size="9" font-weight="bold" fill="#9b59b6">Latent Factors</text>
  <text x="395" y="70" text-anchor="middle" font-size="8" fill="#2c3e50">k = 20-100</text>
  <text x="395" y="85" text-anchor="middle" font-size="7" fill="#7f8c8d">• Action-ness</text>
  <text x="395" y="97" text-anchor="middle" font-size="7" fill="#7f8c8d">• Romance level</text>
  <text x="395" y="109" text-anchor="middle" font-size="7" fill="#7f8c8d">• Complexity</text>
  
  <!-- Prediction formula -->
  <rect x="100" y="140" width="250" height="50" fill="#fff3cd" stroke="#f39c12" stroke-width="2" rx="5"/>
  <text x="225" y="160" text-anchor="middle" font-size="11" font-weight="bold" fill="#2c3e50">Prediction: r̂ᵤᵢ = uᵤ · vᵢ</text>
  <text x="225" y="180" text-anchor="middle" font-size="9" fill="#7f8c8d">Dot product of user and item vectors</text>
</svg>
            '''
        },
        
        'content_based': {
            'name': 'Content-Based Filtering',
            'tagline': '"If you liked X, you\'ll like similar items"',
            'difficulty': 'Beginner',
            'overview': '''
Content-Based Filtering recommends items similar to what the user liked before, based on item features.
Unlike Collaborative Filtering, it analyzes the actual content/attributes of items.
It builds a "user profile" from items they liked, then finds matching items.
            ''',
            'how_it_works': '''
<strong>Steps:</strong>
1. <strong>Extract Features</strong>: Convert item attributes to numerical vectors
   - Text → TF-IDF (Term Frequency-Inverse Document Frequency)
   - Categories → One-hot encoding
   - Tags → Multi-label encoding

2. <strong>Build User Profile</strong>: 
   - Aggregate features of items user liked
   - Weight by rating: profile = Σ(rating × item_features) / Σ(ratings)

3. <strong>Find Similar Items</strong>:
   - Calculate cosine similarity between user profile and all items
   - Rank items by similarity score

4. <strong>Recommend</strong>:
   - Return top-N items user hasn't interacted with
            ''',
            'tfidf_explanation': '''
<strong>TF-IDF (Term Frequency-Inverse Document Frequency):</strong>

TF(t,d) = count(t in d) / total_words(d)
IDF(t) = log(N / docs_containing(t))
TF-IDF = TF × IDF

High TF-IDF = Important term for distinguishing this item
            ''',
            'pros': [
                'No cold-start for items (if features known)',
                'Transparent recommendations',
                'No need for other users\' data',
                'Captures niche preferences'
            ],
            'cons': [
                'Limited to item features',
                'Cannot recommend outside user\'s profile',
                'Requires good feature engineering',
                'User cold-start still exists'
            ],
            'best_for': 'Items with rich metadata (articles, products, movies with descriptions)',
            'svg_diagram': '''
<svg viewBox="0 0 450 220" xmlns="http://www.w3.org/2000/svg">
  <text x="225" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#2c3e50">Content-Based Filtering</text>
  
  <!-- User's liked items -->
  <rect x="10" y="40" width="120" height="100" fill="#e8f4f8" stroke="#3498db" stroke-width="2" rx="5"/>
  <text x="70" y="55" text-anchor="middle" font-size="10" font-weight="bold" fill="#3498db">User's History</text>
  
  <rect x="20" y="65" width="100" height="25" fill="#3498db" rx="3"/>
  <text x="70" y="82" text-anchor="middle" fill="white" font-size="8">Action, Sci-Fi ★★★★★</text>
  
  <rect x="20" y="95" width="100" height="25" fill="#3498db" rx="3"/>
  <text x="70" y="112" text-anchor="middle" fill="white" font-size="8">Action, Drama ★★★★☆</text>
  
  <!-- Arrow -->
  <path d="M135,90 L175,90" stroke="#2c3e50" stroke-width="2" marker-end="url(#arrow1)"/>
  
  <!-- User Profile -->
  <rect x="180" y="50" width="100" height="80" fill="#9b59b6" stroke="#2c3e50" stroke-width="2" rx="5"/>
  <text x="230" y="68" text-anchor="middle" fill="white" font-size="10" font-weight="bold">User Profile</text>
  <text x="230" y="85" text-anchor="middle" fill="white" font-size="8">Action: 0.8</text>
  <text x="230" y="98" text-anchor="middle" fill="white" font-size="8">Sci-Fi: 0.5</text>
  <text x="230" y="111" text-anchor="middle" fill="white" font-size="8">Drama: 0.3</text>
  <text x="230" y="124" text-anchor="middle" fill="white" font-size="8">Romance: 0.1</text>
  
  <!-- Arrow -->
  <path d="M285,90 L325,90" stroke="#2c3e50" stroke-width="2" marker-end="url(#arrow1)"/>
  <text x="305" y="80" text-anchor="middle" font-size="8" fill="#7f8c8d">cosine</text>
  <text x="305" y="100" text-anchor="middle" font-size="8" fill="#7f8c8d">similarity</text>
  
  <!-- Candidate Items -->
  <rect x="330" y="40" width="110" height="120" fill="#f8f9fa" stroke="#2ecc71" stroke-width="2" rx="5"/>
  <text x="385" y="55" text-anchor="middle" font-size="10" font-weight="bold" fill="#2ecc71">Find Similar</text>
  
  <rect x="340" y="65" width="90" height="25" fill="#2ecc71" rx="3"/>
  <text x="385" y="82" text-anchor="middle" fill="white" font-size="8">Movie X (92%)</text>
  
  <rect x="340" y="95" width="90" height="25" fill="#27ae60" rx="3"/>
  <text x="385" y="112" text-anchor="middle" fill="white" font-size="8">Movie Y (87%)</text>
  
  <rect x="340" y="125" width="90" height="25" fill="#1e8449" rx="3"/>
  <text x="385" y="142" text-anchor="middle" fill="white" font-size="8">Movie Z (73%)</text>
  
  <!-- Result -->
  <rect x="150" y="170" width="150" height="40" fill="#e74c3c" rx="5"/>
  <text x="225" y="195" text-anchor="middle" fill="white" font-size="11" font-weight="bold">Recommend Movie X!</text>
</svg>
            '''
        },
        
        'graph_based': {
            'name': 'Graph-Based Methods',
            'tagline': '"Navigate the web of user-item relationships"',
            'difficulty': 'Advanced',
            'overview': '''
Graph-Based methods model the recommendation problem as a graph where users and items are nodes,
and interactions (ratings, clicks) are edges. Recommendations come from traversing this graph.
            ''',
            'how_it_works': '''
<strong>Graph Construction:</strong>
- Nodes: Users (one type) + Items (another type) = Bipartite graph
- Edges: User-Item interactions, weighted by rating/frequency
- Result: Network showing who interacted with what

<strong>Personalized PageRank:</strong>
1. Start random walk from target user
2. At each step: 
   - With prob α: follow a random edge
   - With prob (1-α): jump back to start user
3. Stationary distribution = item relevance scores
4. Recommend items with highest scores

<strong>Random Walk with Restart:</strong>
1. Perform many random walks from user
2. Count how often each item is visited
3. More visits = higher relevance
            ''',
            'pagerank_formula': '''
<strong>PageRank Equation:</strong>
PR(v) = (1-α)/N + α × Σ PR(u)/out_degree(u)

Where:
- α = damping factor (typically 0.85)
- N = total number of nodes
- Sum over all nodes u linking to v
            ''',
            'pros': [
                'Captures complex relationships',
                'Can explain paths (why recommended)',
                'Handles implicit feedback well',
                'Serendipitous discoveries'
            ],
            'cons': [
                'Computationally expensive',
                'Graph can get very large',
                'Cold-start for new nodes',
                'Parameter tuning (damping factor)'
            ],
            'best_for': 'Social networks, implicit feedback, path explanations',
            'svg_diagram': '''
<svg viewBox="0 0 450 220" xmlns="http://www.w3.org/2000/svg">
  <text x="225" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#2c3e50">Graph-Based (PageRank)</text>
  
  <!-- User nodes -->
  <circle cx="60" cy="80" r="22" fill="#3498db" stroke="#2c3e50" stroke-width="2"/>
  <text x="60" y="85" text-anchor="middle" fill="white" font-size="10">You</text>
  
  <circle cx="60" cy="160" r="18" fill="#5dade2"/>
  <text x="60" y="165" text-anchor="middle" fill="white" font-size="9">U1</text>
  
  <circle cx="130" cy="180" r="18" fill="#5dade2"/>
  <text x="130" y="185" text-anchor="middle" fill="white" font-size="9">U2</text>
  
  <!-- Item nodes -->
  <rect x="180" y="50" width="50" height="35" rx="5" fill="#2ecc71"/>
  <text x="205" y="72" text-anchor="middle" fill="white" font-size="9">Item A</text>
  
  <rect x="180" y="110" width="50" height="35" rx="5" fill="#2ecc71"/>
  <text x="205" y="132" text-anchor="middle" fill="white" font-size="9">Item B</text>
  
  <rect x="180" y="170" width="50" height="35" rx="5" fill="#2ecc71"/>
  <text x="205" y="192" text-anchor="middle" fill="white" font-size="9">Item C</text>
  
  <rect x="270" y="80" width="50" height="35" rx="5" fill="#27ae60"/>
  <text x="295" y="102" text-anchor="middle" fill="white" font-size="9">Item D</text>
  
  <!-- Edges (interactions) -->
  <line x1="82" y1="75" x2="175" y2="67" stroke="#3498db" stroke-width="2"/>
  <line x1="82" y1="85" x2="175" y2="125" stroke="#3498db" stroke-width="2"/>
  <line x1="78" y1="155" x2="175" y2="125" stroke="#5dade2" stroke-width="1.5"/>
  <line x1="78" y1="165" x2="175" y2="185" stroke="#5dade2" stroke-width="1.5"/>
  <line x1="148" y1="175" x2="175" y2="185" stroke="#5dade2" stroke-width="1.5"/>
  <line x1="235" y1="125" x2="265" y2="100" stroke="#2ecc71" stroke-width="1.5"/>
  <line x1="235" y1="185" x2="265" y2="105" stroke="#2ecc71" stroke-width="1.5"/>
  
  <!-- Random walk visualization -->
  <path d="M85,80 Q140,50 175,67" stroke="#e74c3c" stroke-width="3" fill="none" stroke-dasharray="5,3"/>
  <path d="M225,67 Q250,75 265,95" stroke="#e74c3c" stroke-width="3" fill="none" stroke-dasharray="5,3"/>
  
  <!-- PageRank scores -->
  <rect x="350" y="50" width="90" height="100" fill="#f8f9fa" stroke="#9b59b6" stroke-width="2" rx="5"/>
  <text x="395" y="68" text-anchor="middle" font-size="10" font-weight="bold" fill="#9b59b6">PageRank</text>
  <text x="395" y="85" text-anchor="middle" font-size="9" fill="#2c3e50">Item D: 0.32</text>
  <text x="395" y="100" text-anchor="middle" font-size="9" fill="#2c3e50">Item A: 0.28</text>
  <text x="395" y="115" text-anchor="middle" font-size="9" fill="#2c3e50">Item B: 0.24</text>
  <text x="395" y="130" text-anchor="middle" font-size="9" fill="#2c3e50">Item C: 0.16</text>
  
  <!-- Legend -->
  <text x="380" y="170" text-anchor="middle" font-size="8" fill="#e74c3c">→ Random Walk Path</text>
  <text x="380" y="185" text-anchor="middle" font-size="8" fill="#7f8c8d">α = 0.85 (damping)</text>
  
  <!-- Result -->
  <rect x="340" y="195" width="100" height="25" fill="#e74c3c" rx="3"/>
  <text x="390" y="212" text-anchor="middle" fill="white" font-size="9" font-weight="bold">Recommend Item D</text>
</svg>
            '''
        },
        
        'popularity': {
            'name': 'Popularity Baseline',
            'tagline': '"What\'s trending? What do most people like?"',
            'difficulty': 'Beginner',
            'overview': '''
Popularity-based methods are the simplest recommendation approach: recommend what's popular.
Despite their simplicity, they're surprisingly effective baselines and solve the cold-start problem.
They're often used as fallback when personalized methods fail.
            ''',
            'how_it_works': '''
<strong>Methods:</strong>

1. <strong>Count-Based</strong>: Recommend most-rated items
   Score = count(ratings for item)

2. <strong>Rating-Based</strong>: Recommend highest-rated items
   Score = average(ratings for item)

3. <strong>Weighted</strong>: Combine popularity and rating
   Score = w × normalized_count + (1-w) × normalized_rating

4. <strong>Bayesian Average (IMDB-style)</strong>:
   WR = (v/(v+m)) × R + (m/(v+m)) × C
   Where:
   - v = number of votes for item
   - m = minimum votes required
   - R = average rating of item
   - C = mean rating across all items
            ''',
            'pros': [
                'Extremely simple to implement',
                'No training required',
                'Solves cold-start for new users',
                'Good baseline benchmark',
                'Fast inference'
            ],
            'cons': [
                'Not personalized at all',
                'Reinforces popularity bias',
                'Same recommendations for everyone',
                'Misses niche preferences'
            ],
            'best_for': 'New user onboarding, baseline comparison, fallback system',
            'svg_diagram': '''
<svg viewBox="0 0 400 200" xmlns="http://www.w3.org/2000/svg">
  <text x="200" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#2c3e50">Popularity Baseline</text>
  
  <!-- Bar chart of popularity -->
  <rect x="50" y="45" width="120" height="20" fill="#3498db"/>
  <text x="180" y="60" font-size="10" fill="#2c3e50">Movie A - 10,000 ratings</text>
  
  <rect x="50" y="70" width="90" height="20" fill="#5dade2"/>
  <text x="150" y="85" font-size="10" fill="#2c3e50">Movie B - 7,500 ratings</text>
  
  <rect x="50" y="95" width="60" height="20" fill="#85c1e9"/>
  <text x="120" y="110" font-size="10" fill="#2c3e50">Movie C - 5,000 ratings</text>
  
  <rect x="50" y="120" width="30" height="20" fill="#aed6f1"/>
  <text x="90" y="135" font-size="10" fill="#2c3e50">Movie D - 2,500 ratings</text>
  
  <!-- Rating stars -->
  <text x="320" y="60" font-size="10" fill="#f39c12">★★★★☆ (4.2)</text>
  <text x="320" y="85" font-size="10" fill="#f39c12">★★★★★ (4.8)</text>
  <text x="320" y="110" font-size="10" fill="#f39c12">★★★☆☆ (3.5)</text>
  <text x="320" y="135" font-size="10" fill="#f39c12">★★★★☆ (4.0)</text>
  
  <!-- Formula box -->
  <rect x="80" y="155" width="240" height="35" fill="#fff3cd" stroke="#f39c12" stroke-width="2" rx="5"/>
  <text x="200" y="170" text-anchor="middle" font-size="9" font-weight="bold" fill="#2c3e50">Bayesian: WR = (v/(v+m))×R + (m/(v+m))×C</text>
  <text x="200" y="183" text-anchor="middle" font-size="8" fill="#7f8c8d">Balances rating quality with popularity</text>
</svg>
            '''
        },
        
        'neural_cf': {
            'name': 'Neural Collaborative Filtering',
            'tagline': '"Deep learning meets recommendations"',
            'difficulty': 'Advanced',
            'overview': '''
Neural CF uses deep neural networks to learn user-item interactions.
Instead of hand-crafted similarity functions, it learns the interaction function from data.
Can capture complex non-linear patterns that traditional methods miss.
            ''',
            'how_it_works': '''
<strong>Architecture Components:</strong>

1. <strong>GMF (Generalized Matrix Factorization)</strong>:
   - Element-wise product of user/item embeddings
   - Output = σ(h × (u ⊙ v))
   - Linear interaction modeling

2. <strong>MLP (Multi-Layer Perceptron)</strong>:
   - Concatenate user/item embeddings
   - Pass through hidden layers with ReLU
   - Learns non-linear interactions

3. <strong>NeuMF (Neural Matrix Factorization)</strong>:
   - Combines GMF + MLP
   - GMF: Linear patterns
   - MLP: Non-linear patterns
   - Final: Concatenate and predict
            ''',
            'pros': [
                'Learns complex patterns',
                'State-of-the-art accuracy',
                'Flexible architecture',
                'Can incorporate side information'
            ],
            'cons': [
                'Requires lots of data',
                'Computationally expensive',
                'Black box (not interpretable)',
                'Needs careful tuning'
            ],
            'best_for': 'Large-scale systems with rich data and compute resources',
            'svg_diagram': '''
<svg viewBox="0 0 450 220" xmlns="http://www.w3.org/2000/svg">
  <text x="225" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#2c3e50">Neural Collaborative Filtering (NeuMF)</text>
  
  <!-- User/Item inputs -->
  <rect x="20" y="50" width="60" height="30" fill="#3498db" rx="3"/>
  <text x="50" y="70" text-anchor="middle" fill="white" font-size="9">User ID</text>
  
  <rect x="20" y="100" width="60" height="30" fill="#2ecc71" rx="3"/>
  <text x="50" y="120" text-anchor="middle" fill="white" font-size="9">Item ID</text>
  
  <!-- Embedding layers -->
  <rect x="110" y="40" width="50" height="50" fill="#9b59b6" rx="3"/>
  <text x="135" y="55" text-anchor="middle" fill="white" font-size="7">GMF</text>
  <text x="135" y="68" text-anchor="middle" fill="white" font-size="7">User</text>
  <text x="135" y="80" text-anchor="middle" fill="white" font-size="7">Embed</text>
  
  <rect x="110" y="100" width="50" height="50" fill="#9b59b6" rx="3"/>
  <text x="135" y="115" text-anchor="middle" fill="white" font-size="7">GMF</text>
  <text x="135" y="128" text-anchor="middle" fill="white" font-size="7">Item</text>
  <text x="135" y="140" text-anchor="middle" fill="white" font-size="7">Embed</text>
  
  <rect x="170" y="40" width="50" height="50" fill="#e67e22" rx="3"/>
  <text x="195" y="55" text-anchor="middle" fill="white" font-size="7">MLP</text>
  <text x="195" y="68" text-anchor="middle" fill="white" font-size="7">User</text>
  <text x="195" y="80" text-anchor="middle" fill="white" font-size="7">Embed</text>
  
  <rect x="170" y="100" width="50" height="50" fill="#e67e22" rx="3"/>
  <text x="195" y="115" text-anchor="middle" fill="white" font-size="7">MLP</text>
  <text x="195" y="128" text-anchor="middle" fill="white" font-size="7">Item</text>
  <text x="195" y="140" text-anchor="middle" fill="white" font-size="7">Embed</text>
  
  <!-- GMF path -->
  <circle cx="270" cy="65" r="20" fill="#9b59b6"/>
  <text x="270" y="62" text-anchor="middle" fill="white" font-size="8">Element</text>
  <text x="270" y="72" text-anchor="middle" fill="white" font-size="8">Product</text>
  
  <!-- MLP path -->
  <rect x="250" y="110" width="40" height="25" fill="#e67e22" rx="3"/>
  <text x="270" y="127" text-anchor="middle" fill="white" font-size="8">MLP</text>
  
  <rect x="250" y="145" width="40" height="25" fill="#d35400" rx="3"/>
  <text x="270" y="162" text-anchor="middle" fill="white" font-size="8">MLP</text>
  
  <!-- Concatenation -->
  <rect x="320" y="90" width="50" height="40" fill="#2c3e50" rx="3"/>
  <text x="345" y="105" text-anchor="middle" fill="white" font-size="8">Concat</text>
  <text x="345" y="120" text-anchor="middle" fill="white" font-size="8">Layer</text>
  
  <!-- Output -->
  <circle cx="410" cy="110" r="25" fill="#e74c3c"/>
  <text x="410" y="107" text-anchor="middle" fill="white" font-size="9">ŷᵤᵢ</text>
  <text x="410" y="120" text-anchor="middle" fill="white" font-size="7">Prediction</text>
  
  <!-- Connections -->
  <line x1="85" y1="65" x2="105" y2="65" stroke="#2c3e50" stroke-width="1.5"/>
  <line x1="85" y1="115" x2="105" y2="125" stroke="#2c3e50" stroke-width="1.5"/>
  <line x1="165" y1="65" x2="250" y2="65" stroke="#9b59b6" stroke-width="1.5"/>
  <line x1="165" y1="125" x2="250" y2="65" stroke="#9b59b6" stroke-width="1.5"/>
  <line x1="225" y1="65" x2="250" y2="122" stroke="#e67e22" stroke-width="1.5"/>
  <line x1="225" y1="125" x2="250" y2="122" stroke="#e67e22" stroke-width="1.5"/>
  <line x1="290" y1="65" x2="315" y2="100" stroke="#9b59b6" stroke-width="1.5"/>
  <line x1="290" y1="157" x2="315" y2="120" stroke="#e67e22" stroke-width="1.5"/>
  <line x1="375" y1="110" x2="385" y2="110" stroke="#2c3e50" stroke-width="2"/>
</svg>
            '''
        }
    }


def generate_dashboard():
    """Generate the enhanced HTML dashboard"""
    
    print("🎨 Generating Enhanced Dashboard...")
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Load data
    viz_files = []
    viz_dir = "../visualizations"
    if os.path.exists(viz_dir):
        for f in os.listdir(viz_dir):
            if f.endswith('.html') and f != 'dashboard.html':
                viz_files.append(f)
    
    hp_data = {'results': None, 'summary': None, 'images': []}
    hp_csv = "../reports/hyperparameter_tuning/all_hyperparameter_results.csv"
    hp_json = "../reports/hyperparameter_tuning/hyperparameter_summary.json"
    if os.path.exists(hp_csv):
        hp_data['results'] = pd.read_csv(hp_csv)
    if os.path.exists(hp_json):
        hp_data['summary'] = load_json(hp_json)
    if os.path.exists("../reports/hyperparameter_tuning"):
        hp_data['images'] = [f for f in os.listdir("../reports/hyperparameter_tuning") if f.endswith('.png')]
    
    # Load multi-dataset results
    multi_data = {'results': None, 'summary': None, 'images': []}
    multi_csv = "../reports/multi_dataset/multi_dataset_results.csv"
    multi_json = "../reports/multi_dataset/multi_dataset_summary.json"
    if os.path.exists(multi_csv):
        multi_data['results'] = pd.read_csv(multi_csv)
    if os.path.exists(multi_json):
        multi_data['summary'] = load_json(multi_json)
    if os.path.exists("../reports/multi_dataset"):
        multi_data['images'] = [f for f in os.listdir("../reports/multi_dataset") if f.endswith('.png')]
    
    algo_info = get_algorithm_explanations()
    
    # Build HTML
    html = generate_html_content(viz_files, hp_data, multi_data, algo_info)
    
    output_path = "../visualizations/dashboard.html"
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"   ✅ Dashboard saved to: {output_path}")
    return output_path


def generate_html_content(viz_files, hp_data, multi_data, algo_info):
    """Generate the full HTML content"""
    
    # Get multi-dataset table HTML
    multi_table_html = ""
    if multi_data['results'] is not None:
        df = multi_data['results']
        multi_table_html = "<table class='results-table'><thead><tr>"
        for col in ['dataset', 'algorithm', 'RMSE', 'NDCG@10', 'Coverage', 'train_time']:
            multi_table_html += f"<th>{col}</th>"
        multi_table_html += "</tr></thead><tbody>"
        for _, row in df.sort_values(['dataset', 'RMSE']).iterrows():
            multi_table_html += "<tr>"
            for col in ['dataset', 'algorithm', 'RMSE', 'NDCG@10', 'Coverage', 'train_time']:
                val = row.get(col, 'N/A')
                if isinstance(val, float):
                    multi_table_html += f"<td>{val:.4f}</td>"
                else:
                    multi_table_html += f"<td>{val}</td>"
            multi_table_html += "</tr>"
        multi_table_html += "</tbody></table>"
    
    # Get multi-dataset images
    multi_images_html = ""
    for img in multi_data.get('images', []):
        img_path = f"../reports/multi_dataset/{img}"
        if os.path.exists(img_path):
            b64 = encode_image_to_base64(img_path)
            if b64:
                multi_images_html += f'''
                <div class="image-card">
                    <img src="data:image/png;base64,{b64}" alt="{img}">
                    <div class="image-caption">{img.replace('_', ' ').replace('.png', '').title()}</div>
                </div>
                '''
    
    # Get hyperparameter images
    hp_images_html = ""
    for img in hp_data.get('images', []):
        img_path = f"../reports/hyperparameter_tuning/{img}"
        if os.path.exists(img_path):
            b64 = encode_image_to_base64(img_path)
            if b64:
                hp_images_html += f'''
                <div class="image-card">
                    <img src="data:image/png;base64,{b64}" alt="{img}">
                    <div class="image-caption">{img.replace('_', ' ').replace('.png', '').title()}</div>
                </div>
                '''
    
    # Generate comprehensive hyperparameter tab content
    hp_tab_content = generate_hyperparameter_tab_content(hp_data)
    
    # Build algorithm cards HTML
    algo_cards_html = ""
    for key, info in algo_info.items():
        pros_html = "".join([f"<li>{p}</li>" for p in info.get('pros', [])])
        cons_html = "".join([f"<li>{c}</li>" for c in info.get('cons', [])])
        
        algo_cards_html += f'''
        <div class="algo-card" id="algo-{key}">
            <div class="algo-header">
                <h3>{info['name']}</h3>
                <span class="difficulty difficulty-{info['difficulty'].lower()}">{info['difficulty']}</span>
            </div>
            <p class="tagline">{info['tagline']}</p>
            
            <div class="algo-diagram">
                {info.get('svg_diagram', '')}
            </div>
            
            <div class="algo-section">
                <h4>📖 Overview</h4>
                <p>{info['overview']}</p>
            </div>
            
            <div class="algo-section">
                <h4>⚙️ How It Works</h4>
                <div class="how-it-works">{info['how_it_works']}</div>
            </div>
            
            <div class="algo-section pros-cons">
                <div class="pros">
                    <h4>✅ Pros</h4>
                    <ul>{pros_html}</ul>
                </div>
                <div class="cons">
                    <h4>❌ Cons</h4>
                    <ul>{cons_html}</ul>
                </div>
            </div>
            
            <div class="algo-section">
                <h4>🎯 Best For</h4>
                <p class="best-for">{info.get('best_for', '')}</p>
            </div>
        </div>
        '''
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Recommendation Systems - Complete Guide & Analysis</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f6fa; color: #2c3e50; line-height: 1.6; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; box-shadow: 0 0 50px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; text-align: center; }}
        .header h1 {{ font-size: 2.8em; margin-bottom: 10px; }}
        .header p {{ font-size: 1.2em; opacity: 0.9; }}
        
        /* Tabs */
        .tabs {{ display: flex; background: #2c3e50; flex-wrap: wrap; }}
        .tab {{ padding: 15px 25px; cursor: pointer; color: #bdc3c7; font-weight: 500; border: none; background: none; transition: all 0.3s; }}
        .tab:hover {{ background: #34495e; color: white; }}
        .tab.active {{ background: #3498db; color: white; }}
        .tab-content {{ display: none; padding: 40px; }}
        .tab-content.active {{ display: block; }}
        
        /* Stats */
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 20px; margin-bottom: 40px; }}
        .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 12px; text-align: center; }}
        .stat-number {{ font-size: 2.5em; font-weight: bold; }}
        .stat-label {{ opacity: 0.9; }}
        
        /* Algorithm Cards */
        .algo-card {{ background: #f8f9fa; border-radius: 16px; padding: 30px; margin-bottom: 30px; border-left: 5px solid #3498db; }}
        .algo-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }}
        .algo-header h3 {{ font-size: 1.8em; color: #2c3e50; }}
        .tagline {{ font-style: italic; color: #7f8c8d; font-size: 1.1em; margin-bottom: 20px; }}
        .difficulty {{ padding: 5px 15px; border-radius: 20px; font-size: 0.85em; font-weight: 500; }}
        .difficulty-beginner {{ background: #d4edda; color: #155724; }}
        .difficulty-intermediate {{ background: #fff3cd; color: #856404; }}
        .difficulty-advanced {{ background: #f8d7da; color: #721c24; }}
        
        .algo-diagram {{ background: white; border-radius: 12px; padding: 20px; margin: 20px 0; text-align: center; }}
        .algo-diagram svg {{ max-width: 100%; height: auto; }}
        
        .algo-section {{ margin: 25px 0; }}
        .algo-section h4 {{ color: #2c3e50; margin-bottom: 10px; font-size: 1.2em; }}
        .algo-section p {{ color: #555; }}
        
        .how-it-works {{ background: white; padding: 20px; border-radius: 8px; white-space: pre-line; font-family: inherit; }}
        
        .pros-cons {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .pros, .cons {{ background: white; padding: 20px; border-radius: 8px; }}
        .pros {{ border-left: 4px solid #27ae60; }}
        .cons {{ border-left: 4px solid #e74c3c; }}
        .pros ul, .cons ul {{ margin-left: 20px; }}
        .pros li, .cons li {{ margin: 8px 0; }}
        
        .best-for {{ background: #e8f4f8; padding: 15px; border-radius: 8px; font-weight: 500; }}
        
        /* Dataset Cards */
        .dataset-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .dataset-card {{ background: white; border-radius: 12px; padding: 25px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .dataset-card h4 {{ color: #3498db; margin-bottom: 15px; }}
        .dataset-card .stats {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
        .dataset-card .stat {{ background: #f8f9fa; padding: 10px; border-radius: 5px; text-align: center; }}
        .dataset-card .stat-value {{ font-size: 1.3em; font-weight: bold; color: #2c3e50; }}
        .dataset-card .stat-name {{ font-size: 0.85em; color: #7f8c8d; }}
        
        /* Tables */
        .table-container {{ overflow-x: auto; margin: 20px 0; }}
        .results-table {{ width: 100%; border-collapse: collapse; }}
        .results-table th {{ background: #2c3e50; color: white; padding: 12px 15px; text-align: left; }}
        .results-table td {{ padding: 10px 15px; border-bottom: 1px solid #e9ecef; }}
        .results-table tr:hover {{ background: #f8f9fa; }}
        
        /* Images */
        .image-gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin: 30px 0; }}
        .image-card {{ background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .image-card img {{ width: 100%; height: auto; }}
        .image-caption {{ padding: 10px 15px; background: #f8f9fa; text-align: center; font-weight: 500; }}
        
        /* Viz links */
        .viz-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 20px; }}
        .viz-card {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); transition: transform 0.3s; }}
        .viz-card:hover {{ transform: translateY(-5px); }}
        .viz-card h4 {{ margin-bottom: 10px; }}
        .viz-card a {{ display: inline-block; padding: 10px 20px; background: #3498db; color: white; text-decoration: none; border-radius: 6px; margin-top: 10px; }}
        
        /* Section */
        .section {{ margin-bottom: 40px; }}
        .section h2 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; margin-bottom: 20px; }}
        .section h3 {{ color: #34495e; margin: 20px 0 15px 0; }}
        
        /* Hyperparameter-specific styles */
        .hp-container {{ max-width: 100%; }}
        .subtitle {{ color: #6c757d; margin-bottom: 25px; font-size: 1.1em; }}
        
        .methodology-box {{ background: #f8f9fa; padding: 25px; border-radius: 12px; border-left: 4px solid #3498db; margin-bottom: 30px; }}
        .methodology-box h4 {{ color: #2c3e50; margin: 15px 0 10px 0; }}
        .methodology-box ol, .methodology-box ul {{ margin-left: 20px; }}
        .methodology-box li {{ margin: 8px 0; line-height: 1.6; }}
        
        .algorithm-section {{ background: #f8f9fa; padding: 25px; border-radius: 12px; margin-bottom: 25px; }}
        .algo-description {{ color: #6c757d; margin-bottom: 20px; padding: 15px; background: white; border-radius: 8px; font-style: italic; }}
        
        .results-summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin-bottom: 20px; }}
        .best-config, .config-range {{ background: white; padding: 20px; border-radius: 8px; }}
        .best-config {{ border-left: 4px solid #27ae60; }}
        .config-range {{ border-left: 4px solid #3498db; }}
        
        .config-table {{ width: 100%; border-collapse: collapse; }}
        .config-table th, .config-table td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #e9ecef; }}
        .config-table th {{ background: #f8f9fa; font-weight: 600; }}
        
        .hyperparameters h4 {{ margin-bottom: 15px; }}
        .param-card {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #9b59b6; }}
        .param-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: wrap; gap: 10px; }}
        .param-name {{ font-weight: bold; font-size: 1.1em; color: #2c3e50; }}
        .param-impact {{ padding: 4px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 500; }}
        .impact-high {{ background: #fee2e2; color: #dc2626; }}
        .impact-medium {{ background: #fef3c7; color: #d97706; }}
        .impact-low {{ background: #d1fae5; color: #059669; }}
        .param-values {{ color: #6c757d; margin-bottom: 10px; font-family: monospace; background: #f8f9fa; padding: 8px 12px; border-radius: 4px; }}
        .param-explanation {{ color: #495057; line-height: 1.7; }}
        
        .hp-results-table tr.best-row {{ background: #d4edda !important; }}
        .hp-results-table tr.best-row:hover {{ background: #c3e6cb !important; }}
        
        .table-note {{ color: #6c757d; font-style: italic; margin-top: 10px; }}
        
        .alert {{ padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .alert-warning {{ background: #fff3cd; border: 1px solid #ffc107; color: #856404; }}
        .alert h4 {{ margin-bottom: 10px; }}
        
        .footer {{ text-align: center; padding: 30px; background: #2c3e50; color: white; }}
        
        @media (max-width: 768px) {{
            .tabs {{ flex-direction: column; }}
            .pros-cons {{ grid-template-columns: 1fr; }}
            .image-gallery {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Recommendation Systems</h1>
            <p>Complete Guide: Algorithms, Explanations, and Multi-Dataset Analysis</p>
            <p style="margin-top:10px; font-size:0.9em;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('overview')">📊 Overview</button>
            <button class="tab" onclick="showTab('algorithms')">🧠 Algorithm Guide</button>
            <button class="tab" onclick="showTab('datasets')">📁 Datasets</button>
            <button class="tab" onclick="showTab('results')">📈 Multi-Dataset Results</button>
            <button class="tab" onclick="showTab('hyperparameters')">🔬 Hyperparameters</button>
            <button class="tab" onclick="showTab('visualizations')">🎨 Visualizations</button>
        </div>
        
        <!-- Overview Tab -->
        <div id="overview" class="tab-content active">
            <div class="section">
                <h2>Project Overview</h2>
                <p>This project implements <strong>8 recommendation algorithms</strong> and evaluates them across <strong>3 different datasets</strong> with varying characteristics. Each algorithm is explained in detail with visual diagrams to help you understand how they work.</p>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">8</div>
                        <div class="stat-label">Algorithms</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">3</div>
                        <div class="stat-label">Datasets</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">24</div>
                        <div class="stat-label">Experiments</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">6</div>
                        <div class="stat-label">Metrics</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Quick Algorithm Comparison</h2>
                <table class="results-table">
                    <thead>
                        <tr>
                            <th>Algorithm</th>
                            <th>Type</th>
                            <th>Difficulty</th>
                            <th>Cold-Start?</th>
                            <th>Best For</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Collaborative Filtering</strong></td>
                            <td>Memory-based</td>
                            <td>Beginner</td>
                            <td>❌ Problem</td>
                            <td>Dense rating data</td>
                        </tr>
                        <tr>
                            <td><strong>Matrix Factorization</strong></td>
                            <td>Model-based</td>
                            <td>Intermediate</td>
                            <td>❌ Problem</td>
                            <td>Large sparse matrices</td>
                        </tr>
                        <tr>
                            <td><strong>Content-Based</strong></td>
                            <td>Feature-based</td>
                            <td>Beginner</td>
                            <td>✅ Items OK</td>
                            <td>Rich item metadata</td>
                        </tr>
                        <tr>
                            <td><strong>Graph-Based</strong></td>
                            <td>Network-based</td>
                            <td>Advanced</td>
                            <td>⚠️ Partial</td>
                            <td>Social networks</td>
                        </tr>
                        <tr>
                            <td><strong>Neural CF</strong></td>
                            <td>Deep Learning</td>
                            <td>Advanced</td>
                            <td>❌ Problem</td>
                            <td>Large-scale systems</td>
                        </tr>
                        <tr>
                            <td><strong>Popularity</strong></td>
                            <td>Baseline</td>
                            <td>Beginner</td>
                            <td>✅ Solved</td>
                            <td>New users, fallback</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- Algorithms Tab -->
        <div id="algorithms" class="tab-content">
            <div class="section">
                <h2>🧠 Algorithm Deep Dive</h2>
                <p>Click on each algorithm to learn how it works, see visual diagrams, and understand when to use it.</p>
                
                {algo_cards_html}
            </div>
        </div>
        
        <!-- Datasets Tab -->
        <div id="datasets" class="tab-content">
            <div class="section">
                <h2>📁 Datasets Used</h2>
                <p>We test all algorithms on <strong>3 different datasets</strong> to understand how they perform under different conditions.</p>
                
                <div class="dataset-grid">
                    <div class="dataset-card">
                        <h4>🎬 MovieLens-Style</h4>
                        <p>Dense explicit ratings (1-5 scale), similar to MovieLens 100K dataset.</p>
                        <div class="stats">
                            <div class="stat"><div class="stat-value">500</div><div class="stat-name">Users</div></div>
                            <div class="stat"><div class="stat-value">1,000</div><div class="stat-name">Items</div></div>
                            <div class="stat"><div class="stat-value">~30K</div><div class="stat-name">Ratings</div></div>
                            <div class="stat"><div class="stat-value">94%</div><div class="stat-name">Sparsity</div></div>
                        </div>
                        <p style="margin-top:15px;"><strong>Best for:</strong> CF, Matrix Factorization</p>
                    </div>
                    
                    <div class="dataset-card">
                        <h4>📦 Amazon-Style</h4>
                        <p>Sparse with text reviews and product categories, e-commerce style.</p>
                        <div class="stats">
                            <div class="stat"><div class="stat-value">2,000</div><div class="stat-name">Users</div></div>
                            <div class="stat"><div class="stat-value">1,000</div><div class="stat-name">Items</div></div>
                            <div class="stat"><div class="stat-value">~28K</div><div class="stat-name">Ratings</div></div>
                            <div class="stat"><div class="stat-value">98.6%</div><div class="stat-name">Sparsity</div></div>
                        </div>
                        <p style="margin-top:15px;"><strong>Best for:</strong> Content-Based, Hybrid</p>
                    </div>
                    
                    <div class="dataset-card">
                        <h4>📚 BookCrossing-Style</h4>
                        <p>Very sparse with many cold-start users/items, long-tail distribution.</p>
                        <div class="stats">
                            <div class="stat"><div class="stat-value">3,000</div><div class="stat-name">Users</div></div>
                            <div class="stat"><div class="stat-value">2,000</div><div class="stat-name">Items</div></div>
                            <div class="stat"><div class="stat-value">~40K</div><div class="stat-name">Ratings</div></div>
                            <div class="stat"><div class="stat-value">99.3%</div><div class="stat-name">Sparsity</div></div>
                        </div>
                        <p style="margin-top:15px;"><strong>Best for:</strong> Graph-Based, Popularity</p>
                    </div>
                </div>
                
                <div class="algo-section">
                    <h3>Why Multiple Datasets Matter</h3>
                    <p>Different algorithms excel under different conditions:</p>
                    <ul style="margin-left:20px;">
                        <li><strong>Collaborative Filtering</strong> struggles with sparse data (cold-start)</li>
                        <li><strong>Content-Based</strong> needs rich item features to work well</li>
                        <li><strong>Popularity</strong> is surprisingly robust across all conditions</li>
                        <li><strong>Graph-Based</strong> can handle sparse data by traversing connections</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <!-- Results Tab -->
        <div id="results" class="tab-content">
            <div class="section">
                <h2>📈 Multi-Dataset Results</h2>
                <p>All 8 algorithms evaluated on all 3 datasets. Results show how performance varies with data characteristics.</p>
                
                <div class="image-gallery">
                    {multi_images_html if multi_images_html else '<p>Run 12_multi_dataset_evaluation.py to generate results.</p>'}
                </div>
                
                <h3>Full Results Table</h3>
                <div class="table-container">
                    {multi_table_html if multi_table_html else '<p>No results available yet.</p>'}
                </div>
            </div>
        </div>
        
        <!-- Hyperparameters Tab -->
        <div id="hyperparameters" class="tab-content">
            {hp_tab_content}
        </div>
        
        <!-- Visualizations Tab -->
        <div id="visualizations" class="tab-content">
            <div class="section">
                <h2>🎨 Interactive Visualizations</h2>
                <p>Click to open interactive graph visualizations showing how each algorithm makes recommendations.</p>
                
                <div class="viz-grid">
'''
    
    for viz_file in sorted(viz_files):
        name = viz_file.replace('.html', '').replace('_', ' ').title()
        html += f'''
                    <div class="viz-card">
                        <h4>{name}</h4>
                        <p>Interactive node-edge visualization</p>
                        <a href="{viz_file}" target="_blank">Open Visualization →</a>
                    </div>
'''
    
    html += '''
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Recommendation System Analysis Dashboard | Comprehensive Guide for Beginners to Advanced</p>
        </div>
    </div>
    
    <script>
        function showTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
        }
    </script>
</body>
</html>
'''
    
    return html


if __name__ == "__main__":
    generate_dashboard()
