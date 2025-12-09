"""
09_hyperparameter_tuning.py
============================
Comprehensive Hyperparameter Tuning for All Algorithms

This script:
1. Tests multiple hyperparameter configurations for each algorithm
2. Documents WHY each hyperparameter matters
3. Records HOW testing was performed
4. Saves detailed results for dashboard visualization
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any
from itertools import product
import time
import warnings
warnings.filterwarnings('ignore')

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import DatasetGenerator

# Create output directories
os.makedirs("../reports/hyperparameter_tuning", exist_ok=True)


# ============================================================================
# HYPERPARAMETER DOCUMENTATION
# ============================================================================

HYPERPARAMETER_DOCS = {
    "Collaborative Filtering": {
        "description": "Memory-based approach using user-user or item-item similarity",
        "parameters": {
            "method": {
                "values": ["user", "item"],
                "why": "User-based finds similar users and recommends what they liked. Item-based finds similar items to what user already liked. Item-based is typically more scalable and stable.",
                "impact": "Affects recommendation diversity and computational complexity"
            },
            "k_neighbors": {
                "values": [5, 10, 20, 30, 50],
                "why": "Number of similar users/items to consider. Too few = noisy recommendations. Too many = over-smoothing and slow computation.",
                "impact": "Trade-off between recommendation quality and diversity"
            },
            "similarity_metric": {
                "values": ["cosine", "pearson"],
                "why": "Cosine measures angle between vectors (good for implicit). Pearson accounts for rating bias (better for explicit ratings with varying scales).",
                "impact": "Affects how similarity is computed between users/items"
            }
        }
    },
    "Matrix Factorization": {
        "description": "Latent factor model decomposing user-item matrix into lower dimensions",
        "parameters": {
            "method": {
                "values": ["svd", "als", "funk_svd"],
                "why": "SVD is direct decomposition. ALS alternates optimization (handles implicit). Funk SVD uses SGD (scalable, handles missing values).",
                "impact": "Training speed, memory usage, and handling of sparse data"
            },
            "n_factors": {
                "values": [10, 20, 50, 100],
                "why": "Dimensionality of latent space. More factors = more expressive but risk overfitting. Fewer = better generalization but may miss patterns.",
                "impact": "Model complexity and ability to capture user/item nuances"
            },
            "regularization": {
                "values": [0.01, 0.02, 0.05, 0.1],
                "why": "L2 penalty to prevent overfitting. Higher values = simpler model. Critical for sparse data where overfitting is common.",
                "impact": "Generalization vs fitting training data"
            }
        }
    },
    "Content-Based": {
        "description": "Recommends items similar to what user liked based on item features",
        "parameters": {
            "tfidf_max_features": {
                "values": [100, 300, 500, 1000],
                "why": "Maximum vocabulary size for text features. More = richer representation but noisier. Fewer = focused on important terms.",
                "impact": "Feature space dimensionality and sparsity"
            },
            "tfidf_ngram_range": {
                "values": ["(1,1)", "(1,2)", "(1,3)"],
                "why": "Unigrams capture single words. Bigrams/trigrams capture phrases like 'science fiction' or 'romantic comedy'.",
                "impact": "Ability to capture multi-word concepts"
            },
            "min_df": {
                "values": [1, 2, 5],
                "why": "Minimum document frequency. Filters rare terms that may be noise or typos. Higher = more robust features.",
                "impact": "Noise reduction vs information loss"
            }
        }
    },
    "Graph-Based": {
        "description": "Models user-item interactions as a graph and uses graph algorithms",
        "parameters": {
            "method": {
                "values": ["pagerank", "random_walk"],
                "why": "PageRank computes global importance scores. Random walk simulates user browsing behavior for personalized recommendations.",
                "impact": "Global vs personalized ranking"
            },
            "damping_factor": {
                "values": [0.75, 0.85, 0.95],
                "why": "Probability of following edges vs jumping randomly. Higher = more influence from graph structure. Lower = more exploration.",
                "impact": "Balance between exploitation and exploration"
            },
            "n_walks": {
                "values": [5, 10, 20],
                "why": "Number of random walks per user. More walks = more stable estimates but slower computation.",
                "impact": "Recommendation stability and computation time"
            }
        }
    },
    "Popularity Baseline": {
        "description": "Non-personalized baseline recommending popular items",
        "parameters": {
            "method": {
                "values": ["count", "rating", "weighted", "bayesian"],
                "why": "Count = most interactions. Rating = highest rated. Weighted = combination. Bayesian = IMDB-style (accounts for rating count).",
                "impact": "Definition of 'popular' - quantity vs quality"
            },
            "popularity_weight": {
                "values": [0.3, 0.5, 0.7],
                "why": "For weighted method: balance between popularity count and average rating. Higher = favor more-rated items.",
                "impact": "Trade-off between well-known and highly-rated"
            },
            "min_ratings": {
                "values": [3, 5, 10, 20],
                "why": "Minimum ratings required to be considered. Filters items with unreliable statistics from few ratings.",
                "impact": "Confidence in popularity estimates"
            }
        }
    }
}


class HyperparameterTuner:
    """Comprehensive hyperparameter tuning with documentation"""
    
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, items_df: pd.DataFrame = None):
        self.train_df = train_df
        self.test_df = test_df
        self.items_df = items_df
        self.results = []
        self.tuning_metadata = {
            "start_time": datetime.now().isoformat(),
            "train_size": len(train_df),
            "test_size": len(test_df),
            "methodology": "Grid Search with held-out test set evaluation"
        }
    
    def _safe_evaluate(self, model, test_df, k_values=[5, 10]) -> Dict:
        """Safely evaluate model and return metrics"""
        try:
            return model.evaluate(test_df, k_values=k_values)
        except Exception as e:
            print(f"      ⚠️ Evaluation error: {e}")
            return {"RMSE": np.nan, "MAE": np.nan, "Precision@10": np.nan, 
                    "Recall@10": np.nan, "NDCG@10": np.nan, "Coverage": np.nan}
    
    def tune_collaborative_filtering(self):
        """Tune Collaborative Filtering with all parameter combinations"""
        print("\n" + "=" * 70)
        print("🔍 TUNING: COLLABORATIVE FILTERING")
        print("=" * 70)
        
        from importlib import import_module
        try:
            cf_module = import_module('01_collaborative_filtering')
            CollaborativeFilter = cf_module.CollaborativeFilter
        except:
            print("   ❌ Could not import CollaborativeFilter")
            return []
        
        param_grid = {
            'method': ['user', 'item'],
            'k_neighbors': [5, 10, 20, 30, 50],
            'similarity_metric': ['cosine', 'pearson']
        }
        
        results = []
        combinations = list(product(param_grid['method'], param_grid['k_neighbors'], 
                                    param_grid['similarity_metric']))
        
        for i, (method, k, sim) in enumerate(combinations, 1):
            print(f"   [{i}/{len(combinations)}] method={method}, k={k}, similarity={sim}")
            
            try:
                start_time = time.time()
                model = CollaborativeFilter(method=method, k_neighbors=k, similarity_metric=sim)
                model.fit(self.train_df)
                train_time = time.time() - start_time
                
                metrics = self._safe_evaluate(model, self.test_df)
                
                results.append({
                    'algorithm': 'Collaborative Filtering',
                    'method': method,
                    'k_neighbors': k,
                    'similarity_metric': sim,
                    'train_time': round(train_time, 3),
                    **metrics
                })
                print(f"      ✅ RMSE: {metrics.get('RMSE', 'N/A'):.4f}" if not np.isnan(metrics.get('RMSE', np.nan)) else "      ⚠️ No RMSE")
            except Exception as e:
                print(f"      ❌ Error: {e}")
        
        self.results.extend(results)
        return results
    
    def tune_matrix_factorization(self):
        """Tune Matrix Factorization with all parameter combinations"""
        print("\n" + "=" * 70)
        print("🔍 TUNING: MATRIX FACTORIZATION")
        print("=" * 70)
        
        from importlib import import_module
        try:
            mf_module = import_module('02_matrix_factorization')
            MatrixFactorization = mf_module.MatrixFactorization
        except:
            print("   ❌ Could not import MatrixFactorization")
            return []
        
        param_grid = {
            'method': ['svd', 'als', 'funk_svd'],
            'n_factors': [10, 20, 50, 100],
            'regularization': [0.01, 0.02, 0.05, 0.1]
        }
        
        results = []
        combinations = list(product(param_grid['method'], param_grid['n_factors'], 
                                    param_grid['regularization']))
        
        for i, (method, n_factors, reg) in enumerate(combinations, 1):
            print(f"   [{i}/{len(combinations)}] method={method}, factors={n_factors}, reg={reg}")
            
            try:
                start_time = time.time()
                model = MatrixFactorization(method=method, n_factors=n_factors, 
                                           n_epochs=15, regularization=reg)
                model.fit(self.train_df)
                train_time = time.time() - start_time
                
                metrics = self._safe_evaluate(model, self.test_df)
                
                results.append({
                    'algorithm': 'Matrix Factorization',
                    'method': method,
                    'n_factors': n_factors,
                    'regularization': reg,
                    'train_time': round(train_time, 3),
                    **metrics
                })
                print(f"      ✅ RMSE: {metrics.get('RMSE', 'N/A'):.4f}" if not np.isnan(metrics.get('RMSE', np.nan)) else "      ⚠️ No RMSE")
            except Exception as e:
                print(f"      ❌ Error: {e}")
        
        self.results.extend(results)
        return results
    
    def tune_content_based(self):
        """Tune Content-Based Filtering with all parameter combinations"""
        print("\n" + "=" * 70)
        print("🔍 TUNING: CONTENT-BASED FILTERING")
        print("=" * 70)
        
        if self.items_df is None:
            print("   ❌ No items dataframe provided")
            return []
        
        from importlib import import_module
        try:
            cb_module = import_module('03_content_based')
            ContentBasedRecommender = cb_module.ContentBasedRecommender
        except:
            print("   ❌ Could not import ContentBasedRecommender")
            return []
        
        param_grid = {
            'tfidf_max_features': [100, 300, 500, 1000],
            'tfidf_ngram_range': [(1, 1), (1, 2), (1, 3)],
            'min_df': [1, 2, 5]
        }
        
        results = []
        combinations = list(product(param_grid['tfidf_max_features'], 
                                    param_grid['tfidf_ngram_range'],
                                    param_grid['min_df']))
        
        for i, (max_features, ngram_range, min_df) in enumerate(combinations, 1):
            print(f"   [{i}/{len(combinations)}] features={max_features}, ngram={ngram_range}, min_df={min_df}")
            
            try:
                start_time = time.time()
                model = ContentBasedRecommender(
                    tfidf_max_features=max_features,
                    tfidf_ngram_range=ngram_range,
                    min_df=min_df
                )
                model.fit(self.items_df, self.train_df, feature_columns=['genres'])
                train_time = time.time() - start_time
                
                metrics = self._safe_evaluate(model, self.test_df)
                
                results.append({
                    'algorithm': 'Content-Based',
                    'tfidf_max_features': max_features,
                    'tfidf_ngram_range': str(ngram_range),
                    'min_df': min_df,
                    'train_time': round(train_time, 3),
                    **metrics
                })
                print(f"      ✅ RMSE: {metrics.get('RMSE', 'N/A'):.4f}" if not np.isnan(metrics.get('RMSE', np.nan)) else "      ⚠️ No RMSE")
            except Exception as e:
                print(f"      ❌ Error: {e}")
        
        self.results.extend(results)
        return results
    
    def tune_graph_based(self):
        """Tune Graph-Based with all parameter combinations"""
        print("\n" + "=" * 70)
        print("🔍 TUNING: GRAPH-BASED")
        print("=" * 70)
        
        from importlib import import_module
        try:
            gb_module = import_module('04_graph_based')
            GraphBasedRecommender = gb_module.GraphBasedRecommender
        except:
            print("   ❌ Could not import GraphBasedRecommender")
            return []
        
        param_grid = {
            'method': ['pagerank', 'random_walk'],
            'damping_factor': [0.75, 0.85, 0.95],
            'n_walks': [5, 10, 20]
        }
        
        results = []
        combinations = list(product(param_grid['method'], param_grid['damping_factor'], 
                                    param_grid['n_walks']))
        
        for i, (method, damping, n_walks) in enumerate(combinations, 1):
            print(f"   [{i}/{len(combinations)}] method={method}, damping={damping}, walks={n_walks}")
            
            try:
                start_time = time.time()
                model = GraphBasedRecommender(method=method, damping_factor=damping, n_walks=n_walks)
                model.fit(self.train_df)
                train_time = time.time() - start_time
                
                metrics = self._safe_evaluate(model, self.test_df)
                
                results.append({
                    'algorithm': 'Graph-Based',
                    'method': method,
                    'damping_factor': damping,
                    'n_walks': n_walks,
                    'train_time': round(train_time, 3),
                    **metrics
                })
                print(f"      ✅ RMSE: {metrics.get('RMSE', 'N/A'):.4f}" if not np.isnan(metrics.get('RMSE', np.nan)) else "      ⚠️ No RMSE")
            except Exception as e:
                print(f"      ❌ Error: {e}")
        
        self.results.extend(results)
        return results
    
    def tune_popularity(self):
        """Tune Popularity Baseline with all parameter combinations"""
        print("\n" + "=" * 70)
        print("🔍 TUNING: POPULARITY BASELINE")
        print("=" * 70)
        
        from importlib import import_module
        try:
            pop_module = import_module('08_popularity_baseline')
            PopularityRecommender = pop_module.PopularityRecommender
        except:
            print("   ❌ Could not import PopularityRecommender")
            return []
        
        param_grid = {
            'method': ['count', 'rating', 'weighted', 'bayesian'],
            'popularity_weight': [0.3, 0.5, 0.7],
            'min_ratings': [3, 5, 10, 20]
        }
        
        results = []
        combinations = list(product(param_grid['method'], param_grid['popularity_weight'], 
                                    param_grid['min_ratings']))
        
        for i, (method, pop_weight, min_ratings) in enumerate(combinations, 1):
            print(f"   [{i}/{len(combinations)}] method={method}, weight={pop_weight}, min={min_ratings}")
            
            try:
                start_time = time.time()
                model = PopularityRecommender(method=method, popularity_weight=pop_weight, 
                                             min_ratings=min_ratings)
                model.fit(self.train_df)
                train_time = time.time() - start_time
                
                metrics = self._safe_evaluate(model, self.test_df)
                
                results.append({
                    'algorithm': 'Popularity Baseline',
                    'method': method,
                    'popularity_weight': pop_weight,
                    'min_ratings': min_ratings,
                    'train_time': round(train_time, 3),
                    **metrics
                })
                print(f"      ✅ RMSE: {metrics.get('RMSE', 'N/A'):.4f}" if not np.isnan(metrics.get('RMSE', np.nan)) else "      ⚠️ No RMSE")
            except Exception as e:
                print(f"      ❌ Error: {e}")
        
        self.results.extend(results)
        return results
    
    def generate_visualizations(self, output_dir: str):
        """Generate comprehensive visualizations for hyperparameter analysis"""
        print("\n📊 Generating hyperparameter visualizations...")
        
        results_df = pd.DataFrame(self.results)
        if len(results_df) == 0:
            print("   ⚠️ No results to visualize")
            return
        
        # Create multi-page figure
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Overall Algorithm Comparison
        ax1 = fig.add_subplot(4, 2, 1)
        algo_stats = results_df.groupby('algorithm')['RMSE'].agg(['mean', 'std', 'min']).reset_index()
        algo_stats = algo_stats.dropna()
        if len(algo_stats) > 0:
            bars = ax1.bar(range(len(algo_stats)), algo_stats['mean'], yerr=algo_stats['std'], 
                          capsize=5, color='steelblue', alpha=0.8)
            ax1.set_xticks(range(len(algo_stats)))
            ax1.set_xticklabels(algo_stats['algorithm'], rotation=45, ha='right')
            ax1.set_ylabel('RMSE')
            ax1.set_title('RMSE by Algorithm (mean ± std across all configs)')
            for i, (bar, min_val) in enumerate(zip(bars, algo_stats['min'])):
                ax1.annotate(f'best: {min_val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                            xytext=(0, 5), textcoords='offset points', ha='center', fontsize=8)
        
        # 2. Best Config per Algorithm
        ax2 = fig.add_subplot(4, 2, 2)
        best_configs = results_df.loc[results_df.groupby('algorithm')['RMSE'].idxmin()]
        if len(best_configs) > 0:
            colors = plt.cm.Set2(np.linspace(0, 1, len(best_configs)))
            bars = ax2.barh(range(len(best_configs)), best_configs['RMSE'], color=colors)
            ax2.set_yticks(range(len(best_configs)))
            ax2.set_yticklabels(best_configs['algorithm'])
            ax2.set_xlabel('RMSE (lower is better)')
            ax2.set_title('Best RMSE per Algorithm')
            for i, bar in enumerate(bars):
                ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{bar.get_width():.4f}', va='center', fontsize=9)
        
        # 3. CF: k_neighbors effect
        ax3 = fig.add_subplot(4, 2, 3)
        cf_results = results_df[results_df['algorithm'] == 'Collaborative Filtering']
        if len(cf_results) > 0 and 'k_neighbors' in cf_results.columns:
            for method in cf_results['method'].dropna().unique():
                subset = cf_results[cf_results['method'] == method]
                grouped = subset.groupby('k_neighbors')['RMSE'].mean()
                ax3.plot(grouped.index, grouped.values, marker='o', label=f'{method}-based', linewidth=2)
            ax3.set_xlabel('k_neighbors')
            ax3.set_ylabel('RMSE')
            ax3.set_title('Collaborative Filtering: Effect of k_neighbors')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. MF: n_factors effect
        ax4 = fig.add_subplot(4, 2, 4)
        mf_results = results_df[results_df['algorithm'] == 'Matrix Factorization']
        if len(mf_results) > 0 and 'n_factors' in mf_results.columns:
            for method in mf_results['method'].dropna().unique():
                subset = mf_results[mf_results['method'] == method]
                grouped = subset.groupby('n_factors')['RMSE'].mean()
                ax4.plot(grouped.index, grouped.values, marker='s', label=method.upper(), linewidth=2)
            ax4.set_xlabel('n_factors (latent dimensions)')
            ax4.set_ylabel('RMSE')
            ax4.set_title('Matrix Factorization: Effect of n_factors')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. MF: Regularization effect
        ax5 = fig.add_subplot(4, 2, 5)
        if len(mf_results) > 0 and 'regularization' in mf_results.columns:
            for method in mf_results['method'].dropna().unique():
                subset = mf_results[mf_results['method'] == method]
                grouped = subset.groupby('regularization')['RMSE'].mean()
                ax5.plot(grouped.index, grouped.values, marker='^', label=method.upper(), linewidth=2)
            ax5.set_xlabel('Regularization (λ)')
            ax5.set_ylabel('RMSE')
            ax5.set_title('Matrix Factorization: Effect of Regularization')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Graph-Based: Damping factor
        ax6 = fig.add_subplot(4, 2, 6)
        gb_results = results_df[results_df['algorithm'] == 'Graph-Based']
        if len(gb_results) > 0 and 'damping_factor' in gb_results.columns:
            for method in gb_results['method'].dropna().unique():
                subset = gb_results[gb_results['method'] == method]
                grouped = subset.groupby('damping_factor')['RMSE'].mean()
                ax6.plot(grouped.index, grouped.values, marker='d', label=method, linewidth=2)
            ax6.set_xlabel('Damping Factor')
            ax6.set_ylabel('RMSE')
            ax6.set_title('Graph-Based: Effect of Damping Factor')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 7. Training Time Comparison
        ax7 = fig.add_subplot(4, 2, 7)
        if 'train_time' in results_df.columns:
            time_stats = results_df.groupby('algorithm')['train_time'].agg(['mean', 'std']).reset_index()
            time_stats = time_stats.dropna()
            if len(time_stats) > 0:
                bars = ax7.bar(range(len(time_stats)), time_stats['mean'], yerr=time_stats['std'],
                              capsize=5, color='coral', alpha=0.8)
                ax7.set_xticks(range(len(time_stats)))
                ax7.set_xticklabels(time_stats['algorithm'], rotation=45, ha='right')
                ax7.set_ylabel('Training Time (seconds)')
                ax7.set_title('Average Training Time by Algorithm')
        
        # 8. RMSE vs Coverage Trade-off
        ax8 = fig.add_subplot(4, 2, 8)
        if 'Coverage' in results_df.columns:
            for algo in results_df['algorithm'].unique():
                subset = results_df[results_df['algorithm'] == algo]
                ax8.scatter(subset['RMSE'], subset['Coverage'] * 100, label=algo, alpha=0.6, s=50)
            ax8.set_xlabel('RMSE (lower is better)')
            ax8.set_ylabel('Coverage % (higher is better)')
            ax8.set_title('RMSE vs Coverage Trade-off')
            ax8.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax8.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/hyperparameter_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {output_dir}/hyperparameter_analysis.png")
    
    def save_results(self, output_dir: str):
        """Save all tuning results and documentation"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save raw results CSV
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(f"{output_dir}/all_tuning_results.csv", index=False)
        
        # 2. Save comprehensive JSON with documentation
        full_report = {
            "metadata": self.tuning_metadata,
            "hyperparameter_documentation": HYPERPARAMETER_DOCS,
            "results_summary": {},
            "best_configurations": {},
            "all_results": self.results
        }
        
        # Add summary per algorithm
        for algo in results_df['algorithm'].unique():
            algo_df = results_df[results_df['algorithm'] == algo]
            
            # Find best configuration
            best_idx = algo_df['RMSE'].idxmin() if not algo_df['RMSE'].isna().all() else None
            best_config = algo_df.loc[best_idx].to_dict() if best_idx is not None else {}
            
            full_report["results_summary"][algo] = {
                "total_configs_tested": len(algo_df),
                "rmse_mean": float(algo_df['RMSE'].mean()) if not algo_df['RMSE'].isna().all() else None,
                "rmse_std": float(algo_df['RMSE'].std()) if not algo_df['RMSE'].isna().all() else None,
                "rmse_min": float(algo_df['RMSE'].min()) if not algo_df['RMSE'].isna().all() else None,
                "rmse_max": float(algo_df['RMSE'].max()) if not algo_df['RMSE'].isna().all() else None,
            }
            
            full_report["best_configurations"][algo] = {
                k: (float(v) if isinstance(v, (np.floating, float)) else v)
                for k, v in best_config.items()
            }
        
        full_report["metadata"]["end_time"] = datetime.now().isoformat()
        full_report["metadata"]["total_configurations_tested"] = len(self.results)
        
        with open(f"{output_dir}/hyperparameter_report.json", 'w') as f:
            json.dump(full_report, f, indent=2, default=str)
        
        # 3. Generate text report
        self._generate_text_report(output_dir, results_df, full_report)
        
        print(f"\n📁 Results saved to: {output_dir}/")
        print(f"   - all_tuning_results.csv")
        print(f"   - hyperparameter_report.json")
        print(f"   - hyperparameter_report.txt")
        print(f"   - hyperparameter_analysis.png")
    
    def _generate_text_report(self, output_dir: str, results_df: pd.DataFrame, full_report: dict):
        """Generate human-readable text report"""
        
        report = f"""
{'='*80}
HYPERPARAMETER TUNING REPORT
{'='*80}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Configurations Tested: {len(self.results)}
Training Set Size: {self.tuning_metadata['train_size']}
Test Set Size: {self.tuning_metadata['test_size']}
Methodology: {self.tuning_metadata['methodology']}

{'='*80}
METHODOLOGY
{'='*80}

Grid Search Approach:
- For each algorithm, we test all combinations of hyperparameters
- Each configuration is trained on the training set
- Evaluation metrics are computed on the held-out test set
- Best configuration is selected based on lowest RMSE

Evaluation Metrics:
- RMSE (Root Mean Squared Error): Prediction accuracy (lower is better)
- MAE (Mean Absolute Error): Average prediction error
- Precision@K: Relevant items in top-K recommendations
- NDCG@K: Ranking quality metric
- Coverage: Percentage of items ever recommended

{'='*80}
HYPERPARAMETER EXPLANATIONS
{'='*80}
"""
        
        for algo, doc in HYPERPARAMETER_DOCS.items():
            report += f"\n{algo}\n{'-'*len(algo)}\n"
            report += f"Description: {doc['description']}\n\n"
            
            for param, info in doc['parameters'].items():
                report += f"  • {param}:\n"
                report += f"    Values tested: {info['values']}\n"
                report += f"    Why it matters: {info['why']}\n"
                report += f"    Impact: {info['impact']}\n\n"
        
        report += f"""
{'='*80}
RESULTS BY ALGORITHM
{'='*80}
"""
        
        for algo in results_df['algorithm'].unique():
            algo_df = results_df[results_df['algorithm'] == algo]
            
            report += f"\n{algo}\n{'-'*len(algo)}\n"
            report += f"Configurations tested: {len(algo_df)}\n"
            
            if not algo_df['RMSE'].isna().all():
                report += f"RMSE: {algo_df['RMSE'].mean():.4f} ± {algo_df['RMSE'].std():.4f}\n"
                report += f"Best RMSE: {algo_df['RMSE'].min():.4f}\n"
                
                best_idx = algo_df['RMSE'].idxmin()
                best_row = algo_df.loc[best_idx]
                
                report += f"\nBest Configuration:\n"
                param_cols = [c for c in best_row.index if c not in 
                             ['algorithm', 'RMSE', 'MAE', 'train_time', 'Coverage'] 
                             and not c.startswith('Precision') and not c.startswith('Recall') 
                             and not c.startswith('NDCG')]
                for col in param_cols:
                    if pd.notna(best_row[col]):
                        report += f"  - {col}: {best_row[col]}\n"
                
                report += f"\nMetrics for best config:\n"
                report += f"  - RMSE: {best_row['RMSE']:.4f}\n"
                if 'MAE' in best_row and pd.notna(best_row['MAE']):
                    report += f"  - MAE: {best_row['MAE']:.4f}\n"
                if 'Precision@10' in best_row and pd.notna(best_row['Precision@10']):
                    report += f"  - Precision@10: {best_row['Precision@10']:.4f}\n"
                if 'NDCG@10' in best_row and pd.notna(best_row['NDCG@10']):
                    report += f"  - NDCG@10: {best_row['NDCG@10']:.4f}\n"
                if 'Coverage' in best_row and pd.notna(best_row['Coverage']):
                    report += f"  - Coverage: {best_row['Coverage']*100:.1f}%\n"
        
        report += f"""

{'='*80}
KEY FINDINGS
{'='*80}

"""
        # Add key findings
        if len(results_df) > 0 and not results_df['RMSE'].isna().all():
            best_overall = results_df.loc[results_df['RMSE'].idxmin()]
            report += f"1. Best Overall Configuration:\n"
            report += f"   Algorithm: {best_overall['algorithm']}\n"
            report += f"   RMSE: {best_overall['RMSE']:.4f}\n\n"
            
            # Find most sensitive parameter
            report += "2. Hyperparameter Sensitivity Observations:\n"
            
            cf_df = results_df[results_df['algorithm'] == 'Collaborative Filtering']
            if len(cf_df) > 0 and 'k_neighbors' in cf_df.columns:
                k_impact = cf_df.groupby('k_neighbors')['RMSE'].mean()
                report += f"   - CF k_neighbors: RMSE range {k_impact.min():.4f} - {k_impact.max():.4f}\n"
            
            mf_df = results_df[results_df['algorithm'] == 'Matrix Factorization']
            if len(mf_df) > 0 and 'n_factors' in mf_df.columns:
                f_impact = mf_df.groupby('n_factors')['RMSE'].mean()
                report += f"   - MF n_factors: RMSE range {f_impact.min():.4f} - {f_impact.max():.4f}\n"
        
        report += f"\n{'='*80}\n"
        
        with open(f"{output_dir}/hyperparameter_report.txt", 'w') as f:
            f.write(report)


def main():
    """Main hyperparameter tuning execution"""
    print("=" * 70)
    print("🔬 COMPREHENSIVE HYPERPARAMETER TUNING")
    print("=" * 70)
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Generate dataset
    print("\n📦 Generating dataset...")
    generator = DatasetGenerator(output_dir="../data")
    dataset = generator.generate_movielens_style(
        n_users=400,
        n_items=800,
        n_ratings=40000
    )
    
    ratings_df = dataset['ratings']
    items_df = dataset['items']
    
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
    print(f"   Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Initialize tuner
    tuner = HyperparameterTuner(train_df, test_df, items_df)
    
    # Run all tuning
    tuner.tune_collaborative_filtering()
    tuner.tune_matrix_factorization()
    tuner.tune_content_based()
    tuner.tune_graph_based()
    tuner.tune_popularity()
    
    # Generate outputs
    output_dir = "../reports/hyperparameter_tuning"
    tuner.generate_visualizations(output_dir)
    tuner.save_results(output_dir)
    
    print("\n" + "=" * 70)
    print("✅ HYPERPARAMETER TUNING COMPLETE!")
    print("=" * 70)
    
    return tuner.results


if __name__ == "__main__":
    main()
