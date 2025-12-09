"""
10_master_comparison.py
========================
Master Comparison Script for All Recommendation Algorithms

This script:
1. Trains all algorithms with their best configurations
2. Evaluates them on the same test set
3. Generates comprehensive comparison reports
4. Creates a unified HTML dashboard with all visualizations
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
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import DatasetGenerator

# Output directories
os.makedirs("../reports/master_comparison", exist_ok=True)
os.makedirs("../visualizations", exist_ok=True)


class AlgorithmBenchmark:
    """Benchmark wrapper for consistent algorithm evaluation"""
    
    def __init__(self, name: str, model: Any, train_time: float = 0):
        self.name = name
        self.model = model
        self.train_time = train_time
        self.metrics = {}
        self.predictions = []
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'train_time': self.train_time,
            'metrics': self.metrics
        }


def train_all_algorithms(train_df: pd.DataFrame, items_df: pd.DataFrame) -> List[AlgorithmBenchmark]:
    """Train all algorithms and return benchmark objects"""
    
    benchmarks = []
    
    # 1. User-Based Collaborative Filtering
    print("\n[1/8] Training User-Based Collaborative Filtering...")
    try:
        from importlib import import_module; cf_mod = import_module('01_collaborative_filtering'); CollaborativeFilter = cf_mod.CollaborativeFilter # import CollaborativeFilter
        import time
        
        start = time.time()
        cf_user = CollaborativeFilter(method='user', k_neighbors=20, similarity_metric='cosine')
        cf_user.fit(train_df)
        train_time = time.time() - start
        
        benchmarks.append(AlgorithmBenchmark("User-CF", cf_user, train_time))
        print(f"   ✅ Trained in {train_time:.2f}s")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # 2. Item-Based Collaborative Filtering
    print("\n[2/8] Training Item-Based Collaborative Filtering...")
    try:
        from importlib import import_module; cf_mod = import_module('01_collaborative_filtering'); CollaborativeFilter = cf_mod.CollaborativeFilter # import CollaborativeFilter
        import time
        
        start = time.time()
        cf_item = CollaborativeFilter(method='item', k_neighbors=20, similarity_metric='cosine')
        cf_item.fit(train_df)
        train_time = time.time() - start
        
        benchmarks.append(AlgorithmBenchmark("Item-CF", cf_item, train_time))
        print(f"   ✅ Trained in {train_time:.2f}s")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # 3. SVD Matrix Factorization
    print("\n[3/8] Training SVD Matrix Factorization...")
    try:
        from importlib import import_module
        mf_mod = import_module('02_matrix_factorization')
        MatrixFactorization = mf_mod.MatrixFactorization
        import time
        
        start = time.time()
        mf_svd = MatrixFactorization(method='svd', n_factors=30)
        mf_svd.fit(train_df)
        train_time = time.time() - start
        
        benchmarks.append(AlgorithmBenchmark("MF-SVD", mf_svd, train_time))
        print(f"   ✅ Trained in {train_time:.2f}s")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # 4. ALS Matrix Factorization
    print("\n[4/8] Training ALS Matrix Factorization...")
    try:
        from importlib import import_module
        mf_mod = import_module('02_matrix_factorization')
        MatrixFactorization = mf_mod.MatrixFactorization
        import time
        
        start = time.time()
        mf_als = MatrixFactorization(method='als', n_factors=30, n_epochs=15, regularization=0.1)
        mf_als.fit(train_df)
        train_time = time.time() - start
        
        benchmarks.append(AlgorithmBenchmark("MF-ALS", mf_als, train_time))
        print(f"   ✅ Trained in {train_time:.2f}s")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # 5. Content-Based Filtering
    print("\n[5/8] Training Content-Based Filtering...")
    try:
        from importlib import import_module
        cb_mod = import_module('03_content_based')
        ContentBasedRecommender = cb_mod.ContentBasedRecommender
        import time
        
        start = time.time()
        cb = ContentBasedRecommender(tfidf_max_features=500, use_features=True)
        cb.fit(items_df, train_df, feature_columns=['genres'])
        train_time = time.time() - start
        
        benchmarks.append(AlgorithmBenchmark("Content-Based", cb, train_time))
        print(f"   ✅ Trained in {train_time:.2f}s")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # 6. Graph-Based (PageRank)
    print("\n[6/8] Training Graph-Based (PageRank)...")
    try:
        from importlib import import_module
        gb_mod = import_module('04_graph_based')
        GraphBasedRecommender = gb_mod.GraphBasedRecommender
        import time
        
        start = time.time()
        graph_pr = GraphBasedRecommender(method='pagerank', damping_factor=0.85)
        graph_pr.fit(train_df)
        train_time = time.time() - start
        
        benchmarks.append(AlgorithmBenchmark("Graph-PageRank", graph_pr, train_time))
        print(f"   ✅ Trained in {train_time:.2f}s")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # 7. Popularity Baseline (Weighted)
    print("\n[7/8] Training Popularity Baseline...")
    try:
        from importlib import import_module
        pop_mod = import_module('08_popularity_baseline')
        PopularityRecommender = pop_mod.PopularityRecommender
        import time
        
        start = time.time()
        pop = PopularityRecommender(method='weighted', popularity_weight=0.5)
        pop.fit(train_df)
        train_time = time.time() - start
        
        benchmarks.append(AlgorithmBenchmark("Popularity", pop, train_time))
        print(f"   ✅ Trained in {train_time:.2f}s")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # 8. Popularity Baseline (Bayesian)
    print("\n[8/10] Training Bayesian Popularity...")
    try:
        from importlib import import_module
        pop_mod = import_module('08_popularity_baseline')
        PopularityRecommender = pop_mod.PopularityRecommender
        import time
        
        start = time.time()
        pop_bayes = PopularityRecommender(method='bayesian')
        pop_bayes.fit(train_df)
        train_time = time.time() - start
        
        benchmarks.append(AlgorithmBenchmark("Pop-Bayesian", pop_bayes, train_time))
        print(f"   ✅ Trained in {train_time:.2f}s")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # 9. Neural CF (NeuMF) - Optional, requires PyTorch
    print("\n[9/10] Training Neural CF (NeuMF)...")
    try:
        import torch
        from importlib import import_module
        ncf_mod = import_module('05_neural_cf')
        NeuralCFRecommender = ncf_mod.NeuralCFRecommender
        import time
        
        start = time.time()
        ncf = NeuralCFRecommender(
            model_type='neumf',
            embedding_dim=32,
            hidden_layers=[64, 32, 16],
            n_epochs=10,
            batch_size=256
        )
        ncf.fit(train_df)
        train_time = time.time() - start
        
        benchmarks.append(AlgorithmBenchmark("Neural-CF", ncf, train_time))
        print(f"   ✅ Trained in {train_time:.2f}s")
    except ImportError:
        print("   ⚠️ Skipped (PyTorch not installed)")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # 10. Hybrid Recommender
    print("\n[10/10] Training Hybrid Recommender...")
    try:
        from importlib import import_module
        hybrid_mod = import_module('06_hybrid')
        HybridRecommender = hybrid_mod.HybridRecommender
        import time
        
        start = time.time()
        hybrid = HybridRecommender(
            strategy='weighted',
            weights={'collaborative': 0.4, 'content': 0.3, 'popularity': 0.3},
            cold_start_threshold=5
        )
        hybrid.fit(train_df, items_df)
        train_time = time.time() - start
        
        benchmarks.append(AlgorithmBenchmark("Hybrid", hybrid, train_time))
        print(f"   ✅ Trained in {train_time:.2f}s")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    return benchmarks


def evaluate_all_algorithms(benchmarks: List[AlgorithmBenchmark], test_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate all algorithms on the same test set"""
    
    print("\n" + "="*60)
    print("EVALUATING ALL ALGORITHMS")
    print("="*60)
    
    results = []
    
    for i, benchmark in enumerate(benchmarks):
        print(f"\n[{i+1}/{len(benchmarks)}] Evaluating {benchmark.name}...")
        
        try:
            metrics = benchmark.model.evaluate(test_df, k_values=[5, 10, 20])
            benchmark.metrics = metrics
            
            results.append({
                'Algorithm': benchmark.name,
                'Train Time (s)': round(benchmark.train_time, 2),
                'RMSE': round(metrics.get('RMSE', 0), 4),
                'MAE': round(metrics.get('MAE', 0), 4),
                'Precision@5': round(metrics.get('Precision@5', 0), 4),
                'Precision@10': round(metrics.get('Precision@10', 0), 4),
                'Precision@20': round(metrics.get('Precision@20', 0), 4),
                'Recall@5': round(metrics.get('Recall@5', 0), 4),
                'Recall@10': round(metrics.get('Recall@10', 0), 4),
                'Recall@20': round(metrics.get('Recall@20', 0), 4),
                'NDCG@5': round(metrics.get('NDCG@5', 0), 4),
                'NDCG@10': round(metrics.get('NDCG@10', 0), 4),
                'NDCG@20': round(metrics.get('NDCG@20', 0), 4),
                'Coverage': round(metrics.get('Coverage', 0), 4)
            })
            
            print(f"   ✅ RMSE: {metrics.get('RMSE', 0):.4f}, Precision@10: {metrics.get('Precision@10', 0):.4f}")
        
        except Exception as e:
            print(f"   ❌ Evaluation failed: {e}")
    
    return pd.DataFrame(results)


def generate_comparison_visualizations(results_df: pd.DataFrame, output_dir: str):
    """Generate comprehensive comparison visualizations"""
    
    print("\n📊 Generating comparison visualizations...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 20))
    
    algorithms = results_df['Algorithm'].tolist()
    n_algos = len(algorithms)
    colors = plt.cm.Set3(np.linspace(0, 1, n_algos))
    
    # 1. RMSE Comparison (lower is better)
    ax1 = fig.add_subplot(3, 3, 1)
    bars = ax1.barh(range(n_algos), results_df['RMSE'], color=colors, edgecolor='black')
    ax1.set_yticks(range(n_algos))
    ax1.set_yticklabels(algorithms)
    ax1.set_xlabel('RMSE (lower is better)')
    ax1.set_title('Rating Prediction: RMSE')
    ax1.invert_yaxis()
    for i, (bar, val) in enumerate(zip(bars, results_df['RMSE'])):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=8)
    
    # 2. MAE Comparison
    ax2 = fig.add_subplot(3, 3, 2)
    bars = ax2.barh(range(n_algos), results_df['MAE'], color=colors, edgecolor='black')
    ax2.set_yticks(range(n_algos))
    ax2.set_yticklabels(algorithms)
    ax2.set_xlabel('MAE (lower is better)')
    ax2.set_title('Rating Prediction: MAE')
    ax2.invert_yaxis()
    
    # 3. Precision@K comparison
    ax3 = fig.add_subplot(3, 3, 3)
    x = np.arange(n_algos)
    width = 0.25
    ax3.bar(x - width, results_df['Precision@5'], width, label='P@5', color='#3498db')
    ax3.bar(x, results_df['Precision@10'], width, label='P@10', color='#2ecc71')
    ax3.bar(x + width, results_df['Precision@20'], width, label='P@20', color='#e74c3c')
    ax3.set_xticks(x)
    ax3.set_xticklabels(algorithms, rotation=45, ha='right')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision@K Comparison')
    ax3.legend()
    
    # 4. Recall@K comparison
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.bar(x - width, results_df['Recall@5'], width, label='R@5', color='#3498db')
    ax4.bar(x, results_df['Recall@10'], width, label='R@10', color='#2ecc71')
    ax4.bar(x + width, results_df['Recall@20'], width, label='R@20', color='#e74c3c')
    ax4.set_xticks(x)
    ax4.set_xticklabels(algorithms, rotation=45, ha='right')
    ax4.set_ylabel('Recall')
    ax4.set_title('Recall@K Comparison')
    ax4.legend()
    
    # 5. NDCG@K comparison
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.bar(x - width, results_df['NDCG@5'], width, label='NDCG@5', color='#3498db')
    ax5.bar(x, results_df['NDCG@10'], width, label='NDCG@10', color='#2ecc71')
    ax5.bar(x + width, results_df['NDCG@20'], width, label='NDCG@20', color='#e74c3c')
    ax5.set_xticks(x)
    ax5.set_xticklabels(algorithms, rotation=45, ha='right')
    ax5.set_ylabel('NDCG')
    ax5.set_title('NDCG@K Comparison')
    ax5.legend()
    
    # 6. Coverage comparison
    ax6 = fig.add_subplot(3, 3, 6)
    bars = ax6.barh(range(n_algos), results_df['Coverage'] * 100, color=colors, edgecolor='black')
    ax6.set_yticks(range(n_algos))
    ax6.set_yticklabels(algorithms)
    ax6.set_xlabel('Coverage (%)')
    ax6.set_title('Catalog Coverage (higher is better)')
    ax6.invert_yaxis()
    
    # 7. Training time comparison
    ax7 = fig.add_subplot(3, 3, 7)
    bars = ax7.barh(range(n_algos), results_df['Train Time (s)'], color=colors, edgecolor='black')
    ax7.set_yticks(range(n_algos))
    ax7.set_yticklabels(algorithms)
    ax7.set_xlabel('Training Time (seconds)')
    ax7.set_title('Training Efficiency')
    ax7.invert_yaxis()
    
    # 8. Radar chart for top 4 algorithms
    ax8 = fig.add_subplot(3, 3, 8, projection='polar')
    
    # Select metrics for radar
    radar_metrics = ['RMSE', 'Precision@10', 'Recall@10', 'NDCG@10', 'Coverage']
    
    # Normalize metrics (0-1 scale, higher is better)
    radar_data = results_df[radar_metrics].copy()
    radar_data['RMSE'] = 1 - (radar_data['RMSE'] - radar_data['RMSE'].min()) / (radar_data['RMSE'].max() - radar_data['RMSE'].min() + 0.001)
    
    for col in ['Precision@10', 'Recall@10', 'NDCG@10', 'Coverage']:
        radar_data[col] = (radar_data[col] - radar_data[col].min()) / (radar_data[col].max() - radar_data[col].min() + 0.001)
    
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    for i, algo in enumerate(algorithms[:4]):
        values = radar_data.iloc[i].tolist()
        values += values[:1]
        ax8.plot(angles, values, 'o-', linewidth=2, label=algo)
        ax8.fill(angles, values, alpha=0.1)
    
    ax8.set_xticks(angles[:-1])
    ax8.set_xticklabels(radar_metrics, size=8)
    ax8.set_title('Multi-Metric Comparison (Top 4)')
    ax8.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    
    # 9. Summary rankings
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')
    
    # Calculate rankings
    rankings = pd.DataFrame()
    rankings['Algorithm'] = algorithms
    rankings['RMSE Rank'] = results_df['RMSE'].rank()
    rankings['Precision Rank'] = results_df['Precision@10'].rank(ascending=False)
    rankings['Coverage Rank'] = results_df['Coverage'].rank(ascending=False)
    rankings['Avg Rank'] = rankings[['RMSE Rank', 'Precision Rank', 'Coverage Rank']].mean(axis=1)
    rankings = rankings.sort_values('Avg Rank')
    
    summary_text = "🏆 ALGORITHM RANKINGS\n" + "="*40 + "\n\n"
    summary_text += f"{'Rank':<6}{'Algorithm':<18}{'Avg Score':<12}\n"
    summary_text += "-"*40 + "\n"
    
    for i, (_, row) in enumerate(rankings.iterrows(), 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        summary_text += f"{medal} {i:<3}{row['Algorithm']:<18}{row['Avg Rank']:.2f}\n"
    
    ax9.text(0.1, 0.95, summary_text, fontsize=10, family='monospace',
             verticalalignment='top', transform=ax9.transAxes)
    ax9.set_title('Overall Rankings')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/algorithm_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save results
    results_df.to_csv(f"{output_dir}/comparison_results.csv", index=False)
    
    # Save rankings
    rankings.to_csv(f"{output_dir}/algorithm_rankings.csv", index=False)
    
    print(f"   ✅ Saved comparison visualizations to {output_dir}/")


def generate_text_report(results_df: pd.DataFrame, output_dir: str):
    """Generate detailed text report"""
    
    report = f"""
================================================================================
                    RECOMMENDATION ALGORITHMS - COMPARISON REPORT
================================================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

================================================================================
                              EXECUTIVE SUMMARY
================================================================================

This report compares {len(results_df)} recommendation algorithms across multiple
evaluation metrics including rating prediction accuracy (RMSE, MAE) and ranking
quality (Precision, Recall, NDCG, Coverage).

BEST PERFORMERS:
----------------
• Best RMSE (Rating Prediction): {results_df.loc[results_df['RMSE'].idxmin(), 'Algorithm']} ({results_df['RMSE'].min():.4f})
• Best Precision@10 (Ranking): {results_df.loc[results_df['Precision@10'].idxmax(), 'Algorithm']} ({results_df['Precision@10'].max():.4f})
• Best Coverage (Diversity): {results_df.loc[results_df['Coverage'].idxmax(), 'Algorithm']} ({results_df['Coverage'].max():.4f})
• Fastest Training: {results_df.loc[results_df['Train Time (s)'].idxmin(), 'Algorithm']} ({results_df['Train Time (s)'].min():.2f}s)

================================================================================
                              DETAILED RESULTS
================================================================================

"""
    
    # Add detailed results for each algorithm
    for _, row in results_df.iterrows():
        report += f"""
{row['Algorithm']}
{'-' * len(row['Algorithm'])}
Training Time: {row['Train Time (s)']}s

Rating Prediction:
  • RMSE: {row['RMSE']}
  • MAE: {row['MAE']}

Ranking Quality:
  • Precision@5:  {row['Precision@5']}    Recall@5:  {row['Recall@5']}    NDCG@5:  {row['NDCG@5']}
  • Precision@10: {row['Precision@10']}    Recall@10: {row['Recall@10']}    NDCG@10: {row['NDCG@10']}
  • Precision@20: {row['Precision@20']}    Recall@20: {row['Recall@20']}    NDCG@20: {row['NDCG@20']}

Diversity:
  • Catalog Coverage: {row['Coverage']:.2%}

"""
    
    report += """
================================================================================
                              METRIC DEFINITIONS
================================================================================

• RMSE (Root Mean Square Error): Measures rating prediction accuracy.
  Lower values indicate better predictions.

• MAE (Mean Absolute Error): Average absolute difference between predicted
  and actual ratings. Lower is better.

• Precision@K: Fraction of recommended items that are relevant.
  Higher values indicate more accurate recommendations.

• Recall@K: Fraction of relevant items that were recommended.
  Higher values indicate better coverage of user interests.

• NDCG@K (Normalized Discounted Cumulative Gain): Measures ranking quality,
  giving higher weight to relevant items appearing earlier.

• Coverage: Percentage of catalog items that can be recommended.
  Higher values indicate better item diversity.

================================================================================
                              RECOMMENDATIONS
================================================================================

For Production Deployment:
1. If accuracy is priority: Use the algorithm with lowest RMSE
2. If coverage/diversity matters: Consider algorithms with higher coverage
3. For cold-start scenarios: Popularity-based methods provide good baselines
4. For real-time systems: Consider training time and inference speed

================================================================================
"""
    
    with open(f"{output_dir}/comparison_report.txt", 'w') as f:
        f.write(report)
    
    print(f"   ✅ Saved text report to {output_dir}/comparison_report.txt")


def main():
    """Main execution"""
    print("="*70)
    print("MASTER COMPARISON - ALL RECOMMENDATION ALGORITHMS")
    print("="*70)
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Create import aliases
    import shutil
    renames = [
        ('01_collaborative_filtering.py', 'collaborative_filtering_01.py'),
        ('02_matrix_factorization.py', 'matrix_factorization_02.py'),
        ('03_content_based.py', 'content_based_03.py'),
        ('04_graph_based.py', 'graph_based_04.py'),
        ('08_popularity_baseline.py', 'popularity_baseline_08.py'),
    ]
    
    for old, new in renames:
        if os.path.exists(old) and not os.path.exists(new):
            shutil.copy(old, new)
    
    # Generate dataset
    print("\n📦 Generating dataset...")
    generator = DatasetGenerator(output_dir="../data")
    dataset = generator.generate_movielens_style(
        n_users=500,
        n_items=1000,
        n_ratings=50000
    )
    
    ratings_df = dataset['ratings']
    items_df = dataset['items']
    
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
    print(f"   Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Train all algorithms
    print("\n" + "="*60)
    print("TRAINING ALL ALGORITHMS")
    print("="*60)
    
    benchmarks = train_all_algorithms(train_df, items_df)
    
    # Evaluate all algorithms
    results_df = evaluate_all_algorithms(benchmarks, test_df)
    
    # Generate visualizations
    generate_comparison_visualizations(results_df, "../reports/master_comparison")
    
    # Generate text report
    generate_text_report(results_df, "../reports/master_comparison")
    
    # Print summary table
    print("\n" + "="*70)
    print("COMPARISON RESULTS SUMMARY")
    print("="*70)
    print(results_df.to_string(index=False))
    
    print("\n" + "="*70)
    print("✅ MASTER COMPARISON COMPLETE!")
    print("="*70)
    print(f"\nOutputs saved to: ../reports/master_comparison/")
    print("  • comparison_results.csv")
    print("  • algorithm_comparison.png")
    print("  • comparison_report.txt")
    print("  • algorithm_rankings.csv")


if __name__ == "__main__":
    main()
