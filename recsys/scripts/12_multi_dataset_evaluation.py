"""
12_multi_dataset_evaluation.py
===============================
Run ALL algorithms on ALL datasets with comprehensive reporting

Datasets:
1. MovieLens-Style (500 users, 1000 items) - Dense, explicit ratings
2. Amazon-Style (5000 users, 2000 items) - Sparse, with text reviews
3. BookCrossing-Style (10000 users, 5000 items) - Very sparse, cold-start heavy

Algorithms:
1. Collaborative Filtering (User-based, Item-based)
2. Matrix Factorization (SVD, ALS, Funk SVD)
3. Content-Based
4. Graph-Based (PageRank, Random Walk)
5. Popularity Baseline (Count, Rating, Weighted, Bayesian)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
data_loader = import_module('data_loader')
DatasetGenerator = data_loader.DatasetGenerator

# Output directories
os.makedirs("../reports/multi_dataset", exist_ok=True)


def load_algorithm(name: str):
    """Dynamically load algorithm class"""
    from importlib import import_module
    
    modules = {
        'CollaborativeFilter': ('01_collaborative_filtering', 'CollaborativeFilter'),
        'MatrixFactorization': ('02_matrix_factorization', 'MatrixFactorization'),
        'ContentBasedRecommender': ('03_content_based', 'ContentBasedRecommender'),
        'GraphBasedRecommender': ('04_graph_based', 'GraphBasedRecommender'),
        'PopularityRecommender': ('08_popularity_baseline', 'PopularityRecommender'),
    }
    
    module_name, class_name = modules[name]
    mod = import_module(module_name)
    return getattr(mod, class_name)


def evaluate_on_dataset(dataset_name: str, dataset: Dict, algorithms_config: List[Dict]) -> List[Dict]:
    """Evaluate all algorithms on a single dataset"""
    
    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name}")
    print(f"{'='*60}")
    
    ratings_df = dataset['ratings']
    items_df = dataset.get('items', None)
    
    print(f"   Users: {ratings_df['user_id'].nunique()}")
    print(f"   Items: {ratings_df['item_id'].nunique()}")
    print(f"   Ratings: {len(ratings_df)}")
    
    # Calculate sparsity
    n_users = ratings_df['user_id'].nunique()
    n_items = ratings_df['item_id'].nunique()
    sparsity = 1 - (len(ratings_df) / (n_users * n_items))
    print(f"   Sparsity: {sparsity:.2%}")
    
    # Split data
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
    print(f"   Train: {len(train_df)}, Test: {len(test_df)}")
    
    results = []
    
    for config in algorithms_config:
        algo_name = config['name']
        algo_class_name = config['class']
        algo_params = config.get('params', {})
        needs_items = config.get('needs_items', False)
        
        print(f"\n   📊 {algo_name}...")
        
        try:
            AlgoClass = load_algorithm(algo_class_name)
            model = AlgoClass(**algo_params)
            
            start_time = time.time()
            
            if needs_items and items_df is not None:
                model.fit(items_df, train_df, feature_columns=['genres'])
            else:
                model.fit(train_df)
            
            train_time = time.time() - start_time
            
            # Evaluate
            metrics = model.evaluate(test_df, k_values=[5, 10, 20])
            
            results.append({
                'dataset': dataset_name,
                'algorithm': algo_name,
                'train_time': round(train_time, 3),
                'sparsity': round(sparsity, 4),
                **metrics
            })
            
            print(f"      ✅ RMSE: {metrics['RMSE']:.4f}, NDCG@10: {metrics['NDCG@10']:.4f}")
            
        except Exception as e:
            print(f"      ❌ Error: {e}")
            results.append({
                'dataset': dataset_name,
                'algorithm': algo_name,
                'train_time': None,
                'sparsity': round(sparsity, 4),
                'RMSE': None,
                'MAE': None,
                'NDCG@10': None,
                'error': str(e)
            })
    
    return results


def generate_multi_dataset_report(all_results: pd.DataFrame, output_dir: str):
    """Generate comprehensive multi-dataset report"""
    
    print("\n📊 Generating multi-dataset report...")
    
    # Save raw results
    all_results.to_csv(f"{output_dir}/multi_dataset_results.csv", index=False)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Filter out failed runs
    valid_results = all_results.dropna(subset=['RMSE'])
    
    # 1. RMSE by Algorithm and Dataset
    ax1 = axes[0, 0]
    pivot_rmse = valid_results.pivot_table(index='algorithm', columns='dataset', values='RMSE', aggfunc='mean')
    pivot_rmse.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_ylabel('RMSE (lower is better)')
    ax1.set_title('RMSE by Algorithm Across Datasets', fontsize=14)
    ax1.legend(title='Dataset', bbox_to_anchor=(1.02, 1))
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. NDCG@10 by Algorithm and Dataset
    ax2 = axes[0, 1]
    pivot_ndcg = valid_results.pivot_table(index='algorithm', columns='dataset', values='NDCG@10', aggfunc='mean')
    pivot_ndcg.plot(kind='bar', ax=ax2, width=0.8, colormap='viridis')
    ax2.set_ylabel('NDCG@10 (higher is better)')
    ax2.set_title('NDCG@10 by Algorithm Across Datasets', fontsize=14)
    ax2.legend(title='Dataset', bbox_to_anchor=(1.02, 1))
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Training Time Comparison
    ax3 = axes[1, 0]
    pivot_time = valid_results.pivot_table(index='algorithm', columns='dataset', values='train_time', aggfunc='mean')
    pivot_time.plot(kind='bar', ax=ax3, width=0.8, colormap='plasma')
    ax3.set_ylabel('Training Time (seconds)')
    ax3.set_title('Training Time by Algorithm Across Datasets', fontsize=14)
    ax3.legend(title='Dataset', bbox_to_anchor=(1.02, 1))
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Dataset Characteristics Impact
    ax4 = axes[1, 1]
    dataset_stats = valid_results.groupby('dataset').agg({
        'sparsity': 'first',
        'RMSE': 'mean',
        'NDCG@10': 'mean'
    }).reset_index()
    
    x = range(len(dataset_stats))
    width = 0.25
    
    ax4.bar([i - width for i in x], dataset_stats['sparsity'], width, label='Sparsity', color='#3498db')
    ax4.bar([i for i in x], dataset_stats['RMSE'], width, label='Avg RMSE', color='#e74c3c')
    ax4.bar([i + width for i in x], dataset_stats['NDCG@10'] * 10, width, label='Avg NDCG×10', color='#2ecc71')
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(dataset_stats['dataset'], rotation=45, ha='right')
    ax4.set_ylabel('Value')
    ax4.set_title('Dataset Characteristics vs Performance', fontsize=14)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/multi_dataset_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    heatmap_data = valid_results.pivot_table(index='algorithm', columns='dataset', values='RMSE', aggfunc='mean')
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=ax, 
                cbar_kws={'label': 'RMSE'})
    ax.set_title('RMSE Heatmap: Algorithms × Datasets', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rmse_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Generate text report
    report = f"""
================================================================================
MULTI-DATASET EVALUATION REPORT
================================================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

================================================================================
DATASETS USED
================================================================================

1. MovieLens-Style Dataset
   - Users: 500
   - Items: 1,000
   - Ratings: ~30,000
   - Sparsity: ~94%
   - Characteristics: Dense explicit ratings (1-5 scale)
   - Best for: Collaborative Filtering, Matrix Factorization

2. Amazon-Style Dataset
   - Users: 2,000
   - Items: 1,000
   - Ratings: ~20,000
   - Sparsity: ~99%
   - Characteristics: Sparse with text reviews
   - Best for: Content-Based, Hybrid methods

3. BookCrossing-Style Dataset
   - Users: 3,000
   - Items: 2,000
   - Ratings: ~30,000
   - Sparsity: ~99.5%
   - Characteristics: Very sparse, many cold-start users
   - Best for: Graph-Based, Popularity Baseline

================================================================================
ALGORITHMS TESTED
================================================================================

1. User-Based CF: Find similar users, recommend their favorites
2. Item-Based CF: Find similar items to user's history
3. MF-SVD: Matrix decomposition via Singular Value Decomposition
4. MF-ALS: Alternating Least Squares optimization
5. Content-Based: TF-IDF features + user profiles
6. Graph-PageRank: Personalized PageRank on user-item graph
7. Popularity-Weighted: Blend of popularity and rating
8. Popularity-Bayesian: IMDB-style weighted average

================================================================================
RESULTS SUMMARY
================================================================================

"""
    
    # Add per-dataset summary
    for dataset in valid_results['dataset'].unique():
        ds_results = valid_results[valid_results['dataset'] == dataset]
        
        report += f"\n--- {dataset} ---\n"
        report += f"Sparsity: {ds_results['sparsity'].iloc[0]:.2%}\n"
        report += f"Best RMSE: {ds_results['RMSE'].min():.4f} ({ds_results.loc[ds_results['RMSE'].idxmin(), 'algorithm']})\n"
        report += f"Best NDCG@10: {ds_results['NDCG@10'].max():.4f} ({ds_results.loc[ds_results['NDCG@10'].idxmax(), 'algorithm']})\n"
    
    # Add overall rankings
    report += "\n================================================================================\n"
    report += "OVERALL ALGORITHM RANKINGS (averaged across datasets)\n"
    report += "================================================================================\n\n"
    
    avg_metrics = valid_results.groupby('algorithm').agg({
        'RMSE': 'mean',
        'NDCG@10': 'mean',
        'train_time': 'mean'
    }).round(4)
    
    report += "By RMSE (lower is better):\n"
    for i, (algo, row) in enumerate(avg_metrics.sort_values('RMSE').iterrows(), 1):
        report += f"  {i}. {algo}: {row['RMSE']:.4f}\n"
    
    report += "\nBy NDCG@10 (higher is better):\n"
    for i, (algo, row) in enumerate(avg_metrics.sort_values('NDCG@10', ascending=False).iterrows(), 1):
        report += f"  {i}. {algo}: {row['NDCG@10']:.4f}\n"
    
    report += "\n================================================================================\n"
    
    with open(f"{output_dir}/multi_dataset_report.txt", 'w') as f:
        f.write(report)
    
    # Save JSON summary
    summary = {
        'generated': datetime.now().isoformat(),
        'datasets': list(valid_results['dataset'].unique()),
        'algorithms': list(valid_results['algorithm'].unique()),
        'results_by_dataset': {},
        'overall_rankings': {
            'by_rmse': avg_metrics.sort_values('RMSE').index.tolist(),
            'by_ndcg': avg_metrics.sort_values('NDCG@10', ascending=False).index.tolist()
        }
    }
    
    for dataset in valid_results['dataset'].unique():
        ds_results = valid_results[valid_results['dataset'] == dataset]
        summary['results_by_dataset'][dataset] = ds_results.to_dict('records')
    
    with open(f"{output_dir}/multi_dataset_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"   ✅ Reports saved to {output_dir}/")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-dataset evaluation')
    parser.add_argument('--real', action='store_true', help='Use real MovieLens-100K dataset')
    args = parser.parse_args()
    
    print("=" * 70)
    print("MULTI-DATASET EVALUATION - ALL ALGORITHMS × ALL DATASETS")
    print("=" * 70)
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    generator = DatasetGenerator(output_dir="../data")
    
    # Generate all three datasets
    print("\n📦 Loading datasets...")
    
    # Dataset 1: MovieLens (real or synthetic)
    if args.real:
        print("\n   1. MovieLens 100K (REAL dataset)...")
        movielens = generator.load_real_movielens_100k()
    else:
        print("\n   1. MovieLens-Style (synthetic, dense explicit ratings)...")
        movielens = generator.generate_movielens_style(
            n_users=500,
            n_items=1000,
            n_ratings=50000
        )
    
    # Dataset 2: Amazon-style (synthetic)
    print("\n   2. Amazon-Style (sparse, with reviews)...")
    amazon_raw = generator.generate_amazon_style(
        n_users=2000,
        n_products=1000,
        n_reviews=30000
    )
    # Normalize to standard format
    amazon = {
        'ratings': amazon_raw['reviews'].rename(columns={'product_id': 'item_id'}),
        'items': amazon_raw['products'].rename(columns={'product_id': 'item_id'})
    }
    
    # Dataset 3: BookCrossing-style (synthetic, very sparse)
    print("\n   3. BookCrossing-Style (very sparse, cold-start)...")
    bookcrossing_raw = generator.generate_bookcrossing_style(
        n_users=3000,
        n_books=2000,
        n_ratings=40000
    )
    # Normalize and filter to explicit ratings only (rating > 0)
    bc_ratings = bookcrossing_raw['ratings'].rename(columns={'book_id': 'item_id'})
    bc_ratings = bc_ratings[bc_ratings['rating'] > 0].copy()
    bookcrossing = {
        'ratings': bc_ratings,
        'items': bookcrossing_raw['books'].rename(columns={'book_id': 'item_id'})
    }
    
    datasets = {
        'MovieLens' + ('-100K' if args.real else '-Style'): movielens,
        'Amazon-Style': amazon,
        'BookCrossing-Style': bookcrossing
    }
    
    # Define algorithms to test
    algorithms = [
        {'name': 'User-Based CF', 'class': 'CollaborativeFilter', 
         'params': {'method': 'user', 'k_neighbors': 20}},
        {'name': 'Item-Based CF', 'class': 'CollaborativeFilter', 
         'params': {'method': 'item', 'k_neighbors': 20}},
        {'name': 'MF-SVD', 'class': 'MatrixFactorization', 
         'params': {'method': 'svd', 'n_factors': 30}},
        {'name': 'MF-ALS', 'class': 'MatrixFactorization', 
         'params': {'method': 'als', 'n_factors': 30, 'n_epochs': 15, 'regularization': 0.1}},
        {'name': 'Content-Based', 'class': 'ContentBasedRecommender', 
         'params': {'tfidf_max_features': 500}, 'needs_items': True},
        {'name': 'Graph-PageRank', 'class': 'GraphBasedRecommender', 
         'params': {'method': 'pagerank', 'damping_factor': 0.85}},
        {'name': 'Popularity-Weighted', 'class': 'PopularityRecommender', 
         'params': {'method': 'weighted', 'popularity_weight': 0.5}},
        {'name': 'Popularity-Bayesian', 'class': 'PopularityRecommender', 
         'params': {'method': 'bayesian'}},
    ]
    
    # Evaluate on all datasets
    all_results = []
    
    for dataset_name, dataset in datasets.items():
        results = evaluate_on_dataset(dataset_name, dataset, algorithms)
        all_results.extend(results)
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Generate report
    generate_multi_dataset_report(results_df, "../reports/multi_dataset")
    
    print("\n" + "=" * 70)
    print("✅ MULTI-DATASET EVALUATION COMPLETE!")
    print("=" * 70)
    
    # Print summary table
    print("\n📊 RESULTS SUMMARY:")
    print("-" * 100)
    summary = results_df.pivot_table(
        index='algorithm', 
        columns='dataset', 
        values=['RMSE', 'NDCG@10'], 
        aggfunc='mean'
    ).round(4)
    print(summary.to_string())
    
    return results_df


if __name__ == "__main__":
    main()
