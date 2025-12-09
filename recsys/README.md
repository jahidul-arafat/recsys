# Recommendation Systems Analysis Framework

## NeurIPS-Quality Research Framework & Interactive Dashboard

A comprehensive Python framework for implementing, evaluating, and comparing recommendation system algorithms across multiple datasets, with an interactive research dashboard.

---

## 🚀 Quick Start

### Option 1: View Dashboard Only
Simply open `dashboard_neurips_enhanced.html` in any modern web browser.

### Option 2: Run Full Pipeline
```bash
# Install dependencies
pip install -r requirements.txt

# Run complete analysis pipeline
cd scripts
python run_all.py

# Or run individual components
python 01_collaborative_filtering.py
python 02_matrix_factorization.py
# ... etc
```

---

## 📁 Project Structure

```
recsys_dashboard_project/
│
├── dashboard_neurips_enhanced.html   # 📊 Interactive research dashboard (208KB)
├── requirements.txt                  # Python dependencies
├── commands.txt                      # Useful command reference
├── README.md                         # This file
│
└── scripts/                          # 🐍 All Python implementations
    │
    ├── ALGORITHM IMPLEMENTATIONS
    │   ├── 01_collaborative_filtering.py     # User-CF & Item-CF
    │   ├── 02_matrix_factorization.py        # SVD, ALS, Funk SVD, SVD++
    │   ├── 03_content_based.py               # TF-IDF content filtering
    │   ├── 04_graph_based.py                 # PageRank & Random Walk
    │   ├── 05_neural_cf.py                   # Neural Collaborative Filtering
    │   ├── 06_hybrid.py                      # Weighted, Switching, Cascade
    │   ├── 07_association_rules.py           # Apriori & FP-Growth
    │   └── 08_popularity_baseline.py         # Popularity baselines
    │
    ├── ANALYSIS & EVALUATION
    │   ├── 09_hyperparameter_tuning.py       # Grid search with CV
    │   ├── 10_master_comparison.py           # Statistical comparisons
    │   ├── 11_generate_dashboard.py          # Dashboard HTML generator
    │   └── 12_multi_dataset_evaluation.py    # Cross-dataset evaluation
    │
    └── UTILITIES
        ├── data_loader.py                    # Dataset loading & preprocessing
        ├── visualization_utils.py            # Plotting & visualization helpers
        └── run_all.py                        # Master execution script
```

---

## 📊 Algorithms Implemented

| # | Algorithm | File | Category |
|---|-----------|------|----------|
| 1 | Collaborative Filtering | `01_collaborative_filtering.py` | Memory-Based |
| 2 | Matrix Factorization | `02_matrix_factorization.py` | Model-Based |
| 3 | Content-Based | `03_content_based.py` | Feature-Based |
| 4 | Graph-Based | `04_graph_based.py` | Network-Based |
| 5 | Neural CF | `05_neural_cf.py` | Deep Learning |
| 6 | Hybrid | `06_hybrid.py` | Ensemble |
| 7 | Association Rules | `07_association_rules.py` | Pattern Mining |
| 8 | Popularity | `08_popularity_baseline.py` | Statistical |

---

## 📈 Datasets Supported

| Dataset | Users | Items | Ratings | Sparsity |
|---------|-------|-------|---------|----------|
| MovieLens-100K | 943 | 1,682 | 100,000 | 93.7% |
| Amazon-Style | 2,000 | 1,000 | 28,000 | 98.6% |
| BookCrossing-Style | 3,000 | 2,000 | 40,000 | 99.3% |

---

## 🎯 Evaluation Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| RMSE | Root Mean Square Error | Lower ↓ |
| MAE | Mean Absolute Error | Lower ↓ |
| NDCG@k | Normalized DCG | Higher ↑ |
| Precision@k | Precision at k | Higher ↑ |
| Recall@k | Recall at k | Higher ↑ |
| Coverage | Catalog coverage | Higher ↑ |

---

## 🔬 Dashboard Features

### 5 Interactive Pages
1. **Overview** - Executive summary, key metrics
2. **Visualizations** - 16 algorithm cards with explanations
3. **Comparison** - Statistical tests, effect sizes
4. **Results** - Dynamic tables with dataset switching
5. **Hyperparameters** - NeurIPS-quality analysis with playground

### Hyperparameter Analysis
- **8 Research Hypotheses** with statistical testing
- **Dynamic Sensitivity Charts** - Update by dataset
- **Interaction Heatmaps** - Parameter interactions
- **3D Response Surface** - Optimization landscape
- **Pareto Frontier** - Multi-objective trade-offs
- **Interactive Playground** - Real-time simulation

---

## 🛠️ Requirements

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=0.24.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
torch>=1.9.0
networkx>=2.6.0
mlxtend>=0.18.0
surprise>=1.1.1
tqdm>=4.61.0
```

---

## 📝 Running Scripts

```bash
cd scripts

# Run all algorithms
python run_all.py

# Individual algorithms
python 01_collaborative_filtering.py
python 02_matrix_factorization.py
python 05_neural_cf.py

# Hyperparameter tuning
python 09_hyperparameter_tuning.py

# Generate comparison report
python 10_master_comparison.py

# Regenerate dashboard
python 11_generate_dashboard.py
```

---

## 📄 License

Research/Educational Use

**Version**: 2.0 (December 2024)
