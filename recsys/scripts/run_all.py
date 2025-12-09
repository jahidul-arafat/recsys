#!/usr/bin/env python3
"""
================================================================================
RECOMMENDATION SYSTEM - COMPREHENSIVE EVALUATION FRAMEWORK
================================================================================
A professional academic evaluation pipeline for recommendation algorithms.

This script provides:
- Interactive execution planning with user confirmation
- Detailed explanations of each algorithm and evaluation phase
- Color-coded console output for clarity
- Option to skip specific steps or run all at once
- Comprehensive progress tracking and timing

Author: Recommendation Systems Research Project
Version: 2.0
================================================================================
"""

import subprocess
import sys
import os
import argparse
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional


# =============================================================================
# ANSI COLOR CODES FOR TERMINAL OUTPUT
# =============================================================================
class Colors:
    """ANSI escape codes for colored terminal output"""
    # Text colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    
    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    
    # Reset
    RESET = '\033[0m'
    
    @classmethod
    def disable(cls):
        """Disable colors (for non-TTY outputs)"""
        for attr in dir(cls):
            if not attr.startswith('_') and attr.isupper():
                setattr(cls, attr, '')


# Check if terminal supports colors
if not sys.stdout.isatty():
    Colors.disable()


# =============================================================================
# ALGORITHM DEFINITIONS WITH ACADEMIC DESCRIPTIONS
# =============================================================================
ALGORITHMS = {
    'collaborative_filtering': {
        'script': '01_collaborative_filtering.py',
        'name': 'Collaborative Filtering',
        'short': 'CF',
        'category': 'Memory-Based',
        'methods': ['User-User kNN', 'Item-Item kNN'],
        'description': """
    Collaborative Filtering is the foundational recommendation technique that 
    leverages the collective behavior of users to make predictions. It operates
    on the principle that users who agreed in the past will agree in the future.
    
    • User-User CF: Finds users with similar rating patterns and recommends 
      items that similar users liked but the target user hasn't seen.
    • Item-Item CF: Finds items that are rated similarly by users and 
      recommends items similar to those the user has liked.
    
    Key Metrics: Cosine Similarity, Pearson Correlation
    Complexity: O(n²) for similarity computation
    Best For: Dense rating matrices with explicit feedback""",
        'timeout': 300,
        'phase': 1
    },
    
    'matrix_factorization': {
        'script': '02_matrix_factorization.py',
        'name': 'Matrix Factorization',
        'short': 'MF',
        'category': 'Model-Based',
        'methods': ['SVD', 'ALS', 'Funk SVD'],
        'description': """
    Matrix Factorization decomposes the user-item rating matrix into lower-
    dimensional latent factor matrices. This captures hidden features that
    explain user preferences and item characteristics.
    
    • SVD: Singular Value Decomposition - direct matrix decomposition
    • ALS: Alternating Least Squares - iterative optimization, handles 
      implicit feedback well
    • Funk SVD: Stochastic Gradient Descent approach - scalable, handles
      missing values naturally
    
    Mathematical Form: R ≈ U × V^T (where U=user factors, V=item factors)
    Complexity: O(k × iterations × nnz) where k=latent factors, nnz=non-zeros
    Best For: Large sparse matrices, capturing latent patterns""",
        'timeout': 400,
        'phase': 1
    },
    
    'content_based': {
        'script': '03_content_based.py',
        'name': 'Content-Based Filtering',
        'short': 'CB',
        'category': 'Content-Based',
        'methods': ['TF-IDF', 'Feature Vectors', 'User Profiles'],
        'description': """
    Content-Based Filtering recommends items similar to those a user has 
    liked in the past, based on item features rather than user behavior.
    It builds a profile of user preferences from item attributes.
    
    • TF-IDF: Term Frequency-Inverse Document Frequency for text features
    • Feature Vectors: Multi-hot encoding of categorical attributes (genres)
    • User Profiles: Weighted average of liked item features
    
    Similarity: Cosine similarity between user profile and item features
    Advantages: No cold-start for items, explainable recommendations
    Best For: Rich item metadata, new item recommendations""",
        'timeout': 300,
        'phase': 1
    },
    
    'graph_based': {
        'script': '04_graph_based.py',
        'name': 'Graph-Based Methods',
        'short': 'Graph',
        'category': 'Graph-Based',
        'methods': ['Personalized PageRank', 'Random Walk with Restart'],
        'description': """
    Graph-Based methods model users and items as nodes in a bipartite graph,
    with edges representing interactions. Recommendations are generated by
    propagating influence through the graph structure.
    
    • Personalized PageRank: Adapts Google's PageRank with user-specific 
      restart probability, measuring item importance relative to user
    • Random Walk with Restart: Simulates random surfer starting from user,
      items visited frequently are recommended
    
    Graph Structure: G = (U ∪ I, E) where U=users, I=items, E=interactions
    Damping Factor: α (typically 0.85) controls exploration vs exploitation
    Best For: Capturing transitive relationships, path-based explanations""",
        'timeout': 400,
        'phase': 1
    },
    
    'neural_cf': {
        'script': '05_neural_cf.py',
        'name': 'Neural Collaborative Filtering',
        'short': 'NCF',
        'category': 'Deep Learning',
        'methods': ['GMF', 'MLP', 'NeuMF'],
        'description': """
    Neural Collaborative Filtering uses deep learning to model complex 
    user-item interactions beyond linear matrix factorization. It learns
    non-linear feature interactions through neural network architectures.
    
    • GMF: Generalized Matrix Factorization - neural version of MF with
      element-wise product of embeddings
    • MLP: Multi-Layer Perceptron - concatenates embeddings and learns
      interactions through hidden layers
    • NeuMF: Neural Matrix Factorization - combines GMF and MLP for both
      linear and non-linear modeling
    
    Architecture: Embedding → Interaction → Hidden Layers → Prediction
    Training: Binary cross-entropy or MSE loss with Adam optimizer
    Best For: Large-scale data, capturing complex non-linear patterns""",
        'timeout': 600,
        'phase': 1,
        'requires_pytorch': True
    },
    
    'hybrid': {
        'script': '06_hybrid.py',
        'name': 'Hybrid Recommender',
        'short': 'Hybrid',
        'category': 'Ensemble',
        'methods': ['Weighted', 'Switching', 'Cascade'],
        'description': """
    Hybrid Recommenders combine multiple recommendation strategies to 
    leverage the strengths of each approach while mitigating weaknesses.
    
    • Weighted: Linear combination of predictions from multiple models
      Final_Score = Σ(wᵢ × Scoreᵢ) where Σwᵢ = 1
    • Switching: Selects algorithm based on context (e.g., cold-start 
      users get content-based, established users get CF)
    • Cascade: Uses one algorithm to produce coarse ranking, another 
      to refine among top candidates
    
    Cold-Start Handling: Automatically switches to content/popularity
    Best For: Production systems, handling diverse user scenarios""",
        'timeout': 400,
        'phase': 1
    },
    
    'association_rules': {
        'script': '07_association_rules.py',
        'name': 'Association Rules',
        'short': 'AR',
        'category': 'Pattern Mining',
        'methods': ['Apriori', 'FP-Growth'],
        'description': """
    Association Rules mining discovers frequently co-occurring item sets
    in transaction data, then generates rules of the form {A} → {B}.
    Originally developed for market basket analysis.
    
    • Apriori: Bottom-up approach, generates candidate itemsets level by 
      level, prunes using minimum support threshold
    • FP-Growth: Constructs FP-tree data structure, mines patterns without
      candidate generation - more efficient
    
    Key Metrics:
      - Support: P(A ∪ B) - frequency of itemset
      - Confidence: P(B|A) - reliability of rule
      - Lift: P(B|A)/P(B) - strength beyond random chance
    
    Best For: Transaction data, "customers who bought X also bought Y" """,
        'timeout': 300,
        'phase': 1
    },
    
    'popularity_baseline': {
        'script': '08_popularity_baseline.py',
        'name': 'Popularity Baseline',
        'short': 'Pop',
        'category': 'Baseline',
        'methods': ['Count', 'Rating', 'Weighted', 'Bayesian'],
        'description': """
    Popularity-based methods serve as essential baselines and are 
    surprisingly effective in practice. They recommend globally or 
    contextually popular items.
    
    • Count-Based: Ranks by number of ratings/interactions
    • Rating-Based: Ranks by average rating score
    • Weighted: Combines count and rating: score = α×count + (1-α)×rating
    • Bayesian Average: IMDB-style weighted mean that pulls ratings toward
      global mean for items with few ratings:
      Bayesian = (v×R + m×C)/(v+m) where v=votes, R=rating, m=min_votes, C=mean
    
    Use Case: Cold-start, sanity check, production fallback
    Best For: New users, trending items, baseline comparison""",
        'timeout': 200,
        'phase': 1
    }
}

EVALUATION_PHASES = {
    'hyperparameter_tuning': {
        'script': '09_hyperparameter_tuning.py',
        'name': 'Hyperparameter Tuning',
        'short': 'HP-Tune',
        'description': """
    Systematic exploration of hyperparameter space for each algorithm
    using grid search with cross-validation.
    
    Parameters Tuned:
    • CF: k_neighbors ∈ {5,10,20,30,50}, similarity ∈ {cosine, pearson}
    • MF: n_factors ∈ {10,20,50,100}, regularization ∈ {0.01,0.02,0.05,0.1}
    • Content: tfidf_features ∈ {100,300,500,1000}, ngram ∈ {(1,1),(1,2),(1,3)}
    • Graph: damping ∈ {0.75, 0.85, 0.95}
    
    Output: Best configurations, sensitivity analysis, performance curves""",
        'timeout': 1800,
        'phase': 2
    },
    
    'master_comparison': {
        'script': '10_master_comparison.py',
        'name': 'Master Comparison',
        'short': 'Compare',
        'description': """
    Head-to-head comparison of all trained algorithms on identical 
    train/test splits for fair evaluation.
    
    Metrics Computed:
    • Prediction: RMSE, MAE (rating accuracy)
    • Ranking: Precision@K, Recall@K, NDCG@K (top-K quality)
    • Coverage: Catalog coverage (recommendation diversity)
    • Efficiency: Training time, prediction latency
    
    Output: Comparison tables, radar charts, statistical significance tests""",
        'timeout': 600,
        'phase': 4
    },
    
    'multi_dataset': {
        'script': '12_multi_dataset_evaluation.py',
        'name': 'Multi-Dataset Evaluation',
        'short': 'Multi-DS',
        'description': """
    Evaluates algorithm robustness across datasets with different 
    characteristics to test generalization.
    
    Datasets:
    1. MovieLens (Real/Synthetic): Dense explicit ratings, ~6% density
    2. Amazon-Style: Sparse with text reviews, ~0.5% density  
    3. BookCrossing-Style: Very sparse, cold-start heavy, ~0.02% density
    
    Analysis: How algorithm performance varies with sparsity, cold-start
    ratio, and rating distribution""",
        'timeout': 1800,
        'phase': 3
    },
    
    'dashboard': {
        'script': '11_generate_dashboard.py',
        'name': 'Dashboard Generation',
        'short': 'Dashboard',
        'description': """
    Generates interactive HTML dashboard consolidating all results.
    
    Contents:
    • Algorithm comparison charts (bar, radar, heatmap)
    • Per-algorithm detailed reports
    • Hyperparameter sensitivity plots
    • Multi-dataset performance matrix
    • Interactive Pyvis network visualizations
    
    Output: visualizations/dashboard.html""",
        'timeout': 120,
        'phase': 4
    }
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_header(title: str, width: int = 80):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_CYAN}{'═' * width}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}{title.center(width)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'═' * width}{Colors.RESET}")


def print_subheader(title: str, width: int = 80):
    """Print a formatted subheader"""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}{'─' * width}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.YELLOW}  {title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.YELLOW}{'─' * width}{Colors.RESET}")


def print_info(label: str, value: str, indent: int = 2):
    """Print an info line"""
    spaces = ' ' * indent
    print(f"{spaces}{Colors.CYAN}{label}:{Colors.RESET} {value}")


def print_success(message: str):
    """Print a success message"""
    print(f"{Colors.BRIGHT_GREEN}  ✓ {message}{Colors.RESET}")


def print_error(message: str):
    """Print an error message"""
    print(f"{Colors.BRIGHT_RED}  ✗ {message}{Colors.RESET}")


def print_warning(message: str):
    """Print a warning message"""
    print(f"{Colors.BRIGHT_YELLOW}  ⚠ {message}{Colors.RESET}")


def print_step(number: int, total: int, name: str, status: str = "pending"):
    """Print a step indicator"""
    status_colors = {
        'pending': Colors.DIM,
        'running': Colors.BRIGHT_YELLOW,
        'success': Colors.BRIGHT_GREEN,
        'failed': Colors.BRIGHT_RED,
        'skipped': Colors.DIM
    }
    status_icons = {
        'pending': '○',
        'running': '●',
        'success': '✓',
        'failed': '✗',
        'skipped': '◌'
    }
    color = status_colors.get(status, Colors.RESET)
    icon = status_icons.get(status, '?')
    print(f"  {color}[{icon}] Step {number}/{total}: {name}{Colors.RESET}")


def check_pytorch() -> bool:
    """Check if PyTorch is available"""
    try:
        import torch
        return True
    except ImportError:
        return False


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def get_user_confirmation(prompt: str, default: bool = True) -> bool:
    """Get yes/no confirmation from user"""
    default_str = "Y/n" if default else "y/N"
    try:
        response = input(f"{Colors.BRIGHT_WHITE}{prompt} [{default_str}]: {Colors.RESET}").strip().lower()
        if response == '':
            return default
        return response in ('y', 'yes')
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def run_script(script_name: str, description: str, extra_args: List[str] = None, 
               timeout: int = 600, verbose: bool = True) -> Tuple[bool, float]:
    """Run a Python script and return success status and elapsed time."""
    script_paths = [script_name, os.path.join('scripts', script_name)]
    script_path = None
    
    for path in script_paths:
        if os.path.exists(path):
            script_path = path
            break
    
    if script_path is None:
        if verbose:
            print_error(f"Script not found: {script_name}")
        return False, 0
    
    try:
        start_time = time.time()
        cmd = [sys.executable, script_path]
        if extra_args:
            cmd.extend(extra_args)
        
        result = subprocess.run(
            cmd,
            cwd=os.getcwd(),
            capture_output=False,
            timeout=timeout
        )
        elapsed = time.time() - start_time
        
        return result.returncode == 0, elapsed
        
    except subprocess.TimeoutExpired:
        if verbose:
            print_error(f"Timed out after {timeout}s")
        return False, timeout
    except Exception as e:
        if verbose:
            print_error(f"Error: {e}")
        return False, 0


# =============================================================================
# EXECUTION PLAN DISPLAY
# =============================================================================

def display_execution_plan(args, has_pytorch: bool) -> Dict[str, List[Dict]]:
    """Display the detailed execution plan and return the plan structure."""
    print_header("📋 EXECUTION PLAN")
    
    plan = {
        'phase1': [],  # Individual algorithms
        'phase2': [],  # Hyperparameter tuning
        'phase3': [],  # Multi-dataset evaluation
        'phase4': []   # Comparison & dashboard
    }
    
    # Configuration summary
    print(f"\n{Colors.BOLD}{Colors.WHITE}  Configuration:{Colors.RESET}")
    print(f"  {'─' * 60}")
    
    config_items = [
        ("Mode", f"{Colors.BRIGHT_GREEN}🚀 FULL EVALUATION{Colors.RESET}" if args.all else "Standard"),
        ("Neural Networks", f"{Colors.BRIGHT_GREEN}Yes ✓{Colors.RESET}" if (args.neural and has_pytorch) else (f"{Colors.DIM}Skipped (no PyTorch){Colors.RESET}" if args.neural else f"{Colors.DIM}No{Colors.RESET}")),
        ("Hyperparameter Tuning", f"{Colors.BRIGHT_GREEN}Yes ✓{Colors.RESET}" if args.full else f"{Colors.DIM}No{Colors.RESET}"),
    ]
    
    for label, value in config_items:
        print(f"  {Colors.CYAN}•{Colors.RESET} {label + ':':<25}{value}")
    
    # Dataset display
    if args.multi:
        print(f"\n  {Colors.BOLD}{Colors.WHITE}📊 Datasets (3):{Colors.RESET}")
        print(f"  {'─' * 60}")
        
        # MovieLens
        if args.real:
            print(f"  {Colors.BRIGHT_GREEN}•{Colors.RESET} {'MovieLens-100K':<20} {Colors.BRIGHT_GREEN}Real ✓{Colors.RESET} {Colors.DIM}(downloaded from GroupLens){Colors.RESET}")
        else:
            print(f"  {Colors.CYAN}•{Colors.RESET} {'MovieLens-Style':<20} {Colors.DIM}Synthetic (943 users, 1682 items){Colors.RESET}")
        
        # Amazon & BookCrossing (always synthetic)
        print(f"  {Colors.CYAN}•{Colors.RESET} {'Amazon-Style':<20} {Colors.DIM}Synthetic (5000 users, 2000 items){Colors.RESET}")
        print(f"  {Colors.CYAN}•{Colors.RESET} {'BookCrossing-Style':<20} {Colors.DIM}Synthetic (10000 users, 5000 items){Colors.RESET}")
    else:
        print(f"\n  {Colors.BOLD}{Colors.WHITE}📊 Dataset:{Colors.RESET}")
        print(f"  {'─' * 60}")
        if args.real:
            print(f"  {Colors.BRIGHT_GREEN}•{Colors.RESET} {'MovieLens-100K':<20} {Colors.BRIGHT_GREEN}Real ✓{Colors.RESET} {Colors.DIM}(downloaded from GroupLens){Colors.RESET}")
        else:
            print(f"  {Colors.CYAN}•{Colors.RESET} {'MovieLens-Style':<20} {Colors.DIM}Synthetic (500 users, 1000 items){Colors.RESET}")
    
    # Phase 1: Individual Algorithms
    print_subheader("PHASE 1: Individual Algorithm Analysis")
    print(f"\n{Colors.DIM}  Each algorithm will be trained and evaluated independently.{Colors.RESET}")
    print(f"{Colors.DIM}  This provides detailed insights into each method's behavior.{Colors.RESET}\n")
    
    step_num = 1
    
    if not args.multi_only:
        for algo_key, algo in ALGORITHMS.items():
            # Skip neural if not enabled or no PyTorch
            if algo.get('requires_pytorch') and (not args.neural or not has_pytorch):
                continue
            
            # Skip neural algorithms for non-neural runs
            if algo_key in ['neural_cf', 'hybrid'] and not args.neural:
                continue
            
            # Skip non-baseline algorithms for quick mode
            if args.quick and algo_key not in ['collaborative_filtering', 'popularity_baseline']:
                continue
                
            plan['phase1'].append({
                'key': algo_key,
                'step': step_num,
                **algo
            })
            
            print(f"  {Colors.BRIGHT_WHITE}[{step_num}]{Colors.RESET} {Colors.BOLD}{algo['name']}{Colors.RESET}")
            print(f"      {Colors.CYAN}Category:{Colors.RESET} {algo['category']}")
            print(f"      {Colors.CYAN}Methods:{Colors.RESET} {', '.join(algo['methods'])}")
            print(f"      {Colors.DIM}Est. time: ~{format_time(algo['timeout']/3)}{Colors.RESET}")
            print()
            step_num += 1
    else:
        print(f"  {Colors.DIM}(Skipped - running multi-dataset evaluation only){Colors.RESET}\n")
    
    # Phase 2: Hyperparameter Tuning
    print_subheader("PHASE 2: Hyperparameter Tuning")
    
    if args.full:
        hp = EVALUATION_PHASES['hyperparameter_tuning']
        plan['phase2'].append({
            'key': 'hyperparameter_tuning',
            'step': step_num,
            **hp
        })
        print(f"\n{Colors.DIM}  Systematic grid search across hyperparameter configurations.{Colors.RESET}\n")
        print(f"  {Colors.BRIGHT_WHITE}[{step_num}]{Colors.RESET} {Colors.BOLD}{hp['name']}{Colors.RESET}")
        print(f"      {Colors.DIM}Est. time: ~{format_time(hp['timeout']/3)}{Colors.RESET}")
        step_num += 1
    else:
        print(f"\n  {Colors.DIM}(Skipped - use --full to enable){Colors.RESET}")
    print()
    
    # Phase 3: Multi-Dataset Evaluation
    print_subheader("PHASE 3: Multi-Dataset Evaluation")
    
    if args.multi:
        md = EVALUATION_PHASES['multi_dataset']
        plan['phase3'].append({
            'key': 'multi_dataset',
            'step': step_num,
            'real': args.real,
            **md
        })
        print(f"\n{Colors.DIM}  Testing algorithm robustness across different data characteristics.{Colors.RESET}\n")
        print(f"  {Colors.BRIGHT_WHITE}[{step_num}]{Colors.RESET} {Colors.BOLD}{md['name']}{Colors.RESET}")
        
        datasets = [
            ("MovieLens", "Real (100K)" if args.real else "Synthetic", "~6% density"),
            ("Amazon-Style", "Synthetic", "~0.5% density"),
            ("BookCrossing-Style", "Synthetic", "~0.02% density")
        ]
        for ds_name, ds_type, ds_density in datasets:
            print(f"      {Colors.CYAN}•{Colors.RESET} {ds_name} ({ds_type}) - {ds_density}")
        
        print(f"      {Colors.DIM}Est. time: ~{format_time(md['timeout']/3)}{Colors.RESET}")
        step_num += 1
    else:
        print(f"\n  {Colors.DIM}(Skipped - use --multi to enable){Colors.RESET}")
    print()
    
    # Phase 4: Comparison & Dashboard
    print_subheader("PHASE 4: Comparison & Dashboard Generation")
    print(f"\n{Colors.DIM}  Consolidate results and generate interactive visualizations.{Colors.RESET}\n")
    
    for key in ['master_comparison', 'dashboard']:
        phase = EVALUATION_PHASES[key]
        plan['phase4'].append({
            'key': key,
            'step': step_num,
            **phase
        })
        print(f"  {Colors.BRIGHT_WHITE}[{step_num}]{Colors.RESET} {Colors.BOLD}{phase['name']}{Colors.RESET}")
        print(f"      {Colors.DIM}Est. time: ~{format_time(phase['timeout']/3)}{Colors.RESET}")
        step_num += 1
        print()
    
    # Summary
    total_steps = sum(len(v) for v in plan.values())
    total_time = sum(
        item.get('timeout', 300) / 3 
        for phase_items in plan.values() 
        for item in phase_items
    )
    
    print(f"  {Colors.BOLD}{'─' * 60}{Colors.RESET}")
    print(f"  {Colors.BRIGHT_WHITE}📊 Total Steps: {total_steps}{Colors.RESET}")
    print(f"  {Colors.BRIGHT_WHITE}⏱  Estimated Time: ~{format_time(total_time)}{Colors.RESET}")
    print()
    
    return plan


def display_algorithm_details(algo_key: str):
    """Display detailed information about an algorithm"""
    if algo_key in ALGORITHMS:
        algo = ALGORITHMS[algo_key]
    elif algo_key in EVALUATION_PHASES:
        algo = EVALUATION_PHASES[algo_key]
    else:
        return
    
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_CYAN}{'─' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}  {algo['name']}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'─' * 70}{Colors.RESET}")
    print(f"{Colors.DIM}{algo['description']}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'─' * 70}{Colors.RESET}")


# =============================================================================
# INTERACTIVE MENU
# =============================================================================

def interactive_menu(plan: Dict[str, List[Dict]], has_pytorch: bool) -> Dict[str, List[Dict]]:
    """Display interactive menu for user to select/skip steps."""
    print_header("🎮 EXECUTION OPTIONS")
    
    print(f"""
  {Colors.BRIGHT_WHITE}Please select how you want to proceed:{Colors.RESET}

  {Colors.BRIGHT_GREEN}[1]{Colors.RESET} ▶  Run all steps automatically {Colors.DIM}(recommended){Colors.RESET}
  {Colors.BRIGHT_YELLOW}[2]{Colors.RESET} ⚙  Select which steps to run {Colors.DIM}(supports ranges: 1-4, 7, 9-12){Colors.RESET}
  {Colors.BRIGHT_CYAN}[3]{Colors.RESET} 📚  View detailed algorithm descriptions
  {Colors.BRIGHT_RED}[4]{Colors.RESET} ✗  Cancel and exit

""")
    
    while True:
        try:
            choice = input(f"{Colors.BRIGHT_WHITE}  Enter choice [1-4]: {Colors.RESET}").strip()
            
            if choice == '1':
                print_success("Running all steps automatically")
                return plan
                
            elif choice == '2':
                return select_individual_steps(plan)
                
            elif choice == '3':
                display_all_descriptions()
                return interactive_menu(plan, has_pytorch)
                
            elif choice == '4':
                print_warning("Execution cancelled by user")
                sys.exit(0)
                
            else:
                print_error("Invalid choice. Please enter 1, 2, 3, or 4.")
                
        except (EOFError, KeyboardInterrupt):
            print()
            print_warning("Execution cancelled by user")
            sys.exit(0)


def select_individual_steps(plan: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """Allow user to select individual steps to run with flexible input"""
    
    print_subheader("Select Steps to Execute")
    
    # Collect all steps
    all_steps = []
    for phase_key in ['phase1', 'phase2', 'phase3', 'phase4']:
        for item in plan[phase_key]:
            all_steps.append((phase_key, item))
    
    total_steps = len(all_steps)
    
    # Display steps with grouping
    print(f"\n{Colors.BOLD}  Available Steps:{Colors.RESET}\n")
    
    current_phase = None
    for phase_key, item in all_steps:
        if phase_key != current_phase:
            current_phase = phase_key
            phase_names = {
                'phase1': 'Individual Algorithms',
                'phase2': 'Hyperparameter Tuning',
                'phase3': 'Multi-Dataset Evaluation',
                'phase4': 'Comparison & Dashboard'
            }
            print(f"  {Colors.DIM}── {phase_names[phase_key]} ──{Colors.RESET}")
        print(f"  {Colors.BRIGHT_WHITE}[{item['step']:>2}]{Colors.RESET} {item['name']}")
    
    # Show input options
    print(f"""
  {Colors.BOLD}Selection Options:{Colors.RESET}
  {Colors.DIM}────────────────────────────────────────────────────────────{Colors.RESET}
  {Colors.CYAN}all{Colors.RESET}       → Run all steps
  {Colors.CYAN}1{Colors.RESET}         → Run only step 1
  {Colors.CYAN}1,3,5{Colors.RESET}     → Run steps 1, 3, and 5
  {Colors.CYAN}1-4{Colors.RESET}       → Run steps 1 through 4
  {Colors.CYAN}1-4,7,9-12{Colors.RESET} → Run steps 1-4, 7, and 9-12
  {Colors.CYAN}11-12{Colors.RESET}     → Run only comparison & dashboard
  {Colors.DIM}────────────────────────────────────────────────────────────{Colors.RESET}
""")
    
    try:
        selection = input(f"{Colors.BRIGHT_WHITE}  Enter steps to run [all]: {Colors.RESET}").strip().lower()
        
        if selection == '' or selection == 'all':
            print_success("Running all steps")
            return plan
        
        # Parse selection
        selected_steps = set()
        
        try:
            parts = selection.replace(' ', '').split(',')
            for part in parts:
                if '-' in part:
                    # Range: e.g., "1-4"
                    start, end = part.split('-')
                    start, end = int(start), int(end)
                    if start > end:
                        start, end = end, start
                    selected_steps.update(range(start, end + 1))
                else:
                    # Single number
                    selected_steps.add(int(part))
            
            # Validate
            invalid_steps = selected_steps - set(range(1, total_steps + 1))
            if invalid_steps:
                print_warning(f"Invalid step numbers ignored: {sorted(invalid_steps)}")
                selected_steps -= invalid_steps
            
            if not selected_steps:
                print_error("No valid steps selected. Running all steps.")
                return plan
            
            # Filter plan to only include selected steps
            for phase_key in plan:
                plan[phase_key] = [
                    item for item in plan[phase_key] 
                    if item['step'] in selected_steps
                ]
            
            # Show what will run
            selected_list = sorted(selected_steps)
            if len(selected_list) <= 5:
                steps_str = ', '.join(map(str, selected_list))
            else:
                steps_str = f"{selected_list[0]}-{selected_list[-1]} ({len(selected_list)} steps)"
            
            print_success(f"Running steps: {steps_str}")
            
            # Show summary of selected steps
            print(f"\n  {Colors.BOLD}Selected:{Colors.RESET}")
            for phase_key in ['phase1', 'phase2', 'phase3', 'phase4']:
                for item in plan[phase_key]:
                    print(f"    {Colors.BRIGHT_GREEN}✓{Colors.RESET} [{item['step']}] {item['name']}")
            print()
            
        except ValueError as e:
            print_error(f"Invalid input format. Running all steps.")
            # Restore original plan
            return display_execution_plan.__wrapped__(args, has_pytorch) if hasattr(display_execution_plan, '__wrapped__') else plan
            
    except (EOFError, KeyboardInterrupt):
        print()
        print_success("Running all steps")
    
    return plan


def display_all_descriptions():
    """Display detailed descriptions of all algorithms"""
    print_header("📚 ALGORITHM DESCRIPTIONS")
    
    for algo_key, algo in ALGORITHMS.items():
        display_algorithm_details(algo_key)
        print()
    
    for phase_key, phase in EVALUATION_PHASES.items():
        display_algorithm_details(phase_key)
        print()
    
    input(f"\n{Colors.DIM}  Press Enter to continue...{Colors.RESET}")


# =============================================================================
# EXECUTION ENGINE
# =============================================================================

def execute_plan(plan: Dict[str, List[Dict]], args) -> List[Dict]:
    """Execute the plan and return results"""
    
    results = []
    total_steps = sum(len(v) for v in plan.values())
    current_step = 0
    
    print_header("🚀 EXECUTING EVALUATION PIPELINE")
    print(f"\n{Colors.DIM}  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}\n")
    
    phase_names = {
        'phase1': 'Individual Algorithm Analysis',
        'phase2': 'Hyperparameter Tuning', 
        'phase3': 'Multi-Dataset Evaluation',
        'phase4': 'Comparison & Dashboard'
    }
    
    for phase_key in ['phase1', 'phase2', 'phase3', 'phase4']:
        phase_items = plan[phase_key]
        
        if not phase_items:
            continue
            
        print_subheader(f"PHASE: {phase_names[phase_key]}")
        
        for item in phase_items:
            current_step += 1
            
            print(f"\n{Colors.BOLD}{Colors.BRIGHT_WHITE}  [{current_step}/{total_steps}] {item['name']}{Colors.RESET}")
            print(f"  {Colors.DIM}{'─' * 60}{Colors.RESET}")
            
            # Build extra args
            extra_args = []
            if item['key'] == 'multi_dataset' and args.real:
                extra_args.append('--real')
            
            # Run the script
            print(f"  {Colors.BRIGHT_YELLOW}⏳ Running...{Colors.RESET}\n")
            
            success, elapsed = run_script(
                item['script'],
                item['name'],
                extra_args=extra_args if extra_args else None,
                timeout=item.get('timeout', 600)
            )
            
            # Record result
            result = {
                'step': current_step,
                'name': item['name'],
                'key': item['key'],
                'success': success,
                'elapsed': elapsed
            }
            results.append(result)
            
            # Print result
            if success:
                print(f"\n  {Colors.BRIGHT_GREEN}✓ Completed in {format_time(elapsed)}{Colors.RESET}")
            else:
                print(f"\n  {Colors.BRIGHT_RED}✗ Failed after {format_time(elapsed)}{Colors.RESET}")
    
    return results


def display_results_summary(results: List[Dict], total_time: float):
    """Display final results summary"""
    
    print_header("📊 EXECUTION SUMMARY")
    
    success_count = sum(1 for r in results if r['success'])
    fail_count = len(results) - success_count
    
    # Results table
    print(f"\n  {Colors.BOLD}{'Step':<6} {'Name':<40} {'Status':<12} {'Time':<10}{Colors.RESET}")
    print(f"  {'─' * 70}")
    
    for r in results:
        status_color = Colors.BRIGHT_GREEN if r['success'] else Colors.BRIGHT_RED
        status_text = "✓ Pass" if r['success'] else "✗ Fail"
        print(f"  {r['step']:<6} {r['name']:<40} {status_color}{status_text:<12}{Colors.RESET} {format_time(r['elapsed']):<10}")
    
    print(f"  {'─' * 70}")
    
    # Summary stats
    print(f"\n  {Colors.BOLD}Results:{Colors.RESET}")
    print(f"    {Colors.BRIGHT_GREEN}✓ Passed: {success_count}{Colors.RESET}")
    if fail_count > 0:
        print(f"    {Colors.BRIGHT_RED}✗ Failed: {fail_count}{Colors.RESET}")
    print(f"    {Colors.BRIGHT_CYAN}⏱  Total Time: {format_time(total_time)}{Colors.RESET}")
    
    # Output locations
    print(f"\n  {Colors.BOLD}Output Locations:{Colors.RESET}")
    print(f"    {Colors.CYAN}📊{Colors.RESET} ../visualizations/dashboard.html  {Colors.DIM}(Main dashboard){Colors.RESET}")
    print(f"    {Colors.CYAN}📈{Colors.RESET} ../visualizations/*.html          {Colors.DIM}(Algorithm visualizations){Colors.RESET}")
    print(f"    {Colors.CYAN}📋{Colors.RESET} ../reports/                       {Colors.DIM}(Statistical reports){Colors.RESET}")
    
    # Final message
    if fail_count == 0:
        print(f"\n{Colors.BOLD}{Colors.BRIGHT_GREEN}  🎉 All steps completed successfully!{Colors.RESET}")
    else:
        print(f"\n{Colors.BOLD}{Colors.BRIGHT_YELLOW}  ⚠ Some steps failed. Check logs above for details.{Colors.RESET}")
    
    print()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Recommendation System - Comprehensive Evaluation Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
{Colors.BOLD}Examples:{Colors.RESET}
  python run_all.py                     # Interactive standard run
  python run_all.py --all               # Everything (all algos, datasets, tuning)
  python run_all.py --multi --neural    # Multi-dataset with Neural CF
  python run_all.py --quick --yes       # Quick run, skip confirmation
        """
    )
    
    parser.add_argument('--quick', action='store_true', 
                       help='Quick test (CF + Popularity only)')
    parser.add_argument('--neural', action='store_true', 
                       help='Include Neural CF & Hybrid (requires PyTorch)')
    parser.add_argument('--multi', action='store_true', 
                       help='Run multi-dataset evaluation')
    parser.add_argument('--multi-only', action='store_true', dest='multi_only',
                       help='Run ONLY multi-dataset (skip individual algorithms)')
    parser.add_argument('--real', action='store_true', 
                       help='Use real MovieLens-100K dataset')
    parser.add_argument('--full', action='store_true', 
                       help='Include hyperparameter tuning')
    parser.add_argument('--all', action='store_true', 
                       help='🚀 Run EVERYTHING (all algos, datasets, real data, tuning)')
    parser.add_argument('--yes', '-y', action='store_true',
                       help='Skip confirmation prompts, run automatically')
    parser.add_argument('--no-color', action='store_true',
                       help='Disable colored output')
    
    args = parser.parse_args()
    
    # Handle --no-color
    if args.no_color:
        Colors.disable()
    
    # Handle --all flag
    if args.all:
        args.full = True
        args.neural = True
        args.multi = True
        args.real = True
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    if os.path.exists('scripts'):
        os.chdir('scripts')
    
    # Welcome banner
    print_header("🎯 RECOMMENDATION SYSTEM EVALUATION FRAMEWORK")
    
    print(f"""
{Colors.DIM}  A comprehensive evaluation pipeline for recommendation algorithms.
  This framework trains, evaluates, and compares 10 different algorithms
  across multiple datasets with interactive visualizations.{Colors.RESET}

{Colors.BOLD}  Version:{Colors.RESET} 2.0
{Colors.BOLD}  Date:{Colors.RESET}    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")
    
    # Check PyTorch
    has_pytorch = check_pytorch()
    
    if args.neural:
        if has_pytorch:
            print_success("PyTorch detected - Neural networks enabled")
        else:
            print_warning("PyTorch not installed - Neural networks will be skipped")
            print(f"  {Colors.DIM}Install with: pip install torch{Colors.RESET}")
    
    # Display execution plan
    plan = display_execution_plan(args, has_pytorch)
    
    # Interactive menu (unless --yes flag)
    if not args.yes:
        plan = interactive_menu(plan, has_pytorch)
    else:
        print_success("Auto-run enabled (--yes flag)")
    
    # Confirm execution
    total_steps = sum(len(v) for v in plan.values())
    
    if total_steps == 0:
        print_warning("No steps selected. Exiting.")
        return
    
    if not args.yes:
        print()
        if not get_user_confirmation(f"  Ready to execute {total_steps} steps?", default=True):
            print_warning("Execution cancelled by user")
            return
    
    # Execute
    total_start = time.time()
    results = execute_plan(plan, args)
    total_time = time.time() - total_start
    
    # Display summary
    display_results_summary(results, total_time)


if __name__ == "__main__":
    main()
