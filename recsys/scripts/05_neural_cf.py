"""
05_neural_cf.py
================
Neural Collaborative Filtering (NCF / NeuMF)

Implements:
- Generalized Matrix Factorization (GMF)
- Multi-Layer Perceptron (MLP)
- Neural Matrix Factorization (NeuMF) - combines GMF + MLP

Features:
- Interactive visualization of embeddings
- Comprehensive statistical reports
- Training curves and convergence analysis
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from pyvis.network import Network
from sklearn.manifold import TSNE
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Create output directories
os.makedirs("../outputs/neural_cf", exist_ok=True)
os.makedirs("../visualizations", exist_ok=True)
os.makedirs("../reports/neural_cf", exist_ok=True)

# Set device with detailed logging
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log_device_info():
    """Log GPU/CPU information"""
    print("=" * 60)
    print("🖥️  DEVICE CONFIGURATION")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"   ✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")
        print(f"   📊 CUDA Version: {torch.version.cuda}")
        print(f"   💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"   🚀 Using: GPU (CUDA)")
    else:
        print("   ⚠️  No GPU detected")
        print(f"   🖥️  Using: CPU")
        print(f"   💡 Tip: Install CUDA-enabled PyTorch for faster training")
    print(f"   📦 PyTorch Version: {torch.__version__}")
    print("=" * 60)


class RatingsDataset(Dataset):
    """PyTorch Dataset for ratings data"""
    
    def __init__(self, user_ids, item_ids, ratings):
        self.user_ids = torch.LongTensor(user_ids)
        self.item_ids = torch.LongTensor(item_ids)
        self.ratings = torch.FloatTensor(ratings)
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]


class GMF(nn.Module):
    """Generalized Matrix Factorization"""
    
    def __init__(self, n_users, n_items, embedding_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.output = nn.Linear(embedding_dim, 1)
        
        # Initialize weights
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
    
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Element-wise product
        interaction = user_emb * item_emb
        
        output = self.output(interaction)
        return output.squeeze()


class MLP(nn.Module):
    """Multi-Layer Perceptron for CF"""
    
    def __init__(self, n_users, n_items, embedding_dim, hidden_layers=[64, 32, 16]):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # MLP layers
        layers = []
        input_dim = embedding_dim * 2
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
        # Initialize
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
    
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Concatenate embeddings
        concat = torch.cat([user_emb, item_emb], dim=-1)
        
        output = self.mlp(concat)
        return output.squeeze()


class NeuMF(nn.Module):
    """Neural Matrix Factorization (GMF + MLP)"""
    
    def __init__(self, n_users, n_items, gmf_dim=32, mlp_dim=32, hidden_layers=[64, 32, 16]):
        super().__init__()
        
        # GMF embeddings
        self.gmf_user_embedding = nn.Embedding(n_users, gmf_dim)
        self.gmf_item_embedding = nn.Embedding(n_items, gmf_dim)
        
        # MLP embeddings
        self.mlp_user_embedding = nn.Embedding(n_users, mlp_dim)
        self.mlp_item_embedding = nn.Embedding(n_items, mlp_dim)
        
        # MLP layers
        layers = []
        input_dim = mlp_dim * 2
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Final prediction layer
        self.output = nn.Linear(gmf_dim + hidden_layers[-1], 1)
        
        # Initialize
        for embedding in [self.gmf_user_embedding, self.gmf_item_embedding,
                         self.mlp_user_embedding, self.mlp_item_embedding]:
            nn.init.normal_(embedding.weight, std=0.01)
    
    def forward(self, user_ids, item_ids):
        # GMF path
        gmf_user = self.gmf_user_embedding(user_ids)
        gmf_item = self.gmf_item_embedding(item_ids)
        gmf_output = gmf_user * gmf_item
        
        # MLP path
        mlp_user = self.mlp_user_embedding(user_ids)
        mlp_item = self.mlp_item_embedding(item_ids)
        mlp_concat = torch.cat([mlp_user, mlp_item], dim=-1)
        mlp_output = self.mlp(mlp_concat)
        
        # Combine
        concat = torch.cat([gmf_output, mlp_output], dim=-1)
        output = self.output(concat)
        
        return output.squeeze()


class NeuralCFRecommender:
    """
    Neural Collaborative Filtering Recommender
    
    Supports GMF, MLP, and NeuMF architectures
    """
    
    def __init__(
        self,
        model_type: str = 'neumf',  # 'gmf', 'mlp', 'neumf'
        embedding_dim: int = 32,
        hidden_layers: List[int] = [64, 32, 16],
        learning_rate: float = 0.001,
        batch_size: int = 256,
        n_epochs: int = 20,
        weight_decay: float = 1e-5,
        random_state: int = 42
    ):
        self.model_type = model_type
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.weight_decay = weight_decay
        self.random_state = random_state
        
        # Model
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        
        # Mappings
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        
        # Data
        self.global_mean = 3.0
        self.user_ratings = defaultdict(dict)
        
        # Training history
        self.training_history = []
        
        # Statistics
        self.stats = {
            'n_users': 0,
            'n_items': 0,
            'n_ratings': 0
        }
        
        torch.manual_seed(random_state)
        np.random.seed(random_state)
    
    def fit(self, ratings_df: pd.DataFrame) -> 'NeuralCFRecommender':
        """Train the neural CF model"""
        print(f"\n🔧 Training Neural CF ({self.model_type.upper()})...")
        print(f"   embedding_dim: {self.embedding_dim}")
        print(f"   hidden_layers: {self.hidden_layers}")
        print(f"   learning_rate: {self.learning_rate}")
        print(f"   batch_size: {self.batch_size}")
        print(f"   n_epochs: {self.n_epochs}")
        
        # Create mappings
        unique_users = ratings_df['user_id'].unique()
        unique_items = ratings_df['item_id'].unique()
        
        self.user_mapping = {u: i for i, u in enumerate(unique_users)}
        self.item_mapping = {it: i for i, it in enumerate(unique_items)}
        self.reverse_user_mapping = {i: u for u, i in self.user_mapping.items()}
        self.reverse_item_mapping = {i: it for it, i in self.item_mapping.items()}
        
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        self.stats['n_users'] = n_users
        self.stats['n_items'] = n_items
        self.stats['n_ratings'] = len(ratings_df)
        
        self.global_mean = ratings_df['rating'].mean()
        
        # Store user ratings
        for _, row in ratings_df.iterrows():
            self.user_ratings[row['user_id']][row['item_id']] = row['rating']
        
        # Create model
        if self.model_type == 'gmf':
            self.model = GMF(n_users, n_items, self.embedding_dim)
        elif self.model_type == 'mlp':
            self.model = MLP(n_users, n_items, self.embedding_dim, self.hidden_layers)
        else:  # neumf
            self.model = NeuMF(n_users, n_items, self.embedding_dim, self.embedding_dim, self.hidden_layers)
        
        self.model.to(device)
        print(f"   🚀 Model loaded on: {device}")
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Prepare data
        user_indices = ratings_df['user_id'].map(self.user_mapping).values
        item_indices = ratings_df['item_id'].map(self.item_mapping).values
        ratings = ratings_df['rating'].values
        
        dataset = RatingsDataset(user_indices, item_indices, ratings)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        print("   Training...")
        for epoch in range(self.n_epochs):
            self.model.train()
            total_loss = 0
            n_batches = 0
            
            for user_batch, item_batch, rating_batch in dataloader:
                user_batch = user_batch.to(device)
                item_batch = item_batch.to(device)
                rating_batch = rating_batch.to(device)
                
                self.optimizer.zero_grad()
                
                predictions = self.model(user_batch, item_batch)
                loss = self.criterion(predictions, rating_batch)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            rmse = np.sqrt(avg_loss)
            
            self.training_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'rmse': rmse
            })
            
            if (epoch + 1) % 5 == 0:
                print(f"      Epoch {epoch + 1}/{self.n_epochs}: Loss = {avg_loss:.4f}, RMSE = {rmse:.4f}")
        
        print(f"   ✅ Model trained: {n_users} users, {n_items} items")
        
        return self
    
    def predict(self, user_id: Any, item_id: Any) -> float:
        """Predict rating for user-item pair"""
        if user_id not in self.user_mapping or item_id not in self.item_mapping:
            return self.global_mean
        
        self.model.eval()
        
        user_idx = torch.LongTensor([self.user_mapping[user_id]]).to(device)
        item_idx = torch.LongTensor([self.item_mapping[item_id]]).to(device)
        
        with torch.no_grad():
            prediction = self.model(user_idx, item_idx).item()
        
        return np.clip(prediction, 1, 5)
    
    def recommend(
        self,
        user_id: Any,
        n_recommendations: int = 10,
        exclude_rated: bool = True
    ) -> List[Tuple[Any, float]]:
        """Generate top-N recommendations"""
        if user_id not in self.user_mapping:
            return self._get_popular_items(n_recommendations)
        
        self.model.eval()
        
        user_idx = self.user_mapping[user_id]
        rated_items = set(self.user_ratings[user_id].keys()) if exclude_rated else set()
        
        # Predict for all items
        all_items = []
        batch_size = 1000
        
        item_indices = [i for i in range(self.stats['n_items'])
                       if self.reverse_item_mapping[i] not in rated_items]
        
        for i in range(0, len(item_indices), batch_size):
            batch_items = item_indices[i:i + batch_size]
            
            user_tensor = torch.LongTensor([user_idx] * len(batch_items)).to(device)
            item_tensor = torch.LongTensor(batch_items).to(device)
            
            with torch.no_grad():
                predictions = self.model(user_tensor, item_tensor).cpu().numpy()
            
            for item_idx, pred in zip(batch_items, predictions):
                item_id = self.reverse_item_mapping[item_idx]
                all_items.append((item_id, np.clip(pred, 1, 5)))
        
        # Sort and return top N
        all_items.sort(key=lambda x: x[1], reverse=True)
        return all_items[:n_recommendations]
    
    def _get_popular_items(self, n: int) -> List[Tuple[Any, float]]:
        """Return popular items for cold start"""
        item_counts = defaultdict(int)
        item_ratings = defaultdict(list)
        
        for uid, ratings in self.user_ratings.items():
            for iid, r in ratings.items():
                item_counts[iid] += 1
                item_ratings[iid].append(r)
        
        popular = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:n]
        return [(iid, np.mean(item_ratings[iid])) for iid, _ in popular]
    
    def get_user_embedding(self, user_id: Any) -> np.ndarray:
        """Get user embedding vector"""
        if user_id not in self.user_mapping:
            return None
        
        user_idx = self.user_mapping[user_id]
        
        if self.model_type == 'neumf':
            gmf_emb = self.model.gmf_user_embedding.weight[user_idx].detach().cpu().numpy()
            mlp_emb = self.model.mlp_user_embedding.weight[user_idx].detach().cpu().numpy()
            return np.concatenate([gmf_emb, mlp_emb])
        else:
            return self.model.user_embedding.weight[user_idx].detach().cpu().numpy()
    
    def get_item_embedding(self, item_id: Any) -> np.ndarray:
        """Get item embedding vector"""
        if item_id not in self.item_mapping:
            return None
        
        item_idx = self.item_mapping[item_id]
        
        if self.model_type == 'neumf':
            gmf_emb = self.model.gmf_item_embedding.weight[item_idx].detach().cpu().numpy()
            mlp_emb = self.model.mlp_item_embedding.weight[item_idx].detach().cpu().numpy()
            return np.concatenate([gmf_emb, mlp_emb])
        else:
            return self.model.item_embedding.weight[item_idx].detach().cpu().numpy()
    
    def evaluate(
        self,
        test_df: pd.DataFrame,
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """Evaluate model on test set"""
        print("\n📊 Evaluating model...")
        
        self.model.eval()
        
        # Rating prediction
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
        
        # Coverage
        all_recommended = set()
        sample_users = list(self.user_mapping.keys())[:100]
        for user_id in sample_users:
            recs = self.recommend(user_id, n_recommendations=10)
            all_recommended.update([item for item, _ in recs])
        
        metrics['Coverage'] = len(all_recommended) / self.stats['n_items']
        
        return metrics


def create_interactive_visualization(
    ncf_model: NeuralCFRecommender,
    sample_users: int = 40,
    sample_items: int = 60,
    output_path: str = "../visualizations/neural_cf.html",
    metrics: Dict = None
):
    """Create interactive visualization of neural embeddings with enhanced template"""
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
    
    # Sample users and items
    user_ids = list(ncf_model.user_mapping.keys())[:sample_users]
    item_ids = list(ncf_model.item_mapping.keys())[:sample_items]
    
    # Get embeddings
    user_embeddings = np.array([ncf_model.get_user_embedding(u) for u in user_ids])
    item_embeddings = np.array([ncf_model.get_item_embedding(i) for i in item_ids])
    
    # t-SNE projection
    all_embeddings = np.vstack([user_embeddings, item_embeddings])
    
    perplexity = min(30, len(all_embeddings) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    user_positions = embeddings_2d[:len(user_ids)] * 400
    item_positions = embeddings_2d[len(user_ids):] * 400
    
    # Add user nodes with enhanced tooltips
    for i, user_id in enumerate(user_ids):
        x, y = user_positions[i]
        n_ratings = len(ncf_model.user_ratings[user_id])
        embedding_norm = np.linalg.norm(user_embeddings[i])
        
        net.add_node(
            f"U_{user_id}",
            label=f"U{user_id}",
            color="#4285f4",
            size=22 + min(n_ratings * 0.5, 15),
            x=float(x),
            y=float(y),
            shape="dot",
            title=f"<div style='padding:8px;'><strong>👤 User {user_id}</strong><br><br>"
                  f"<b>Ratings:</b> {n_ratings}<br>"
                  f"<b>Embedding Norm:</b> {embedding_norm:.3f}<br>"
                  f"<b>Position:</b> t-SNE projected</div>",
            group="user"
        )
    
    # Add item nodes with enhanced tooltips
    for i, item_id in enumerate(item_ids):
        x, y = item_positions[i]
        embedding_norm = np.linalg.norm(item_embeddings[i])
        
        net.add_node(
            f"I_{item_id}",
            label=f"I{item_id}",
            color="#34a853",
            size=20,
            x=float(x),
            y=float(y),
            shape="dot",
            title=f"<div style='padding:8px;'><strong>📦 Item {item_id}</strong><br><br>"
                  f"<b>Embedding Norm:</b> {embedding_norm:.3f}<br>"
                  f"<b>Position:</b> t-SNE projected</div>",
            group="item"
        )
    
    # Collect recommendations for tables
    all_recommendations = []
    edge_count = 0
    
    # Add recommendation edges with descriptive labels
    for user_id in user_ids[:15]:
        recs = ncf_model.recommend(user_id, n_recommendations=3)
        for item_id, score in recs:
            reason = f"Neural network prediction (score={score:.2f})"
            all_recommendations.append((item_id, score, reason))
            
            if f"I_{item_id}" in [n['id'] for n in net.nodes]:
                label_text = f"pred {score:.1f}"
                net.add_edge(
                    f"U_{user_id}",
                    f"I_{item_id}",
                    color="#ea4335",
                    width=2 + score * 0.3,
                    label=label_text,
                    title=f"<div style='padding:8px;'><b>🎯 Recommendation</b><br><br>"
                          f"<b>User:</b> {user_id}<br>"
                          f"<b>Item:</b> {item_id}<br>"
                          f"<b>Predicted:</b> {score:.2f}/5<br>"
                          f"<b>Model:</b> {ncf_model.model_type.upper()}</div>",
                    dashes=True,
                    font={"size": 9, "color": "#ea4335", "background": "rgba(255,255,255,0.8)"}
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
        "Items": len(item_ids),
        "Edges": edge_count,
        "Embedding Dim": ncf_model.embedding_dim,
        "Model": ncf_model.model_type.upper()
    }
    
    # Node and edge types for legend
    node_types = [
        {"color": "#4285f4", "label": "Users", "description": "User embeddings (t-SNE projected)"},
        {"color": "#34a853", "label": "Items", "description": "Item embeddings (t-SNE projected)"}
    ]
    
    edge_types = [
        {"color": "#ea4335", "label": "Prediction (pred X.X)", "style": "dashed", "description": "Neural network predicted preference"}
    ]
    
    # Generate enhanced or basic visualization
    if use_enhanced:
        generate_enhanced_visualization(
            net=net,
            algorithm_key="neural_cf",
            output_path=output_path,
            metrics=metrics,
            graph_stats=graph_stats,
            node_types=node_types,
            edge_types=edge_types,
            recommendations_data=recommendations_data
        )
    else:
        # Fallback: basic visualization with simple legend
        legend_html = """
        <div style="position: absolute; top: 10px; right: 10px; background: white; 
                    padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    font-family: Arial, sans-serif; font-size: 12px;">
            <h3 style="margin: 0 0 10px 0; font-size: 14px;">Neural CF Embeddings</h3>
            <p style="font-size: 11px; color: #666;">t-SNE projection</p>
            <div style="margin: 5px 0;">
                <span style="display: inline-block; width: 15px; height: 15px; 
                            background: #4285f4; border-radius: 50%; margin-right: 8px;"></span>
                Users
            </div>
            <div style="margin: 5px 0;">
                <span style="display: inline-block; width: 15px; height: 15px; 
                            background: #34a853; border-radius: 50%; margin-right: 8px;"></span>
                Items
            </div>
            <div style="margin: 5px 0;">
                <span style="display: inline-block; width: 30px; height: 2px; 
                            background: #ea4335; margin-right: 8px;"></span>
                Recommendations
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
    ncf_model: NeuralCFRecommender,
    metrics: Dict[str, float],
    output_dir: str = "../reports/neural_cf"
):
    """Generate comprehensive statistical report"""
    print("\n📈 Generating statistical report...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Neural CF Analysis ({ncf_model.model_type.upper()})', fontsize=14)
    
    # 1. Training curve
    ax1 = axes[0, 0]
    epochs = [h['epoch'] for h in ncf_model.training_history]
    losses = [h['loss'] for h in ncf_model.training_history]
    ax1.plot(epochs, losses, 'b-o', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training Loss Curve')
    ax1.grid(True, alpha=0.3)
    
    # 2. RMSE over epochs
    ax2 = axes[0, 1]
    rmses = [h['rmse'] for h in ncf_model.training_history]
    ax2.plot(epochs, rmses, 'r-s', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('RMSE')
    ax2.set_title('Training RMSE Curve')
    ax2.grid(True, alpha=0.3)
    
    # 3. Embedding norm distribution
    ax3 = axes[0, 2]
    user_norms = []
    for u in list(ncf_model.user_mapping.keys())[:100]:
        emb = ncf_model.get_user_embedding(u)
        if emb is not None:
            user_norms.append(np.linalg.norm(emb))
    
    item_norms = []
    for i in list(ncf_model.item_mapping.keys())[:100]:
        emb = ncf_model.get_item_embedding(i)
        if emb is not None:
            item_norms.append(np.linalg.norm(emb))
    
    ax3.hist(user_norms, bins=20, alpha=0.6, label='Users', color='blue')
    ax3.hist(item_norms, bins=20, alpha=0.6, label='Items', color='green')
    ax3.set_xlabel('Embedding Norm')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Embedding Norm Distribution')
    ax3.legend()
    
    # 4. Prediction distribution
    ax4 = axes[1, 0]
    sample_predictions = []
    sample_users = list(ncf_model.user_mapping.keys())[:50]
    sample_items = list(ncf_model.item_mapping.keys())[:50]
    for u in sample_users:
        for i in sample_items[:10]:
            sample_predictions.append(ncf_model.predict(u, i))
    ax4.hist(sample_predictions, bins=30, color='coral', edgecolor='white', alpha=0.7)
    ax4.set_xlabel('Predicted Rating')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Prediction Distribution (sample)')
    
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
    
    # 6. Model architecture info
    ax6 = axes[1, 2]
    ax6.axis('off')
    model_info = f"""
    Model Type: {ncf_model.model_type.upper()}
    
    Architecture:
    - Embedding Dim: {ncf_model.embedding_dim}
    - Hidden Layers: {ncf_model.hidden_layers}
    
    Training:
    - Epochs: {ncf_model.n_epochs}
    - Batch Size: {ncf_model.batch_size}
    - Learning Rate: {ncf_model.learning_rate}
    
    Dataset:
    - Users: {ncf_model.stats['n_users']:,}
    - Items: {ncf_model.stats['n_items']:,}
    - Ratings: {ncf_model.stats['n_ratings']:,}
    """
    ax6.text(0.1, 0.9, model_info, transform=ax6.transAxes, fontsize=20,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax6.set_title('Model Configuration')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/analysis_figures.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Text report
    report = f"""
================================================================================
NEURAL COLLABORATIVE FILTERING - STATISTICAL REPORT
================================================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MODEL CONFIGURATION
-------------------
Model Type: {ncf_model.model_type.upper()}
Embedding Dimension: {ncf_model.embedding_dim}
Hidden Layers: {ncf_model.hidden_layers}
Learning Rate: {ncf_model.learning_rate}
Batch Size: {ncf_model.batch_size}
Epochs: {ncf_model.n_epochs}
Weight Decay: {ncf_model.weight_decay}

DATASET STATISTICS
------------------
Number of Users: {ncf_model.stats['n_users']:,}
Number of Items: {ncf_model.stats['n_items']:,}
Number of Ratings: {ncf_model.stats['n_ratings']:,}

TRAINING RESULTS
----------------
Final Loss: {ncf_model.training_history[-1]['loss']:.4f}
Final Training RMSE: {ncf_model.training_history[-1]['rmse']:.4f}

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
    print("NEURAL COLLABORATIVE FILTERING")
    print("=" * 70)
    
    # Log device information (GPU/CPU)
    log_device_info()
    
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
    
    # Test different model types
    model_types = ['gmf', 'mlp', 'neumf']
    
    for model_type in model_types:
        print("\n" + "=" * 70)
        print(f"TRAINING {model_type.upper()} MODEL")
        print("=" * 70)
        
        model = NeuralCFRecommender(
            model_type=model_type,
            embedding_dim=32,
            hidden_layers=[64, 32, 16],
            learning_rate=0.001,
            batch_size=256,
            n_epochs=15
        )
        model.fit(train_df)
        
        metrics = model.evaluate(test_df)
        print(f"\n📊 {model_type.upper()} Metrics:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        create_interactive_visualization(
            model,
            sample_users=35,
            sample_items=50,
            output_path=f"../visualizations/neural_cf_{model_type}.html",
            metrics=metrics
        )
        
        generate_statistical_report(
            model, metrics,
            output_dir=f"../reports/neural_cf/{model_type}"
        )
    
    print("\n" + "=" * 70)
    print("✅ NEURAL CF COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
