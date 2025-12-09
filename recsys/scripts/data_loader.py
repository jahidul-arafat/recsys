"""
data_loader.py
================
Data Loader Utility for Recommendation System Project

This module provides:
1. REAL MovieLens 100K Dataset (automatic download)
2. Synthetic data generators that mimic the structure of:
   - MovieLens 100K Dataset
   - Amazon Product Reviews (Electronics)
   - Book-Crossing Dataset

These datasets allow you to test all algorithms with both real and synthetic data.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import os
from datetime import datetime, timedelta
import random
import zipfile
import urllib.request
import shutil

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# URLs for real datasets
MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


class DatasetGenerator:
    """Generate synthetic recommendation datasets and load real datasets"""
    
    def __init__(self, output_dir: str = "../data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def load_real_movielens_100k(self) -> Dict[str, pd.DataFrame]:
        """
        Download and load the REAL MovieLens 100K dataset
        
        Returns:
            Dict with 'ratings', 'users', 'items' DataFrames
        """
        print(f"🎬 Loading REAL MovieLens 100K dataset...")
        
        ml_dir = os.path.join(self.output_dir, "ml-100k")
        zip_path = os.path.join(self.output_dir, "ml-100k.zip")
        
        # Download if not exists
        if not os.path.exists(ml_dir):
            print("   📥 Downloading MovieLens 100K...")
            try:
                urllib.request.urlretrieve(MOVIELENS_100K_URL, zip_path)
                print("   📦 Extracting...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.output_dir)
                os.remove(zip_path)
                print("   ✅ Download complete!")
            except Exception as e:
                print(f"   ⚠️ Download failed: {e}")
                print("   🔄 Falling back to synthetic data...")
                return self.generate_movielens_style()
        
        # Load ratings (u.data: user_id, item_id, rating, timestamp)
        ratings_path = os.path.join(ml_dir, "u.data")
        ratings_df = pd.read_csv(
            ratings_path, 
            sep='\t', 
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            encoding='latin-1'
        )
        
        # Load user demographics (u.user: user_id, age, gender, occupation, zip)
        users_path = os.path.join(ml_dir, "u.user")
        users_df = pd.read_csv(
            users_path,
            sep='|',
            names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
            encoding='latin-1'
        )
        
        # Load movie metadata (u.item: movie_id, title, release_date, ..., genres)
        items_path = os.path.join(ml_dir, "u.item")
        genre_cols = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 
                     'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                     'Sci-Fi', 'Thriller', 'War', 'Western']
        items_df = pd.read_csv(
            items_path,
            sep='|',
            names=['item_id', 'title', 'release_date', 'video_release', 'imdb_url'] + genre_cols,
            encoding='latin-1'
        )
        
        # Convert genre binary columns to pipe-separated string
        def get_genres(row):
            genres = [g for g in genre_cols if row[g] == 1]
            return '|'.join(genres) if genres else 'unknown'
        
        items_df['genres'] = items_df.apply(get_genres, axis=1)
        items_df = items_df[['item_id', 'title', 'release_date', 'genres']]
        
        n_users = ratings_df['user_id'].nunique()
        n_items = ratings_df['item_id'].nunique()
        n_ratings = len(ratings_df)
        sparsity = 1 - n_ratings / (n_users * n_items)
        
        print(f"   Users: {n_users}, Items: {n_items}, Ratings: {n_ratings}")
        print(f"   📊 Sparsity: {sparsity:.2%}")
        
        return {
            'ratings': ratings_df,
            'users': users_df,
            'items': items_df,
            'dataset_name': 'MovieLens-100K (Real)'
        }
        
    def generate_movielens_style(
        self, 
        n_users: int = 943, 
        n_items: int = 1682,
        n_ratings: int = 100000,
        rating_range: Tuple[int, int] = (1, 5)
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate MovieLens-style dataset
        
        Returns:
            Dict with 'ratings', 'users', 'items' DataFrames
        """
        print(f"🎬 Generating MovieLens-style dataset...")
        print(f"   Users: {n_users}, Items: {n_items}, Ratings: {n_ratings}")
        
        # Generate user IDs and item IDs for ratings
        # Use power-law distribution to simulate real user behavior
        # (some users rate many items, most rate few)
        user_weights = np.random.pareto(1.5, n_users) + 1
        user_weights = user_weights / user_weights.sum()
        
        item_weights = np.random.pareto(1.2, n_items) + 1
        item_weights = item_weights / item_weights.sum()
        
        user_ids = np.random.choice(range(1, n_users + 1), n_ratings, p=user_weights)
        item_ids = np.random.choice(range(1, n_items + 1), n_ratings, p=item_weights)
        
        # Generate ratings with some correlation to item popularity
        base_ratings = np.random.normal(3.5, 1.0, n_ratings)
        item_bias = np.random.normal(0, 0.5, n_items)
        user_bias = np.random.normal(0, 0.3, n_users)
        
        ratings = base_ratings + item_bias[item_ids - 1] + user_bias[user_ids - 1]
        ratings = np.clip(ratings, rating_range[0], rating_range[1])
        ratings = np.round(ratings).astype(int)
        
        # Generate timestamps
        base_date = datetime(1998, 1, 1)
        timestamps = [base_date + timedelta(days=random.randint(0, 730)) for _ in range(n_ratings)]
        timestamps = [int(ts.timestamp()) for ts in timestamps]
        
        ratings_df = pd.DataFrame({
            'user_id': user_ids,
            'item_id': item_ids,
            'rating': ratings,
            'timestamp': timestamps
        }).drop_duplicates(subset=['user_id', 'item_id']).reset_index(drop=True)
        
        # Generate user demographics
        ages = np.random.choice([18, 25, 35, 45, 55, 65], n_users, p=[0.15, 0.25, 0.25, 0.2, 0.1, 0.05])
        genders = np.random.choice(['M', 'F'], n_users, p=[0.6, 0.4])
        occupations = np.random.choice([
            'student', 'engineer', 'writer', 'executive', 'healthcare',
            'educator', 'artist', 'scientist', 'homemaker', 'other'
        ], n_users)
        
        users_df = pd.DataFrame({
            'user_id': range(1, n_users + 1),
            'age': ages,
            'gender': genders,
            'occupation': occupations
        })
        
        # Generate movie metadata
        genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 
                  'Documentary', 'Drama', 'Fantasy', 'Horror', 'Musical',
                  'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        
        movie_titles = [f"Movie_{i}" for i in range(1, n_items + 1)]
        movie_years = np.random.choice(range(1950, 1999), n_items)
        movie_genres = [
            '|'.join(random.sample(genres, random.randint(1, 4))) 
            for _ in range(n_items)
        ]
        
        items_df = pd.DataFrame({
            'item_id': range(1, n_items + 1),
            'title': movie_titles,
            'release_year': movie_years,
            'genres': movie_genres
        })
        
        print(f"   ✅ Generated {len(ratings_df)} unique ratings")
        
        return {
            'ratings': ratings_df,
            'users': users_df,
            'items': items_df,
            'dataset_name': 'MovieLens-Style'
        }
    
    def generate_amazon_style(
        self,
        n_users: int = 5000,
        n_products: int = 2000,
        n_reviews: int = 50000
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate Amazon Product Reviews-style dataset
        
        Returns:
            Dict with 'reviews', 'products' DataFrames
        """
        print(f"📦 Generating Amazon-style dataset...")
        print(f"   Users: {n_users}, Products: {n_products}, Reviews: {n_reviews}")
        
        # Power-law distribution for users and products
        user_weights = np.random.pareto(2.0, n_users) + 1
        user_weights = user_weights / user_weights.sum()
        
        product_weights = np.random.pareto(1.5, n_products) + 1
        product_weights = product_weights / product_weights.sum()
        
        user_ids = np.random.choice([f"U{i:05d}" for i in range(n_users)], n_reviews, p=user_weights)
        product_ids = np.random.choice([f"P{i:05d}" for i in range(n_products)], n_reviews, p=product_weights)
        
        # Generate ratings (5-star scale)
        # Amazon reviews tend to be bimodal (mostly 5s and 1s)
        rating_probs = [0.15, 0.05, 0.1, 0.2, 0.5]  # 1-5 stars
        ratings = np.random.choice([1, 2, 3, 4, 5], n_reviews, p=rating_probs)
        
        # Generate review text snippets (simplified)
        positive_words = ['great', 'excellent', 'love', 'perfect', 'amazing', 'best', 'fantastic']
        negative_words = ['terrible', 'awful', 'broken', 'disappointed', 'waste', 'poor', 'bad']
        neutral_words = ['okay', 'decent', 'average', 'acceptable', 'fair', 'alright']
        
        def generate_review(rating):
            if rating >= 4:
                words = random.sample(positive_words, 3)
            elif rating <= 2:
                words = random.sample(negative_words, 3)
            else:
                words = random.sample(neutral_words, 3)
            return f"Product is {words[0]}. Quality is {words[1]}. Overall {words[2]}."
        
        reviews = [generate_review(r) for r in ratings]
        
        # Generate timestamps
        base_date = datetime(2020, 1, 1)
        timestamps = [base_date + timedelta(days=random.randint(0, 365)) for _ in range(n_reviews)]
        
        reviews_df = pd.DataFrame({
            'user_id': user_ids,
            'product_id': product_ids,
            'rating': ratings,
            'review_text': reviews,
            'timestamp': timestamps,
            'helpful_votes': np.random.poisson(2, n_reviews)
        }).drop_duplicates(subset=['user_id', 'product_id']).reset_index(drop=True)
        
        # Generate product metadata
        categories = ['Electronics', 'Computers', 'Camera', 'Phone', 'Audio', 
                      'Gaming', 'Accessories', 'Storage', 'Networking', 'Wearables']
        brands = ['TechPro', 'DigiMax', 'SmartGear', 'ElectroBrand', 'GadgetCo',
                  'InnoTech', 'PrimeTech', 'UltraDevice', 'CoreElec', 'NextGen']
        
        products_df = pd.DataFrame({
            'product_id': [f"P{i:05d}" for i in range(n_products)],
            'title': [f"Product_{i}" for i in range(n_products)],
            'category': np.random.choice(categories, n_products),
            'brand': np.random.choice(brands, n_products),
            'price': np.round(np.random.lognormal(4, 1, n_products), 2),
            'description': [f"High-quality {random.choice(categories).lower()} product with advanced features." 
                           for _ in range(n_products)]
        })
        
        print(f"   ✅ Generated {len(reviews_df)} unique reviews")
        
        return {
            'reviews': reviews_df,
            'products': products_df,
            'dataset_name': 'Amazon-Style'
        }
    
    def generate_bookcrossing_style(
        self,
        n_users: int = 10000,
        n_books: int = 5000,
        n_ratings: int = 100000
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate Book-Crossing-style dataset (very sparse)
        
        Returns:
            Dict with 'ratings', 'users', 'books' DataFrames
        """
        print(f"📚 Generating Book-Crossing-style dataset...")
        print(f"   Users: {n_users}, Books: {n_books}, Ratings: {n_ratings}")
        
        # Very sparse - most users rate few books
        user_weights = np.random.pareto(3.0, n_users) + 1
        user_weights = user_weights / user_weights.sum()
        
        book_weights = np.random.pareto(2.5, n_books) + 1
        book_weights = book_weights / book_weights.sum()
        
        user_ids = np.random.choice(range(1, n_users + 1), n_ratings, p=user_weights)
        book_ids = np.random.choice(range(1, n_books + 1), n_ratings, p=book_weights)
        
        # Book-Crossing has many 0 ratings (implicit) and explicit 1-10
        # Generate mix: 40% implicit (0), 60% explicit (1-10)
        is_implicit = np.random.choice([True, False], n_ratings, p=[0.4, 0.6])
        explicit_ratings = np.random.choice(range(1, 11), n_ratings, p=[0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.15, 0.2, 0.15, 0.1])
        ratings = np.where(is_implicit, 0, explicit_ratings)
        
        ratings_df = pd.DataFrame({
            'user_id': user_ids,
            'book_id': book_ids,
            'rating': ratings
        }).drop_duplicates(subset=['user_id', 'book_id']).reset_index(drop=True)
        
        # Generate user locations
        locations = [
            'New York, USA', 'London, UK', 'Paris, France', 'Berlin, Germany',
            'Tokyo, Japan', 'Sydney, Australia', 'Toronto, Canada', 'Mumbai, India',
            'São Paulo, Brazil', 'Moscow, Russia'
        ]
        ages = np.random.choice(range(15, 80), n_users)
        ages = np.where(ages < 18, np.nan, ages)  # Some missing ages
        
        users_df = pd.DataFrame({
            'user_id': range(1, n_users + 1),
            'location': np.random.choice(locations, n_users),
            'age': ages
        })
        
        # Generate book metadata
        authors = [f"Author_{i}" for i in range(500)]
        publishers = ['Penguin', 'HarperCollins', 'Simon & Schuster', 'Random House',
                      'Macmillan', 'Hachette', 'Wiley', 'Oxford Press', 'Cambridge', 'Bloomsbury']
        
        books_df = pd.DataFrame({
            'book_id': range(1, n_books + 1),
            'isbn': [f"978{random.randint(1000000000, 9999999999)}" for _ in range(n_books)],
            'title': [f"Book_{i}" for i in range(1, n_books + 1)],
            'author': np.random.choice(authors, n_books),
            'year': np.random.choice(range(1950, 2005), n_books),
            'publisher': np.random.choice(publishers, n_books)
        })
        
        print(f"   ✅ Generated {len(ratings_df)} unique ratings")
        sparsity = 1 - len(ratings_df) / (n_users * n_books)
        print(f"   📊 Matrix sparsity: {sparsity:.4%}")
        
        return {
            'ratings': ratings_df,
            'users': users_df,
            'books': books_df,
            'dataset_name': 'BookCrossing-Style'
        }
    
    def save_datasets(self, dataset: Dict[str, pd.DataFrame], prefix: str):
        """Save dataset DataFrames to CSV files"""
        for key, df in dataset.items():
            if isinstance(df, pd.DataFrame):
                filepath = os.path.join(self.output_dir, f"{prefix}_{key}.csv")
                df.to_csv(filepath, index=False)
                print(f"   💾 Saved {filepath}")
    
    def load_dataset(self, prefix: str) -> Dict[str, pd.DataFrame]:
        """Load dataset from saved CSV files"""
        result = {}
        for filename in os.listdir(self.output_dir):
            if filename.startswith(prefix) and filename.endswith('.csv'):
                key = filename.replace(f"{prefix}_", "").replace(".csv", "")
                filepath = os.path.join(self.output_dir, filename)
                result[key] = pd.read_csv(filepath)
        return result


def get_unified_format(dataset: Dict[str, pd.DataFrame], dataset_type: str) -> pd.DataFrame:
    """
    Convert any dataset to unified format for algorithm compatibility
    
    Returns DataFrame with columns: user_id, item_id, rating, [optional: timestamp, text]
    """
    if dataset_type == 'movielens':
        return dataset['ratings'].rename(columns={'item_id': 'item_id'})
    
    elif dataset_type == 'amazon':
        return dataset['reviews'].rename(columns={
            'product_id': 'item_id',
            'review_text': 'text'
        })[['user_id', 'item_id', 'rating', 'timestamp', 'text']]
    
    elif dataset_type == 'bookcrossing':
        df = dataset['ratings'].rename(columns={'book_id': 'item_id'})
        # Filter out implicit ratings (0) for explicit feedback algorithms
        return df[df['rating'] > 0].copy()
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def get_item_features(dataset: Dict[str, pd.DataFrame], dataset_type: str) -> pd.DataFrame:
    """Extract item features for content-based filtering"""
    if dataset_type == 'movielens':
        return dataset['items'].rename(columns={'item_id': 'item_id'})
    
    elif dataset_type == 'amazon':
        return dataset['products'].rename(columns={'product_id': 'item_id'})
    
    elif dataset_type == 'bookcrossing':
        return dataset['books'].rename(columns={'book_id': 'item_id'})
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("RECOMMENDATION SYSTEM DATA LOADER")
    print("=" * 60)
    
    generator = DatasetGenerator(output_dir="../data")
    
    # Generate all three datasets
    print("\n" + "=" * 60)
    movielens = generator.generate_movielens_style()
    generator.save_datasets(movielens, "movielens")
    
    print("\n" + "=" * 60)
    amazon = generator.generate_amazon_style()
    generator.save_datasets(amazon, "amazon")
    
    print("\n" + "=" * 60)
    bookcrossing = generator.generate_bookcrossing_style()
    generator.save_datasets(bookcrossing, "bookcrossing")
    
    print("\n" + "=" * 60)
    print("✅ All datasets generated successfully!")
    print("=" * 60)
    
    # Show statistics
    print("\n📊 Dataset Statistics:")
    for name, data in [('MovieLens', movielens), ('Amazon', amazon), ('BookCrossing', bookcrossing)]:
        ratings_key = 'ratings' if 'ratings' in data else 'reviews'
        df = data[ratings_key]
        n_users = df['user_id'].nunique()
        n_items = df['item_id'].nunique() if 'item_id' in df.columns else df['product_id'].nunique() if 'product_id' in df.columns else df['book_id'].nunique()
        n_interactions = len(df)
        sparsity = 1 - n_interactions / (n_users * n_items)
        print(f"   {name}: {n_users} users, {n_items} items, {n_interactions} interactions, {sparsity:.2%} sparse")
