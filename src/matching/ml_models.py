"""
ML Models Module
File: src/matching/ml_models.py

Implements K-Nearest Neighbors and SVM for matching lost and found items.
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class KNNMatcher:
    """K-Nearest Neighbors based matching"""

    def __init__(self, n_neighbors=5, metric='cosine'):
        """
        Initialize KNN matcher
        
        Args:
            n_neighbors: Number of neighbors to retrieve
            metric: Distance metric ('cosine', 'euclidean', 'manhattan')
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.model = None
        self.found_features = None
        self.found_ids = None
        self.found_categories = None
        self.scaler = StandardScaler()

    def fit(self, found_features, found_ids, found_categories=None):
        """
        Fit KNN model on found items
        
        Args:
            found_features: Feature matrix of found items (N, D)
            found_ids: List of found item IDs
            found_categories: List of found item categories
        """
        print(f"\nTraining KNN model...")
        print(f"  Features shape: {found_features.shape}")
        print(f"  Metric: {self.metric}")
        print(f"  Neighbors: {self.n_neighbors}")

        # Normalize features
        self.found_features = self.scaler.fit_transform(found_features)
        self.found_ids = found_ids
        self.found_categories = found_categories

        # Create and fit KNN model
        self.model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, len(found_ids)),
            metric=self.metric,
            algorithm='auto'
        )

        self.model.fit(self.found_features)
        print(f"✓ KNN model trained successfully")

    def predict(self, lost_feature, lost_category=None, filter_category=True):
        """
        Find matches for a lost item
        
        Args:
            lost_feature: Feature vector of lost item (1, D)
            lost_category: Category of lost item
            filter_category: Whether to filter by category
            
        Returns:
            List of matches with IDs, distances, and ranks
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Normalize lost item features
        lost_feature_normalized = self.scaler.transform(
            lost_feature.reshape(1, -1))

        # Find neighbors
        distances, indices = self.model.kneighbors(lost_feature_normalized)

        # Convert to similarity scores (1 / (1 + distance))
        similarities = 1 / (1 + distances[0])

        # Create matches
        matches = []
        for rank, (idx, similarity, distance) in enumerate(zip(indices[0], similarities, distances[0]), 1):
            found_id = self.found_ids[idx]
            found_category = self.found_categories[idx] if self.found_categories is not None else None

            # Category filtering
            if filter_category and lost_category and found_category:
                if lost_category != found_category:
                    continue

            matches.append({
                'rank': rank,
                'found_id': found_id,
                'similarity': float(similarity),
                'distance': float(distance),
                'found_category': found_category
            })

        return matches

    def predict_batch(self, lost_features, lost_categories=None, filter_category=True):
        """
        Find matches for multiple lost items
        
        Args:
            lost_features: Feature matrix (M, D)
            lost_categories: List of categories
            filter_category: Whether to filter by category
            
        Returns:
            Dictionary mapping indices to matches
        """
        all_matches = {}

        for i in range(len(lost_features)):
            lost_feature = lost_features[i]
            lost_category = lost_categories[i] if lost_categories is not None else None

            matches = self.predict(
                lost_feature, lost_category, filter_category)
            all_matches[i] = matches

        return all_matches

    def save(self, filepath):
        """Save model to file"""
        model_data = {
            'model': self.model,
            'found_features': self.found_features,
            'found_ids': self.found_ids,
            'found_categories': self.found_categories,
            'scaler': self.scaler,
            'n_neighbors': self.n_neighbors,
            'metric': self.metric
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✓ KNN model saved to: {filepath}")

    @classmethod
    def load(cls, filepath):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        matcher = cls(
            n_neighbors=model_data['n_neighbors'],
            metric=model_data['metric']
        )

        matcher.model = model_data['model']
        matcher.found_features = model_data['found_features']
        matcher.found_ids = model_data['found_ids']
        matcher.found_categories = model_data['found_categories']
        matcher.scaler = model_data['scaler']

        print(f"✓ KNN model loaded from: {filepath}")
        return matcher


class SVMMatcher:
    """SVM-based matching classifier"""

    def __init__(self, kernel='rbf', C=1.0):
        """
        Initialize SVM matcher
        
        Args:
            kernel: SVM kernel ('rbf', 'linear', 'poly')
            C: Regularization parameter
        """
        self.kernel = kernel
        self.C = C
        self.model = None
        self.scaler = StandardScaler()
        self.trained = False

    def create_training_pairs(self, lost_features, found_features,
                              lost_ids, found_ids,
                              lost_categories, found_categories,
                              positive_ratio=0.3):
        """
        Create training pairs for SVM
        
        Args:
            lost_features: Lost item features (N, D)
            found_features: Found item features (M, D)
            lost_ids: Lost item IDs
            found_ids: Found item IDs
            lost_categories: Lost item categories
            found_categories: Found item categories
            positive_ratio: Ratio of positive to negative samples
            
        Returns:
            X_train, y_train
        """
        print("\nCreating training pairs for SVM...")

        X_pairs = []
        y_labels = []

        # Create positive pairs (same category items - assumed matches)
        print("  Creating positive pairs (same category)...")
        for i, (lost_feat, lost_cat) in enumerate(zip(lost_features, lost_categories)):
            # Find items in same category
            same_category_indices = [
                j for j, found_cat in enumerate(found_categories)
                if found_cat == lost_cat
            ]

            if same_category_indices:
                # Sample a few positive pairs
                n_positive = min(3, len(same_category_indices))
                sampled_indices = np.random.choice(
                    same_category_indices, n_positive, replace=False)

                for j in sampled_indices:
                    found_feat = found_features[j]

                    # Create feature vector: [lost_feat, found_feat, difference, product]
                    pair_features = self._create_pair_features(
                        lost_feat, found_feat)
                    X_pairs.append(pair_features)
                    y_labels.append(1)  # Positive match

        n_positive = len(y_labels)
        print(f"  ✓ Created {n_positive} positive pairs")

        # Create negative pairs (different category or random)
        print("  Creating negative pairs (different category)...")
        n_negative_needed = int(n_positive / positive_ratio) - n_positive

        for _ in range(n_negative_needed):
            # Random lost and found items
            i = np.random.randint(0, len(lost_features))
            j = np.random.randint(0, len(found_features))

            # Prefer different categories for negative samples
            if lost_categories[i] == found_categories[j]:
                # Try to find different category
                different_cat_indices = [
                    idx for idx, cat in enumerate(found_categories)
                    if cat != lost_categories[i]
                ]
                if different_cat_indices:
                    j = np.random.choice(different_cat_indices)

            lost_feat = lost_features[i]
            found_feat = found_features[j]

            pair_features = self._create_pair_features(lost_feat, found_feat)
            X_pairs.append(pair_features)
            y_labels.append(0)  # Negative match

        print(f"  ✓ Created {n_negative_needed} negative pairs")

        X_train = np.array(X_pairs)
        y_train = np.array(y_labels)

        print(f"\n  Total training pairs: {len(X_train)}")
        print(
            f"  Positive samples: {np.sum(y_train == 1)} ({np.sum(y_train == 1)/len(y_train)*100:.1f}%)")
        print(
            f"  Negative samples: {np.sum(y_train == 0)} ({np.sum(y_train == 0)/len(y_train)*100:.1f}%)")

        return X_train, y_train

    def _create_pair_features(self, feat1, feat2):
        """
        Create feature vector from pair of items
        
        Combines: [feat1, feat2, |feat1-feat2|, feat1*feat2]
        """
        difference = np.abs(feat1 - feat2)
        product = feat1 * feat2

        # Concatenate all features
        pair_features = np.concatenate([feat1, feat2, difference, product])

        return pair_features

    def fit(self, X_train, y_train):
        """
        Train SVM classifier
        
        Args:
            X_train: Training features (N, D)
            y_train: Training labels (N,)
        """
        print(f"\nTraining SVM classifier...")
        print(f"  Kernel: {self.kernel}")
        print(f"  C: {self.C}")

        # Normalize features
        X_train_normalized = self.scaler.fit_transform(X_train)

        # Train SVM
        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            probability=True,  # Enable probability estimates
            random_state=42
        )

        self.model.fit(X_train_normalized, y_train)
        self.trained = True

        # Training accuracy
        train_acc = self.model.score(X_train_normalized, y_train)
        print(f"✓ SVM model trained successfully")
        print(f"  Training accuracy: {train_acc:.3f}")

    def predict_match_probability(self, lost_feature, found_feature):
        """
        Predict match probability for a pair
        
        Args:
            lost_feature: Lost item feature vector
            found_feature: Found item feature vector
            
        Returns:
            Match probability (0-1)
        """
        if not self.trained:
            raise ValueError("Model not trained. Call fit() first.")

        # Create pair features
        pair_features = self._create_pair_features(lost_feature, found_feature)
        pair_features_normalized = self.scaler.transform(
            pair_features.reshape(1, -1))

        # Get probability
        prob = self.model.predict_proba(pair_features_normalized)[
            0][1]  # Probability of class 1 (match)

        return prob

    def predict(self, lost_feature, found_features, found_ids,
                lost_category=None, found_categories=None,
                filter_category=True, top_k=5):
        """
        Find matches for a lost item
        
        Args:
            lost_feature: Lost item feature vector
            found_features: All found item features (N, D)
            found_ids: Found item IDs
            lost_category: Lost item category
            found_categories: Found item categories
            filter_category: Whether to filter by category
            top_k: Number of top matches to return
            
        Returns:
            List of matches with IDs and probabilities
        """
        if not self.trained:
            raise ValueError("Model not trained. Call fit() first.")

        matches = []

        for idx, (found_feat, found_id) in enumerate(zip(found_features, found_ids)):
            found_category = found_categories[idx] if found_categories is not None else None

            # Category filtering
            if filter_category and lost_category and found_category:
                if lost_category != found_category:
                    continue

            # Get match probability
            prob = self.predict_match_probability(lost_feature, found_feat)

            matches.append({
                'found_id': found_id,
                'similarity': float(prob),
                'found_category': found_category
            })

        # Sort by probability and get top-K
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        top_matches = matches[:top_k]

        # Add ranks
        for rank, match in enumerate(top_matches, 1):
            match['rank'] = rank

        return top_matches

    def save(self, filepath):
        """Save model to file"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'kernel': self.kernel,
            'C': self.C,
            'trained': self.trained
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✓ SVM model saved to: {filepath}")

    @classmethod
    def load(cls, filepath):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        matcher = cls(
            kernel=model_data['kernel'],
            C=model_data['C']
        )

        matcher.model = model_data['model']
        matcher.scaler = model_data['scaler']
        matcher.trained = model_data['trained']

        print(f"✓ SVM model loaded from: {filepath}")
        return matcher


# Example usage and testing
if __name__ == "__main__":
    print("="*80)
    print("ML MODELS TEST")
    print("="*80)

    # Generate sample data
    np.random.seed(42)

    # Simulated features
    n_lost = 20
    n_found = 50
    n_features = 310  # 184 image + 126 text

    lost_features = np.random.rand(n_lost, n_features)
    found_features = np.random.rand(n_found, n_features)

    lost_ids = [f"L{i:03d}" for i in range(n_lost)]
    found_ids = [f"F{i:03d}" for i in range(n_found)]

    categories = ['bags', 'books', 'keys']
    lost_categories = [categories[i % len(categories)] for i in range(n_lost)]
    found_categories = [categories[i %
                                   len(categories)] for i in range(n_found)]

    # Test KNN
    print("\n" + "="*80)
    print("TESTING KNN MATCHER")
    print("="*80)

    knn_matcher = KNNMatcher(n_neighbors=5, metric='cosine')
    knn_matcher.fit(found_features, found_ids, found_categories)

    # Test prediction
    test_lost_feature = lost_features[0]
    test_lost_category = lost_categories[0]

    knn_matches = knn_matcher.predict(
        test_lost_feature, test_lost_category, filter_category=True)

    print(f"\nKNN Matches for lost item (category: {test_lost_category}):")
    for match in knn_matches[:3]:
        print(
            f"  Rank {match['rank']}: {match['found_id']} - Similarity: {match['similarity']:.3f}")

    # Test SVM
    print("\n" + "="*80)
    print("TESTING SVM MATCHER")
    print("="*80)

    svm_matcher = SVMMatcher(kernel='rbf', C=1.0)

    # Create training pairs
    X_train, y_train = svm_matcher.create_training_pairs(
        lost_features, found_features,
        lost_ids, found_ids,
        lost_categories, found_categories
    )

    # Train
    svm_matcher.fit(X_train, y_train)

    # Test prediction
    svm_matches = svm_matcher.predict(
        test_lost_feature, found_features, found_ids,
        test_lost_category, found_categories,
        filter_category=True, top_k=5
    )

    print(f"\nSVM Matches for lost item (category: {test_lost_category}):")
    for match in svm_matches[:3]:
        print(
            f"  Rank {match['rank']}: {match['found_id']} - Probability: {match['similarity']:.3f}")

    print("\n✓ ML models test complete!")
