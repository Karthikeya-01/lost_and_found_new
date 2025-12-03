"""
Train ML Models Script
File: scripts/train_ml_models.py

Trains KNN and SVM models and saves them for use in the app.
"""

import sys
from pathlib import Path
import pickle
import numpy as np

# Add project root to path BEFORE importing from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now import from src
from src.matching.ml_models import KNNMatcher, SVMMatcher


def load_features():
    """Load extracted features"""
    print("="*80)
    print("LOADING FEATURES")
    print("="*80)

    features_path = project_root / 'features' / 'feature_matrices.pkl'

    with open(features_path, 'rb') as f:
        feature_matrices = pickle.load(f)

    print(f"\n✓ Loaded feature matrices")
    print(f"  Lost items: {len(feature_matrices['lost']['ids'])}")
    print(f"  Found items: {len(feature_matrices['found']['ids'])}")

    return feature_matrices


def train_knn_model(feature_matrices, n_neighbors=5, metric='cosine'):
    """
    Train KNN model
    
    Args:
        feature_matrices: Feature data
        n_neighbors: Number of neighbors
        metric: Distance metric
        
    Returns:
        Trained KNN matcher
    """
    print("\n" + "="*80)
    print("TRAINING KNN MODEL")
    print("="*80)

    # Get found item features (these will be in the index)
    found_features = feature_matrices['found']['combined_features']
    found_ids = feature_matrices['found']['ids']
    found_categories = feature_matrices['found']['categories']

    # Create and train KNN matcher
    knn_matcher = KNNMatcher(n_neighbors=n_neighbors, metric=metric)
    knn_matcher.fit(found_features, found_ids, found_categories)

    # Test on a sample
    print("\n📊 Testing on sample lost item...")
    lost_features = feature_matrices['lost']['combined_features']
    lost_categories = feature_matrices['lost']['categories']

    if len(lost_features) > 0:
        test_matches = knn_matcher.predict(
            lost_features[0],
            lost_categories[0],
            filter_category=True
        )

        print(f"  Sample matches: {len(test_matches)}")
        if test_matches:
            print(
                f"  Top match similarity: {test_matches[0]['similarity']:.3f}")

    return knn_matcher


def train_svm_model(feature_matrices, kernel='rbf', C=1.0):
    """
    Train SVM model
    
    Args:
        feature_matrices: Feature data
        kernel: SVM kernel
        C: Regularization parameter
        
    Returns:
        Trained SVM matcher
    """
    print("\n" + "="*80)
    print("TRAINING SVM MODEL")
    print("="*80)

    # Get features
    lost_features = feature_matrices['lost']['combined_features']
    found_features = feature_matrices['found']['combined_features']

    lost_ids = feature_matrices['lost']['ids']
    found_ids = feature_matrices['found']['ids']

    lost_categories = feature_matrices['lost']['categories']
    found_categories = feature_matrices['found']['categories']

    # Create SVM matcher
    svm_matcher = SVMMatcher(kernel=kernel, C=C)

    # Create training pairs
    X_train, y_train = svm_matcher.create_training_pairs(
        lost_features, found_features,
        lost_ids, found_ids,
        lost_categories, found_categories,
        positive_ratio=0.3
    )

    # Train
    svm_matcher.fit(X_train, y_train)

    # Test on a sample
    print("\n📊 Testing on sample lost item...")
    if len(lost_features) > 0:
        test_matches = svm_matcher.predict(
            lost_features[0],
            found_features,
            found_ids,
            lost_categories[0],
            found_categories,
            filter_category=True,
            top_k=5
        )

        print(f"  Sample matches: {len(test_matches)}")
        if test_matches:
            print(
                f"  Top match probability: {test_matches[0]['similarity']:.3f}")

    return svm_matcher


def save_models(knn_matcher, svm_matcher):
    """Save trained models"""
    print("\n" + "="*80)
    print("SAVING MODELS")
    print("="*80)

    models_dir = project_root / 'models'
    models_dir.mkdir(exist_ok=True)

    # Save KNN
    knn_path = models_dir / 'knn_matcher.pkl'
    knn_matcher.save(knn_path)

    # Save SVM
    svm_path = models_dir / 'svm_matcher.pkl'
    svm_matcher.save(svm_path)

    print(f"\n✓ Models saved to: {models_dir}")


def compare_models(feature_matrices, knn_matcher, svm_matcher):
    """Compare model performance on sample items"""
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)

    lost_features = feature_matrices['lost']['combined_features']
    found_features = feature_matrices['found']['combined_features']
    lost_categories = feature_matrices['lost']['categories']
    found_categories = feature_matrices['found']['categories']
    found_ids = feature_matrices['found']['ids']

    # Test on first 3 lost items
    n_test = min(3, len(lost_features))

    for i in range(n_test):
        print(f"\n{'─'*80}")
        print(f"Lost Item {i+1} (Category: {lost_categories[i]})")
        print(f"{'─'*80}")

        # KNN predictions
        knn_matches = knn_matcher.predict(
            lost_features[i],
            lost_categories[i],
            filter_category=True
        )

        print("\n🔵 KNN Top 3 Matches:")
        for match in knn_matches[:3]:
            print(
                f"  {match['rank']}. {match['found_id']} - Similarity: {match['similarity']:.3f}")

        # SVM predictions
        svm_matches = svm_matcher.predict(
            lost_features[i],
            found_features,
            found_ids,
            lost_categories[i],
            found_categories,
            filter_category=True,
            top_k=5
        )

        print("\n🟢 SVM Top 3 Matches:")
        for match in svm_matches[:3]:
            print(
                f"  {match['rank']}. {match['found_id']} - Probability: {match['similarity']:.3f}")


def main():
    """Main training pipeline"""
    print("\n" + "="*80)
    print("ML MODELS TRAINING PIPELINE")
    print("="*80)

    # Load features
    feature_matrices = load_features()

    # Train KNN
    knn_matcher = train_knn_model(
        feature_matrices,
        n_neighbors=5,
        metric='cosine'
    )

    # Train SVM
    svm_matcher = train_svm_model(
        feature_matrices,
        kernel='rbf',
        C=1.0
    )

    # Save models
    save_models(knn_matcher, svm_matcher)

    # Compare models
    compare_models(feature_matrices, knn_matcher, svm_matcher)

    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)

    print("\n📁 Trained models saved:")
    print("  • models/knn_matcher.pkl")
    print("  • models/svm_matcher.pkl")

    print("\n🎯 Next steps:")
    print("  1. Models are ready to use in the Streamlit app")
    print("  2. Users can select between Similarity, KNN, or SVM")
    print("  3. Compare performance and choose the best model")


if __name__ == "__main__":
    main()
