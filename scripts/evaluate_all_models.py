"""
Comprehensive Model Evaluation Script
File: scripts/evaluate_all_models.py

Computes all relevant metrics for Similarity-based, KNN, and SVM models.
Requires ground truth file for accurate evaluation.
"""

import sys
from pathlib import Path
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    accuracy_score, confusion_matrix, 
    classification_report, roc_auc_score, 
    average_precision_score
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.matching.similarity import SimilarityComputer, FastSimilarityComputer
from src.matching.ranking import MatchRanker
from src.matching.ml_models import KNNMatcher, SVMMatcher


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, features_path, models_path, ground_truth_file=None):
        """
        Initialize evaluator
        
        Args:
            features_path: Path to features directory
            models_path: Path to models directory
            ground_truth_file: Path to ground truth JSON (optional)
        """
        self.features_path = Path(features_path)
        self.models_path = Path(models_path)
        self.ground_truth = None
        
        # Load ground truth if provided
        if ground_truth_file and Path(ground_truth_file).exists():
            with open(ground_truth_file, 'r') as f:
                self.ground_truth = json.load(f)
            print(f"✓ Loaded ground truth for {len(self.ground_truth)} items")
        else:
            print("⚠ No ground truth provided. Some metrics will not be available.")
        
        # Load features
        self.load_features()
        
        # Load models
        self.load_models()
    
    def load_features(self):
        """Load extracted features"""
        with open(self.features_path / 'feature_matrices.pkl', 'rb') as f:
            self.feature_matrices = pickle.load(f)
        
        with open(self.features_path / 'all_features.pkl', 'rb') as f:
            self.all_features = pickle.load(f)
        
        print(f"✓ Loaded features")
        print(f"  Lost items: {len(self.feature_matrices['lost']['ids'])}")
        print(f"  Found items: {len(self.feature_matrices['found']['ids'])}")
    
    def load_models(self):
        """Load trained models"""
        # Similarity-based
        self.similarity_computer = SimilarityComputer(weights={'image': 0.6, 'text': 0.4})
        print("✓ Initialized Similarity-based matcher")
        
        # KNN
        knn_path = self.models_path / 'knn_matcher.pkl'
        if knn_path.exists():
            self.knn_matcher = KNNMatcher.load(knn_path)
            print("✓ Loaded KNN matcher")
        else:
            self.knn_matcher = None
            print("⚠ KNN model not found")
        
        # SVM
        svm_path = self.models_path / 'svm_matcher.pkl'
        if svm_path.exists():
            self.svm_matcher = SVMMatcher.load(svm_path)
            print("✓ Loaded SVM matcher")
        else:
            self.svm_matcher = None
            print("⚠ SVM model not found")
    
    def compute_matches_similarity(self, top_k=5):
        """Compute matches using Similarity-based model"""
        print("\nComputing Similarity-based matches...")
        
        lost_features = [f for f in self.all_features if f['status'] == 'lost']
        found_features = [f for f in self.all_features if f['status'] == 'found']
        
        matches = {}
        for lost_item in lost_features:
            item_matches = []
            for found_item in found_features:
                match = self.similarity_computer.compute_match_score(
                    lost_item, found_item,
                    require_category_match=True,
                    require_temporal_validity=False
                )
                
                if match['valid']:
                    item_matches.append({
                        'found_id': found_item['id'],
                        'similarity': match['combined_similarity'],
                        'rank': None
                    })
            
            # Sort and rank
            item_matches.sort(key=lambda x: x['similarity'], reverse=True)
            for rank, match in enumerate(item_matches[:top_k], 1):
                match['rank'] = rank
            
            matches[lost_item['id']] = item_matches[:top_k]
        
        return matches
    
    def compute_matches_knn(self, top_k=5):
        """Compute matches using KNN model"""
        if self.knn_matcher is None:
            return None
        
        print("\nComputing KNN matches...")
        
        lost_features_list = [f for f in self.all_features if f['status'] == 'lost']
        
        matches = {}
        for lost_item in lost_features_list:
            new_features = np.concatenate([lost_item['image_features'], lost_item['text_features']])
            
            knn_matches = self.knn_matcher.predict(
                new_features,
                lost_item['category'],
                filter_category=True
            )
            
            matches[lost_item['id']] = knn_matches[:top_k]
        
        return matches
    
    def compute_matches_svm(self, top_k=5):
        """Compute matches using SVM model"""
        if self.svm_matcher is None:
            return None
        
        print("\nComputing SVM matches...")
        
        lost_features_list = [f for f in self.all_features if f['status'] == 'lost']
        found_features = self.feature_matrices['found']['combined_features']
        found_ids = self.feature_matrices['found']['ids']
        found_categories = self.feature_matrices['found']['categories']
        
        matches = {}
        for lost_item in lost_features_list:
            new_features = np.concatenate([lost_item['image_features'], lost_item['text_features']])
            
            svm_matches = self.svm_matcher.predict(
                new_features,
                found_features,
                found_ids,
                lost_item['category'],
                found_categories,
                filter_category=True,
                top_k=top_k
            )
            
            matches[lost_item['id']] = svm_matches
        
        return matches
    
    def compute_precision_at_k(self, matches, k_values=[1, 3, 5]):
        """Compute Precision@K"""
        if not self.ground_truth:
            return None
        
        precisions = {}
        
        for k in k_values:
            correct = 0
            total = 0
            
            for lost_id, item_matches in matches.items():
                if lost_id not in self.ground_truth:
                    continue
                
                correct_found_id = self.ground_truth[lost_id]
                top_k_matches = item_matches[:k]
                found_ids = [m['found_id'] for m in top_k_matches]
                
                if correct_found_id in found_ids:
                    correct += 1
                total += 1
            
            precisions[f'P@{k}'] = correct / total if total > 0 else 0.0
        
        return precisions
    
    def compute_recall_at_k(self, matches, k_values=[1, 3, 5]):
        """Compute Recall@K"""
        if not self.ground_truth:
            return None
        
        recalls = {}
        
        for k in k_values:
            correct = 0
            total = 0
            
            for lost_id, item_matches in matches.items():
                if lost_id not in self.ground_truth:
                    continue
                
                correct_found_id = self.ground_truth[lost_id]
                top_k_matches = item_matches[:k]
                found_ids = [m['found_id'] for m in top_k_matches]
                
                if correct_found_id in found_ids:
                    correct += 1
                total += 1
            
            recalls[f'R@{k}'] = correct / total if total > 0 else 0.0
        
        return recalls
    
    def compute_f1_at_k(self, precision, recall):
        """Compute F1@K from precision and recall"""
        f1_scores = {}
        
        for k in [1, 3, 5]:
            p = precision.get(f'P@{k}', 0)
            r = recall.get(f'R@{k}', 0)
            
            if p + r > 0:
                f1_scores[f'F1@{k}'] = 2 * (p * r) / (p + r)
            else:
                f1_scores[f'F1@{k}'] = 0.0
        
        return f1_scores
    
    def compute_mrr(self, matches):
        """Compute Mean Reciprocal Rank"""
        if not self.ground_truth:
            return None
        
        reciprocal_ranks = []
        
        for lost_id, item_matches in matches.items():
            if lost_id not in self.ground_truth:
                continue
            
            correct_found_id = self.ground_truth[lost_id]
            
            for rank, match in enumerate(item_matches, 1):
                if match['found_id'] == correct_found_id:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def compute_ndcg_at_k(self, matches, k=5):
        """Compute Normalized Discounted Cumulative Gain"""
        if not self.ground_truth:
            return None
        
        ndcg_scores = []
        
        for lost_id, item_matches in matches.items():
            if lost_id not in self.ground_truth:
                continue
            
            correct_found_id = self.ground_truth[lost_id]
            
            # Compute DCG
            dcg = 0.0
            for i, match in enumerate(item_matches[:k], 1):
                rel = 1 if match['found_id'] == correct_found_id else 0
                dcg += rel / np.log2(i + 1)
            
            # Compute IDCG (ideal DCG)
            idcg = 1.0 / np.log2(2)  # Best case: correct match at rank 1
            
            # NDCG
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    def compute_map(self, matches, k=5):
        """Compute Mean Average Precision"""
        if not self.ground_truth:
            return None
        
        ap_scores = []
        
        for lost_id, item_matches in matches.items():
            if lost_id not in self.ground_truth:
                continue
            
            correct_found_id = self.ground_truth[lost_id]
            
            # Compute Average Precision
            relevant_found = 0
            ap = 0.0
            
            for i, match in enumerate(item_matches[:k], 1):
                if match['found_id'] == correct_found_id:
                    relevant_found += 1
                    ap += relevant_found / i
            
            ap = ap / 1 if relevant_found > 0 else 0.0  # Only 1 relevant item
            ap_scores.append(ap)
        
        return np.mean(ap_scores) if ap_scores else 0.0
    
    def compute_top_k_accuracy(self, matches, k=1):
        """Compute Top-K Accuracy"""
        if not self.ground_truth:
            return None
        
        correct = 0
        total = 0
        
        for lost_id, item_matches in matches.items():
            if lost_id not in self.ground_truth:
                continue
            
            correct_found_id = self.ground_truth[lost_id]
            top_k_matches = item_matches[:k]
            found_ids = [m['found_id'] for m in top_k_matches]
            
            if correct_found_id in found_ids:
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0.0
    
    def compute_confidence_metrics(self, matches):
        """Compute confidence-based metrics"""
        similarities = []
        
        for lost_id, item_matches in matches.items():
            for match in item_matches:
                similarities.append(match.get('similarity', 0.0))
        
        if not similarities:
            return {}
        
        return {
            'mean_similarity': np.mean(similarities),
            'median_similarity': np.median(similarities),
            'std_similarity': np.std(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities),
            'q25_similarity': np.percentile(similarities, 25),
            'q75_similarity': np.percentile(similarities, 75)
        }
    
    def compute_z_scores(self, matches):
        """Compute z-scores for similarity distributions"""
        similarities = []
        
        for lost_id, item_matches in matches.items():
            for match in item_matches:
                similarities.append(match.get('similarity', 0.0))
        
        if len(similarities) < 2:
            return None
        
        mean = np.mean(similarities)
        std = np.std(similarities)
        
        if std == 0:
            return None
        
        z_scores = [(s - mean) / std for s in similarities]
        
        return {
            'mean_z_score': np.mean(z_scores),
            'std_z_score': np.std(z_scores),
            'min_z_score': np.min(z_scores),
            'max_z_score': np.max(z_scores),
            'samples_above_2std': sum(1 for z in z_scores if abs(z) > 2)
        }
    
    def evaluate_model(self, model_name, matches):
        """Evaluate a single model"""
        print(f"\n{'='*80}")
        print(f"EVALUATING {model_name.upper()}")
        print(f"{'='*80}")
        
        metrics = {
            'model': model_name,
            'total_items': len(matches)
        }
        
        # Ranking metrics (require ground truth)
        if self.ground_truth:
            # Precision@K
            precision = self.compute_precision_at_k(matches)
            metrics.update(precision)
            
            # Recall@K
            recall = self.compute_recall_at_k(matches)
            metrics.update(recall)
            
            # F1@K
            f1 = self.compute_f1_at_k(precision, recall)
            metrics.update(f1)
            
            # MRR
            metrics['MRR'] = self.compute_mrr(matches)
            
            # NDCG@5
            metrics['NDCG@5'] = self.compute_ndcg_at_k(matches, k=5)
            
            # MAP@5
            metrics['MAP@5'] = self.compute_map(matches, k=5)
            
            # Top-K Accuracy
            metrics['Top-1 Accuracy'] = self.compute_top_k_accuracy(matches, k=1)
            metrics['Top-3 Accuracy'] = self.compute_top_k_accuracy(matches, k=3)
            metrics['Top-5 Accuracy'] = self.compute_top_k_accuracy(matches, k=5)
        
        # Confidence metrics (always available)
        confidence = self.compute_confidence_metrics(matches)
        metrics.update(confidence)
        
        # Z-scores (always available)
        z_scores = self.compute_z_scores(matches)
        if z_scores:
            metrics.update(z_scores)
        
        return metrics
    
    def create_comparison_table(self, all_metrics):
        """Create comparison table of all models"""
        df = pd.DataFrame(all_metrics)
        
        # Reorder columns for better readability
        primary_cols = ['model', 'total_items']
        ranking_cols = ['P@1', 'P@3', 'P@5', 'R@1', 'R@3', 'R@5', 
                       'F1@1', 'F1@3', 'F1@5', 'MRR', 'NDCG@5', 'MAP@5',
                       'Top-1 Accuracy', 'Top-3 Accuracy', 'Top-5 Accuracy']
        confidence_cols = ['mean_similarity', 'median_similarity', 'std_similarity']
        
        # Only include columns that exist
        cols = primary_cols + [c for c in ranking_cols if c in df.columns] + \
               [c for c in confidence_cols if c in df.columns]
        
        return df[cols]
    
    def plot_comparison(self, all_metrics, output_dir):
        """Create comparison plots"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(all_metrics)
        
        # Plot 1: Precision, Recall, F1 comparison
        if 'P@1' in df.columns:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Precision@K
            p_cols = ['P@1', 'P@3', 'P@5']
            df_p = df[['model'] + p_cols].set_index('model')
            df_p.plot(kind='bar', ax=axes[0], rot=0)
            axes[0].set_title('Precision@K')
            axes[0].set_ylabel('Precision')
            axes[0].set_ylim([0, 1])
            axes[0].legend(title='K')
            axes[0].grid(axis='y', alpha=0.3)
            
            # Recall@K
            r_cols = ['R@1', 'R@3', 'R@5']
            df_r = df[['model'] + r_cols].set_index('model')
            df_r.plot(kind='bar', ax=axes[1], rot=0)
            axes[1].set_title('Recall@K')
            axes[1].set_ylabel('Recall')
            axes[1].set_ylim([0, 1])
            axes[1].legend(title='K')
            axes[1].grid(axis='y', alpha=0.3)
            
            # F1@K
            f1_cols = ['F1@1', 'F1@3', 'F1@5']
            df_f1 = df[['model'] + f1_cols].set_index('model')
            df_f1.plot(kind='bar', ax=axes[2], rot=0)
            axes[2].set_title('F1-Score@K')
            axes[2].set_ylabel('F1-Score')
            axes[2].set_ylim([0, 1])
            axes[2].legend(title='K')
            axes[2].grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'precision_recall_f1.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_dir / 'precision_recall_f1.png'}")
            plt.close()
        
        # Plot 2: Ranking metrics
        if 'MRR' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ranking_metrics = ['MRR', 'NDCG@5', 'MAP@5', 'Top-1 Accuracy']
            df_ranking = df[['model'] + ranking_metrics].set_index('model')
            df_ranking.plot(kind='bar', ax=ax, rot=0)
            ax.set_title('Ranking Metrics Comparison')
            ax.set_ylabel('Score')
            ax.set_ylim([0, 1])
            ax.legend(title='Metric')
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'ranking_metrics.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_dir / 'ranking_metrics.png'}")
            plt.close()
        
        # Plot 3: Confidence metrics
        if 'mean_similarity' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = np.arange(len(df))
            width = 0.25
            
            ax.bar(x - width, df['mean_similarity'], width, label='Mean', alpha=0.8)
            ax.bar(x, df['median_similarity'], width, label='Median', alpha=0.8)
            ax.bar(x + width, df['std_similarity'], width, label='Std Dev', alpha=0.8)
            
            ax.set_xlabel('Model')
            ax.set_ylabel('Similarity Score')
            ax.set_title('Similarity Score Statistics')
            ax.set_xticks(x)
            ax.set_xticklabels(df['model'])
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'confidence_metrics.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_dir / 'confidence_metrics.png'}")
            plt.close()
    
    def run_complete_evaluation(self, top_k=5):
        """Run complete evaluation for all models"""
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*80)
        
        all_metrics = []
        
        # Similarity-based
        sim_matches = self.compute_matches_similarity(top_k=top_k)
        sim_metrics = self.evaluate_model('Similarity-based', sim_matches)
        all_metrics.append(sim_metrics)
        
        # KNN
        if self.knn_matcher:
            knn_matches = self.compute_matches_knn(top_k=top_k)
            knn_metrics = self.evaluate_model('KNN', knn_matches)
            all_metrics.append(knn_metrics)
        
        # SVM
        if self.svm_matcher:
            svm_matches = self.compute_matches_svm(top_k=top_k)
            svm_metrics = self.evaluate_model('SVM', svm_matches)
            all_metrics.append(svm_metrics)
        
        # Create comparison table
        print("\n" + "="*80)
        print("COMPARISON TABLE")
        print("="*80)
        
        comparison_df = self.create_comparison_table(all_metrics)
        print("\n" + comparison_df.to_string(index=False))
        
        # Save results
        output_dir = project_root / 'evaluation'
        output_dir.mkdir(exist_ok=True)
        
        # Save CSV
        comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
        print(f"\n✓ Saved: {output_dir / 'model_comparison.csv'}")
        
        # Save detailed JSON
        with open(output_dir / 'detailed_metrics.json', 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"✓ Saved: {output_dir / 'detailed_metrics.json'}")
        
        # Create plots
        self.plot_comparison(all_metrics, output_dir)
        
        return all_metrics, comparison_df


def main():
    """Main evaluation pipeline"""
    # Configure paths
    FEATURES_PATH = project_root / 'features'
    MODELS_PATH = project_root / 'models'
    GROUND_TRUTH_FILE = project_root / 'dataset' / 'ground_truth_75.json'  # Optional
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        FEATURES_PATH,
        MODELS_PATH,
        GROUND_TRUTH_FILE if GROUND_TRUTH_FILE.exists() else None
    )
    
    # Run evaluation
    all_metrics, comparison_df = evaluator.run_complete_evaluation(top_k=5)
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)
    
    print("\n📊 Results saved to: evaluation/")
    print("  • model_comparison.csv")
    print("  • detailed_metrics.json")
    print("  • precision_recall_f1.png")
    print("  • ranking_metrics.png")
    print("  • confidence_metrics.png")
    
    if not evaluator.ground_truth:
        print("\n💡 Note: Ground truth file not found.")
        print("   Create 'dataset/ground_truth.json' for full metrics:")
        print('   {"L001": "F042", "L002": "F015", ...}')


if __name__ == "__main__":
    main()