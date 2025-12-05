import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from itertools import product

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))
# Add project root to path
project_root = Path(__file__).parent.parent

class MatchingEvaluator:

    def __init__(self, results_path):
        self.results_path = Path(results_path)
        self.results = None
        self.ground_truth = None

    def load_results(self):
        results_file = self.results_path / 'matching_results.pkl'
        with open(results_file, 'rb') as f:
            self.results = pickle.load(f)

        print(f"✓ Loaded matching results")
        print(f"  Method: {self.results['method']}")
        print(f"  Lost items: {len(self.results['matches'])}")

    def load_ground_truth(self, ground_truth_file):
        with open(ground_truth_file, 'r') as f:
            self.ground_truth = json.load(f)

        print(f"✓ Loaded ground truth for {len(self.ground_truth)} items")

    def compute_precision_at_k(self, k=None):
        if not self.ground_truth:
            print("⚠ Ground truth not available")
            return None

        precisions = []

        for lost_id, item_matches in self.results['matches'].items():
            if lost_id not in self.ground_truth:
                continue

            correct_found_id = self.ground_truth[lost_id]

            # Get top-K matches
            if k:
                top_k_matches = item_matches[:k]
            else:
                top_k_matches = item_matches

            # Check if correct match is in top-K
            found_ids = [m.get('found_id') for m in top_k_matches]

            if correct_found_id in found_ids:
                precision = 1.0
            else:
                precision = 0.0

            precisions.append(precision)

        avg_precision = np.mean(precisions) if precisions else 0.0

        return {
            'precision@k': avg_precision,
            'k': k or len(item_matches),
            'n_items': len(precisions)
        }

    def compute_recall_at_k(self, k=None):
        if not self.ground_truth:
            print("⚠ Ground truth not available")
            return None

        correct_in_top_k = 0
        total = 0

        for lost_id, item_matches in self.results['matches'].items():
            if lost_id not in self.ground_truth:
                continue

            total += 1
            correct_found_id = self.ground_truth[lost_id]

            # Get top-K matches
            if k:
                top_k_matches = item_matches[:k]
            else:
                top_k_matches = item_matches

            found_ids = [m.get('found_id') for m in top_k_matches]

            if correct_found_id in found_ids:
                correct_in_top_k += 1

        recall = correct_in_top_k / total if total > 0 else 0.0

        return {
            'recall@k': recall,
            'k': k or len(item_matches),
            'correct_in_top_k': correct_in_top_k,
            'total': total
        }

    def compute_mean_reciprocal_rank(self):
        if not self.ground_truth:
            print("⚠ Ground truth not available")
            return None

        reciprocal_ranks = []

        for lost_id, item_matches in self.results['matches'].items():
            if lost_id not in self.ground_truth:
                continue

            correct_found_id = self.ground_truth[lost_id]

            # Find rank of correct match
            for i, match in enumerate(item_matches, 1):
                if match.get('found_id') == correct_found_id:
                    reciprocal_ranks.append(1.0 / i)
                    break
            else:
                # Correct match not found in list
                reciprocal_ranks.append(0.0)

        mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

        return {
            'mrr': mrr,
            'n_items': len(reciprocal_ranks)
        }

    def compute_top_1_accuracy(self):
        if not self.ground_truth:
            print("⚠ Ground truth not available")
            return None

        correct = 0
        total = 0

        for lost_id, item_matches in self.results['matches'].items():
            if lost_id not in self.ground_truth:
                continue

            total += 1

            if not item_matches:
                continue

            correct_found_id = self.ground_truth[lost_id]
            top_match_id = item_matches[0].get('found_id')

            if top_match_id == correct_found_id:
                correct += 1

        accuracy = correct / total if total > 0 else 0.0

        return {
            'top_1_accuracy': accuracy,
            'correct': correct,
            'total': total
        }

    def evaluate_all_metrics(self, k_values=[1, 3, 5, 10]):
        print(f"\n{'='*80}")
        print("EVALUATION METRICS")
        print(f"{'='*80}\n")

        if not self.ground_truth:
            print(
                "⚠ Ground truth not available. Please provide ground truth for evaluation.")
            print("  Create a JSON file mapping lost_id -> correct_found_id")
            return None

        metrics = {}

        # Top-1 Accuracy
        top1 = self.compute_top_1_accuracy()
        metrics['top_1_accuracy'] = top1
        print(
            f"Top-1 Accuracy: {top1['top_1_accuracy']:.3f} ({top1['correct']}/{top1['total']})")

        # MRR
        mrr = self.compute_mean_reciprocal_rank()
        metrics['mrr'] = mrr
        print(f"Mean Reciprocal Rank: {mrr['mrr']:.3f}")

        print()

        # Precision and Recall at different K
        for k in k_values:
            precision = self.compute_precision_at_k(k)
            recall = self.compute_recall_at_k(k)

            metrics[f'precision@{k}'] = precision
            metrics[f'recall@{k}'] = recall

            print(f"Precision@{k}: {precision['precision@k']:.3f}")
            print(f"Recall@{k}: {recall['recall@k']:.3f}")
            print()

        return metrics

    def analyze_similarity_distribution(self):
        print(f"\n{'='*80}")
        print("SIMILARITY DISTRIBUTION ANALYSIS")
        print(f"{'='*80}\n")

        all_similarities = []
        best_similarities = []

        for lost_id, item_matches in self.results['matches'].items():
            for match in item_matches:
                sim = match.get('similarity') or match.get(
                    'combined_similarity', 0)
                all_similarities.append(sim)

            if item_matches:
                best_sim = item_matches[0].get(
                    'similarity') or item_matches[0].get('combined_similarity', 0)
                best_similarities.append(best_sim)

        if not all_similarities:
            print("No similarities to analyze")
            return

        print(f"All Matches ({len(all_similarities)} total):")
        print(f"  Mean: {np.mean(all_similarities):.3f}")
        print(f"  Median: {np.median(all_similarities):.3f}")
        print(f"  Std Dev: {np.std(all_similarities):.3f}")
        print(f"  Min: {np.min(all_similarities):.3f}")
        print(f"  Max: {np.max(all_similarities):.3f}")

        print(f"\nBest Matches ({len(best_similarities)} items):")
        print(f"  Mean: {np.mean(best_similarities):.3f}")
        print(f"  Median: {np.median(best_similarities):.3f}")
        print(f"  Std Dev: {np.std(best_similarities):.3f}")

        # Plot distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # All similarities
        axes[0].hist(all_similarities, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(all_similarities), color='red', linestyle='--',
                        label=f'Mean: {np.mean(all_similarities):.3f}')
        axes[0].set_xlabel('Similarity Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of All Match Similarities')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Best similarities
        axes[1].hist(best_similarities, bins=30,
                     edgecolor='black', alpha=0.7, color='green')
        axes[1].axvline(np.mean(best_similarities), color='red', linestyle='--',
                        label=f'Mean: {np.mean(best_similarities):.3f}')
        axes[1].set_xlabel('Similarity Score')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Best Match Similarities')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()

        output_dir = self.results_path.parent / 'evaluation'
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / 'similarity_distribution.png',
                    dpi=300, bbox_inches='tight')
        print(
            f"\n✓ Saved plot to: {output_dir / 'similarity_distribution.png'}")
        plt.show()

    def analyze_matches_by_category(self):
        print(f"\n{'='*80}")
        print("MATCHES BY CATEGORY")
        print(f"{'='*80}\n")

        # Load features to get categories
        features_file = self.results_path.parent / 'all_features.pkl'
        with open(features_file, 'rb') as f:
            all_features = pickle.load(f)

        # Map IDs to categories
        id_to_category = {f['id']: f['category'] for f in all_features}

        # Analyze by category
        category_stats = {}

        for lost_id, item_matches in self.results['matches'].items():
            category = id_to_category.get(lost_id, 'unknown')

            if category not in category_stats:
                category_stats[category] = {
                    'count': 0,
                    'with_matches': 0,
                    'avg_similarity': [],
                    'num_matches': []
                }

            category_stats[category]['count'] += 1

            if item_matches:
                category_stats[category]['with_matches'] += 1
                best_sim = item_matches[0].get(
                    'similarity') or item_matches[0].get('combined_similarity', 0)
                category_stats[category]['avg_similarity'].append(best_sim)
                category_stats[category]['num_matches'].append(
                    len(item_matches))

        # Print statistics
        for category in sorted(category_stats.keys()):
            stats = category_stats[category]
            print(f"{category}:")
            print(f"  Total lost items: {stats['count']}")
            print(
                f"  With matches: {stats['with_matches']} ({stats['with_matches']/stats['count']*100:.1f}%)")
            if stats['avg_similarity']:
                print(
                    f"  Avg best similarity: {np.mean(stats['avg_similarity']):.3f}")
                print(
                    f"  Avg num matches: {np.mean(stats['num_matches']):.1f}")
            print()

    def suggest_optimal_threshold(self):
        all_similarities = []

        for lost_id, item_matches in self.results['matches'].items():
            for match in item_matches:
                sim = match.get('similarity') or match.get(
                    'combined_similarity', 0)
                all_similarities.append(sim)

        if not all_similarities:
            return

        mean = np.mean(all_similarities)
        median = np.median(all_similarities)
        q1 = np.percentile(all_similarities, 25)
        q3 = np.percentile(all_similarities, 75)

        print(f"\n{'='*80}")
        print("THRESHOLD RECOMMENDATIONS")
        print(f"{'='*80}\n")

        print("Based on similarity distribution:")
        print(f"  Conservative (Q3): {q3:.3f} - High precision, lower recall")
        print(
            f"  Balanced (Median): {median:.3f} - Balance between precision and recall")
        print(f"  Aggressive (Q1): {q1:.3f} - Higher recall, lower precision")
        print(f"  Mean: {mean:.3f}")

        print(
            f"\nCurrent threshold: {self.results['config']['min_similarity']}")


# Example usage
if __name__ == "__main__":
    # Configure paths
    RESULTS_PATH = Path(__file__).parent.parent / 'features' / 'matches'

    # Initialize evaluator
    evaluator = MatchingEvaluator(RESULTS_PATH)

    # Load results
    evaluator.load_results()

    # Analyze without ground truth
    print("\n" + "="*80)
    print("ANALYSIS WITHOUT GROUND TRUTH")
    print("="*80)

    evaluator.analyze_similarity_distribution()
    evaluator.analyze_matches_by_category()
    evaluator.suggest_optimal_threshold()

    GROUND_TRUTH_FILE = project_root / 'dataset' / 'ground_truth_75.json'  # Optional

    #uncomment these lines, after creating ground_truth.json to evaluate with ground truth:
    evaluator.load_ground_truth(GROUND_TRUTH_FILE)
    metrics = evaluator.evaluate_all_metrics(k_values=[1, 3, 5, 10])

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

    print("\n💡 To evaluate with ground truth:")
    print("  1. Create ground_truth.json with format:")
    print(
        '     {"lost_id_1": "correct_found_id_1", "lost_id_2": "correct_found_id_2", ...}')
    print("  2. Run: evaluator.load_ground_truth('path/to/ground_truth.json')")
    print("  3. Run: evaluator.evaluate_all_metrics()")
