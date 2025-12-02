"""
Complete Matching Pipeline
File: scripts/run_matching.py

Loads extracted features and performs matching between lost and found items.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from matching.ranking import MatchRanker, FastMatchRetriever, MatchingPipeline
from matching.similarity import SimilarityComputer, FastSimilarityComputer
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


class LostAndFoundMatcher:
    """Main class for matching lost and found items"""

    def __init__(self, features_path, output_path=None):
        """
        Initialize matcher
        
        Args:
            features_path: Path to extracted features directory
            output_path: Path to save matching results
        """
        self.features_path = Path(features_path)
        self.output_path = Path(
            output_path) if output_path else self.features_path / 'matches'
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Load data
        self.all_features = None
        self.feature_matrices = None

    def load_features(self):
        """Load extracted features"""
        print("Loading extracted features...")

        # Load feature matrices
        matrices_file = self.features_path / 'feature_matrices.pkl'
        with open(matrices_file, 'rb') as f:
            self.feature_matrices = pickle.load(f)

        print(f"✓ Loaded feature matrices")
        print(f"  Lost items: {len(self.feature_matrices['lost']['ids'])}")
        print(f"  Found items: {len(self.feature_matrices['found']['ids'])}")

        # Load all features
        features_file = self.features_path / 'all_features.pkl'
        with open(features_file, 'rb') as f:
            self.all_features = pickle.load(f)

        print(f"✓ Loaded {len(self.all_features)} total items")

    def run_matching(self, method='fast',
                     image_weight=0.6, text_weight=0.4,
                     top_k=5, min_similarity=0.3,
                     require_category_match=True,
                     image_method='cosine',
                     text_method='cosine'):
        """
        Run matching algorithm
        
        Args:
            method: 'fast' (matrix-based) or 'slow' (item-by-item)
            image_weight: Weight for image features
            text_weight: Weight for text features
            top_k: Number of matches to return per item
            min_similarity: Minimum similarity threshold
            require_category_match: Filter by category
            image_method: Similarity method for images
            text_method: Similarity method for text
            
        Returns:
            Dictionary with matches
        """
        print(f"\n{'='*80}")
        print("RUNNING MATCHING ALGORITHM")
        print(f"{'='*80}")

        print(f"\nConfiguration:")
        print(f"  Method: {method}")
        print(f"  Image weight: {image_weight}")
        print(f"  Text weight: {text_weight}")
        print(f"  Top-K: {top_k}")
        print(f"  Min similarity: {min_similarity}")
        print(f"  Category match required: {require_category_match}")
        print(f"  Image similarity method: {image_method}")
        print(f"  Text similarity method: {text_method}")

        weights = {'image': image_weight, 'text': text_weight}

        if method == 'fast':
            # Use fast matrix-based matching
            similarity_computer = FastSimilarityComputer(weights=weights)
            ranker = FastMatchRetriever(
                top_k=top_k, min_similarity=min_similarity)

            print("\nComputing similarity matrices...")
            similarities = similarity_computer.compute_all_similarities(
                self.feature_matrices['lost'],
                self.feature_matrices['found'],
                image_method=image_method,
                text_method=text_method
            )

            print("✓ Similarity matrices computed")
            print(
                f"  Shape: {similarities['combined_similarity_matrix'].shape}")

            print("\nRetrieving top matches...")
            categories_lost = self.feature_matrices['lost']['categories'] if require_category_match else None
            categories_found = self.feature_matrices['found']['categories'] if require_category_match else None

            matches = ranker.retrieve_matches_from_matrix(
                similarities['combined_similarity_matrix'],
                self.feature_matrices['lost']['ids'],
                self.feature_matrices['found']['ids'],
                categories_lost=categories_lost,
                categories_found=categories_found,
                filter_category=require_category_match
            )

            return {
                'matches': matches,
                'similarity_matrices': similarities,
                'method': 'fast',
                'config': {
                    'weights': weights,
                    'top_k': top_k,
                    'min_similarity': min_similarity,
                    'require_category_match': require_category_match
                }
            }

        else:
            # Use slower item-by-item matching
            similarity_computer = SimilarityComputer(weights=weights)
            ranker = MatchRanker(top_k=top_k, min_similarity=min_similarity)

            lost_items = [
                f for f in self.all_features if f['status'] == 'lost']
            found_items = [
                f for f in self.all_features if f['status'] == 'found']

            print(f"\nComputing matches for {len(lost_items)} lost items...")

            all_matches = {}
            for lost_item in tqdm(lost_items, desc="Matching"):
                matches = []
                for found_item in found_items:
                    match = similarity_computer.compute_match_score(
                        lost_item, found_item,
                        require_category_match=require_category_match,
                        image_method=image_method,
                        text_method=text_method
                    )

                    if match['valid']:
                        matches.append(match)

                # Rank and get top-K
                top_matches = ranker.get_top_k_matches(matches)
                all_matches[lost_item['id']] = top_matches

            return {
                'matches': all_matches,
                'method': 'slow',
                'config': {
                    'weights': weights,
                    'top_k': top_k,
                    'min_similarity': min_similarity,
                    'require_category_match': require_category_match
                }
            }

    def save_results(self, results):
        """
        Save matching results
        
        Args:
            results: Results from run_matching
        """
        print(f"\n{'='*80}")
        print("SAVING RESULTS")
        print(f"{'='*80}")

        # Save raw results
        results_file = self.output_path / 'matching_results.pkl'
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"✓ Saved raw results to: {results_file}")

        # Save matches as JSON (more readable)
        matches_json = {}
        for lost_id, item_matches in results['matches'].items():
            matches_json[lost_id] = [
                {
                    'rank': m.get('rank', i+1),
                    'found_id': m.get('found_id'),
                    'similarity': float(m.get('similarity') or m.get('combined_similarity', 0))
                }
                for i, m in enumerate(item_matches)
            ]

        json_file = self.output_path / 'matches.json'
        with open(json_file, 'w') as f:
            json.dump(matches_json, f, indent=2)
        print(f"✓ Saved matches to: {json_file}")

        # Create summary DataFrame
        summary_data = []
        for lost_id, item_matches in results['matches'].items():
            summary_data.append({
                'lost_id': lost_id,
                'num_matches': len(item_matches),
                'best_match': item_matches[0].get('found_id') if item_matches else None,
                'best_similarity': float(item_matches[0].get('similarity') or
                                         item_matches[0].get('combined_similarity', 0)) if item_matches else 0
            })

        summary_df = pd.DataFrame(summary_data)
        csv_file = self.output_path / 'matching_summary.csv'
        summary_df.to_csv(csv_file, index=False)
        print(f"✓ Saved summary to: {csv_file}")

        # Create detailed report
        self.create_report(results)

    def create_report(self, results):
        """Create detailed matching report"""
        report = []
        report.append("="*80)
        report.append("MATCHING RESULTS REPORT")
        report.append("="*80)
        report.append("")

        # Configuration
        config = results['config']
        report.append("Configuration:")
        report.append(f"  Method: {results['method']}")
        report.append(f"  Image weight: {config['weights']['image']}")
        report.append(f"  Text weight: {config['weights']['text']}")
        report.append(f"  Top-K: {config['top_k']}")
        report.append(f"  Min similarity: {config['min_similarity']}")
        report.append(f"  Category match: {config['require_category_match']}")
        report.append("")

        # Overall statistics
        matches = results['matches']
        total_lost = len(matches)
        with_matches = len([m for m in matches.values() if len(m) > 0])
        without_matches = total_lost - with_matches

        report.append("Overall Statistics:")
        report.append(f"  Total lost items: {total_lost}")
        report.append(
            f"  Items with matches: {with_matches} ({with_matches/total_lost*100:.1f}%)")
        report.append(
            f"  Items without matches: {without_matches} ({without_matches/total_lost*100:.1f}%)")
        report.append("")

        # Similarity distribution
        all_similarities = []
        for item_matches in matches.values():
            for match in item_matches:
                sim = match.get('similarity') or match.get(
                    'combined_similarity', 0)
                all_similarities.append(sim)

        if all_similarities:
            report.append("Similarity Distribution:")
            report.append(f"  Mean: {np.mean(all_similarities):.3f}")
            report.append(f"  Median: {np.median(all_similarities):.3f}")
            report.append(f"  Std Dev: {np.std(all_similarities):.3f}")
            report.append(f"  Min: {np.min(all_similarities):.3f}")
            report.append(f"  Max: {np.max(all_similarities):.3f}")
            report.append("")

        # Sample matches
        report.append("Sample Matches (Top 5 Lost Items):")
        report.append("-"*80)

        for i, (lost_id, item_matches) in enumerate(list(matches.items())[:5], 1):
            report.append(f"\n{i}. Lost Item: {lost_id}")
            if item_matches:
                report.append(f"   Found {len(item_matches)} matches:")
                # Top 3 matches
                for j, match in enumerate(item_matches[:3], 1):
                    sim = match.get('similarity') or match.get(
                        'combined_similarity', 0)
                    report.append(
                        f"     {j}. Found {match.get('found_id')}: {sim:.3f}")
            else:
                report.append("   No matches found")

        report.append("")
        report.append("="*80)

        # Save report
        report_text = "\n".join(report)
        print("\n" + report_text)

        report_file = self.output_path / 'matching_report.txt'
        with open(report_file, 'w') as f:
            f.write(report_text)
        print(f"\n✓ Saved report to: {report_file}")

    def get_matches_for_item(self, lost_id, results):
        """Get matches for a specific lost item"""
        matches = results['matches'].get(lost_id, [])

        if not matches:
            print(f"No matches found for {lost_id}")
            return None

        print(f"\nMatches for {lost_id}:")
        print("-"*60)

        for i, match in enumerate(matches, 1):
            sim = match.get('similarity') or match.get(
                'combined_similarity', 0)
            found_id = match.get('found_id')
            print(f"Rank {i}: {found_id} - Similarity: {sim:.4f}")

        return matches

    def run_complete_pipeline(self, **kwargs):
        """Run the complete matching pipeline"""
        print("\n" + "="*80)
        print("LOST AND FOUND MATCHING PIPELINE")
        print("="*80)

        # Load features
        self.load_features()

        # Run matching
        results = self.run_matching(**kwargs)

        # Save results
        self.save_results(results)

        print("\n" + "="*80)
        print("✓ MATCHING PIPELINE COMPLETE!")
        print("="*80)

        return results


# Main execution
if __name__ == "__main__":
    # Configure paths
    FEATURES_PATH = Path('./features')
    OUTPUT_PATH = Path('./features/matches')

    # Initialize matcher
    matcher = LostAndFoundMatcher(FEATURES_PATH, OUTPUT_PATH)

    # Run complete pipeline with default parameters
    results = matcher.run_complete_pipeline(
        method='fast',              # Use fast matrix-based matching
        image_weight=0.6,           # 60% weight to image features
        text_weight=0.4,            # 40% weight to text features
        top_k=5,                    # Return top 5 matches
        min_similarity=0.3,         # Minimum similarity threshold
        require_category_match=True,  # Only match same categories
        image_method='cosine',      # Cosine similarity for images
        text_method='cosine'        # Cosine similarity for text
    )

    print("\n📂 Output files created:")
    print(f"  • matching_results.pkl - Complete results object")
    print(f"  • matches.json - Matches in JSON format")
    print(f"  • matching_summary.csv - Summary statistics")
    print(f"  • matching_report.txt - Detailed report")

    print("\n🎯 Next steps:")
    print("  1. Evaluate matching quality")
    print("  2. Tune parameters (weights, thresholds)")
    print("  3. Build Streamlit UI")
    print("  4. Add evaluation metrics")

    # Example: Get matches for a specific item
    if len(results['matches']) > 0:
        sample_lost_id = list(results['matches'].keys())[0]
        print(f"\n📋 Example - Matches for {sample_lost_id}:")
        matcher.get_matches_for_item(sample_lost_id, results)
