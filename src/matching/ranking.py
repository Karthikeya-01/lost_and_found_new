"""
Ranking and Matching Module
File: src/matching/ranking.py

Ranks and retrieves top-K matches for lost items.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional


class MatchRanker:
    """Rank and retrieve top-K matches for items"""

    def __init__(self, top_k=5, min_similarity=0.3):
        """
        Initialize match ranker
        
        Args:
            top_k: Number of top matches to return
            min_similarity: Minimum similarity threshold
        """
        self.top_k = top_k
        self.min_similarity = min_similarity

    def rank_matches(self, matches, sort_by='combined_similarity'):
        """
        Rank matches by similarity score
        
        Args:
            matches: List of match dictionaries
            sort_by: Key to sort by
            
        Returns:
            Sorted list of matches
        """
        # Filter by minimum similarity
        valid_matches = [
            m for m in matches
            if m.get(sort_by, 0) >= self.min_similarity
        ]

        # Sort by similarity (descending)
        ranked = sorted(
            valid_matches,
            key=lambda x: x.get(sort_by, 0),
            reverse=True
        )

        return ranked

    def get_top_k_matches(self, matches, k=None, sort_by='combined_similarity'):
        """
        Get top-K matches
        
        Args:
            matches: List of match dictionaries
            k: Number of matches to return (default: self.top_k)
            sort_by: Key to sort by
            
        Returns:
            Top-K matches
        """
        k = k or self.top_k
        ranked = self.rank_matches(matches, sort_by=sort_by)
        return ranked[:k]

    def rank_matches_for_lost_item(self, lost_item_id, all_matches, k=None):
        """
        Get top-K matches for a specific lost item
        
        Args:
            lost_item_id: ID of lost item
            all_matches: List of all match dictionaries
            k: Number of matches to return
            
        Returns:
            Top-K matches for this lost item
        """
        # Filter matches for this lost item
        item_matches = [
            m for m in all_matches
            if m.get('lost_id') == lost_item_id
        ]

        return self.get_top_k_matches(item_matches, k=k)

    def create_match_summary(self, match):
        """
        Create a readable summary of a match
        
        Args:
            match: Match dictionary
            
        Returns:
            Summary string
        """
        summary = f"Match: Lost {match['lost_id']} ↔ Found {match['found_id']}\n"
        summary += f"  Category: {match['lost_category']}\n"
        summary += f"  Combined Similarity: {match['combined_similarity']:.3f}\n"
        summary += f"  Image Similarity: {match['image_similarity']:.3f}\n"
        summary += f"  Text Similarity: {match['text_similarity']:.3f}\n"

        return summary

    def create_ranking_report(self, lost_item_id, matches):
        """
        Create a detailed report of matches for a lost item
        
        Args:
            lost_item_id: ID of lost item
            matches: List of matches
            
        Returns:
            Report string
        """
        report = f"{'='*80}\n"
        report += f"MATCHES FOR LOST ITEM: {lost_item_id}\n"
        report += f"{'='*80}\n\n"

        if not matches:
            report += "No matches found above the similarity threshold.\n"
            return report

        for i, match in enumerate(matches, 1):
            report += f"Rank {i}:\n"
            report += f"  Found Item: {match['found_id']}\n"
            report += f"  Category: {match['found_category']}\n"
            report += f"  Combined Similarity: {match['combined_similarity']:.4f}\n"
            report += f"  Image Similarity: {match['image_similarity']:.4f}\n"
            report += f"  Text Similarity: {match['text_similarity']:.4f}\n"
            report += f"  Confidence: {self._get_confidence_label(match['combined_similarity'])}\n"
            report += "\n"

        return report

    def _get_confidence_label(self, similarity):
        """Get confidence label based on similarity score"""
        if similarity >= 0.8:
            return "Very High"
        elif similarity >= 0.6:
            return "High"
        elif similarity >= 0.4:
            return "Medium"
        else:
            return "Low"


class FastMatchRetriever:
    """Fast retrieval of matches using pre-computed similarity matrices"""

    def __init__(self, top_k=5, min_similarity=0.3):
        """
        Initialize fast match retriever
        
        Args:
            top_k: Number of top matches to return
            min_similarity: Minimum similarity threshold
        """
        self.top_k = top_k
        self.min_similarity = min_similarity

    def retrieve_matches_from_matrix(self, similarity_matrix,
                                     lost_ids, found_ids,
                                     categories_lost=None,
                                     categories_found=None,
                                     filter_category=True):
        """
        Retrieve matches from pre-computed similarity matrix
        
        Args:
            similarity_matrix: (N_lost, M_found) similarity matrix
            lost_ids: List of lost item IDs
            found_ids: List of found item IDs
            categories_lost: List of categories for lost items
            categories_found: List of categories for found items
            filter_category: Whether to filter by matching categories
            
        Returns:
            Dictionary mapping lost_id -> list of top matches
        """
        all_matches = {}

        for i, lost_id in enumerate(lost_ids):
            # Get similarities for this lost item
            similarities = similarity_matrix[i]

            # Create match list
            matches = []
            for j, found_id in enumerate(found_ids):
                sim = similarities[j]

                # Check category match if needed
                if filter_category and categories_lost and categories_found:
                    if categories_lost[i] != categories_found[j]:
                        continue

                # Check minimum similarity
                if sim < self.min_similarity:
                    continue

                matches.append({
                    'found_id': found_id,
                    'similarity': float(sim),
                    'rank': None  # Will be set after sorting
                })

            # Sort by similarity
            matches.sort(key=lambda x: x['similarity'], reverse=True)

            # Add ranks and take top-K
            for rank, match in enumerate(matches[:self.top_k], 1):
                match['rank'] = rank

            all_matches[lost_id] = matches[:self.top_k]

        return all_matches

    def get_top_k_for_item(self, lost_item_idx, similarity_matrix,
                           found_ids, k=None):
        """
        Get top-K matches for a single lost item
        
        Args:
            lost_item_idx: Index of lost item in matrix
            similarity_matrix: Similarity matrix
            found_ids: List of found item IDs
            k: Number of matches to return
            
        Returns:
            List of top-K matches
        """
        k = k or self.top_k

        # Get similarities
        similarities = similarity_matrix[lost_item_idx]

        # Get top-K indices
        top_k_indices = np.argsort(similarities)[::-1][:k]

        # Filter by minimum similarity
        matches = []
        for rank, idx in enumerate(top_k_indices, 1):
            if similarities[idx] >= self.min_similarity:
                matches.append({
                    'rank': rank,
                    'found_id': found_ids[idx],
                    'found_idx': int(idx),
                    'similarity': float(similarities[idx])
                })

        return matches

    def create_match_dataframe(self, all_matches):
        """
        Convert matches to pandas DataFrame for easy analysis
        
        Args:
            all_matches: Dictionary of matches from retrieve_matches_from_matrix
            
        Returns:
            DataFrame with all matches
        """
        rows = []

        for lost_id, matches in all_matches.items():
            for match in matches:
                rows.append({
                    'lost_id': lost_id,
                    'found_id': match['found_id'],
                    'rank': match['rank'],
                    'similarity': match['similarity']
                })

        return pd.DataFrame(rows)


class MatchingPipeline:
    """Complete matching pipeline combining similarity computation and ranking"""

    def __init__(self, similarity_computer, ranker, use_fast=True):
        """
        Initialize matching pipeline
        
        Args:
            similarity_computer: SimilarityComputer or FastSimilarityComputer
            ranker: MatchRanker or FastMatchRetriever
            use_fast: Whether to use fast matrix operations
        """
        self.similarity_computer = similarity_computer
        self.ranker = ranker
        self.use_fast = use_fast

    def match_all_items(self, lost_features, found_features,
                        require_category_match=True):
        """
        Match all lost items to found items
        
        Args:
            lost_features: List of lost item features or feature matrix
            found_features: List of found item features or feature matrix
            require_category_match: Filter by category
            
        Returns:
            Dictionary with matches for each lost item
        """
        if self.use_fast and isinstance(lost_features, dict):
            # Fast matrix-based matching
            return self._match_fast(lost_features, found_features, require_category_match)
        else:
            # Slower item-by-item matching
            return self._match_slow(lost_features, found_features, require_category_match)

    def _match_fast(self, lost_matrix, found_matrix, require_category_match):
        """Fast matching using matrices"""
        from .similarity import FastSimilarityComputer

        # Compute similarity matrices
        similarities = self.similarity_computer.compute_all_similarities(
            lost_matrix, found_matrix
        )

        # Retrieve matches
        categories_lost = lost_matrix.get(
            'categories') if require_category_match else None
        categories_found = found_matrix.get(
            'categories') if require_category_match else None

        matches = self.ranker.retrieve_matches_from_matrix(
            similarities['combined_similarity_matrix'],
            lost_matrix['ids'],
            found_matrix['ids'],
            categories_lost=categories_lost,
            categories_found=categories_found,
            filter_category=require_category_match
        )

        return matches

    def _match_slow(self, lost_features, found_features, require_category_match):
        """Slower item-by-item matching"""
        # Compute all similarities
        all_matches = self.similarity_computer.batch_compute_similarities(
            lost_features, found_features,
            require_category_match=require_category_match
        )

        # Organize by lost item
        matches_by_lost = {}
        for lost_item in lost_features:
            lost_id = lost_item['id']
            item_matches = self.ranker.rank_matches_for_lost_item(
                lost_id, all_matches)
            matches_by_lost[lost_id] = item_matches

        return matches_by_lost

    def match_single_item(self, lost_item, found_features,
                          require_category_match=True):
        """
        Match a single lost item to all found items
        
        Args:
            lost_item: Single lost item features
            found_features: List of all found item features
            require_category_match: Filter by category
            
        Returns:
            Top-K matches for this item
        """
        matches = []

        for found_item in found_features:
            match = self.similarity_computer.compute_match_score(
                lost_item, found_item,
                require_category_match=require_category_match
            )

            if match['valid']:
                matches.append(match)

        return self.ranker.get_top_k_matches(matches)


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("RANKING AND MATCHING TEST")
    print("="*80)

    # Create sample matches
    sample_matches = [
        {'lost_id': 'L001', 'found_id': 'F001', 'lost_category': 'bags',
         'found_category': 'bags', 'combined_similarity': 0.85,
         'image_similarity': 0.82, 'text_similarity': 0.89},
        {'lost_id': 'L001', 'found_id': 'F002', 'lost_category': 'bags',
         'found_category': 'bags', 'combined_similarity': 0.72,
         'image_similarity': 0.70, 'text_similarity': 0.75},
        {'lost_id': 'L001', 'found_id': 'F003', 'lost_category': 'bags',
         'found_category': 'bags', 'combined_similarity': 0.45,
         'image_similarity': 0.50, 'text_similarity': 0.38},
    ]

    # Initialize ranker
    ranker = MatchRanker(top_k=3, min_similarity=0.3)

    # Rank matches
    print("\nRanking matches for lost item L001...")
    top_matches = ranker.get_top_k_matches(sample_matches)

    print(f"\nFound {len(top_matches)} matches:\n")
    for i, match in enumerate(top_matches, 1):
        print(
            f"Rank {i}: Found {match['found_id']} - Similarity: {match['combined_similarity']:.3f}")

    # Create detailed report
    print("\n" + "="*80)
    report = ranker.create_ranking_report('L001', top_matches)
    print(report)

    # Test fast retrieval
    print("="*80)
    print("Testing fast matrix-based retrieval...")
    print("="*80)

    fast_retriever = FastMatchRetriever(top_k=5, min_similarity=0.3)

    # Create sample similarity matrix
    similarity_matrix = np.array([
        [0.85, 0.72, 0.45, 0.30, 0.65],  # Lost item 1 vs all found
        [0.62, 0.88, 0.55, 0.40, 0.70],  # Lost item 2 vs all found
    ])

    lost_ids = ['L001', 'L002']
    found_ids = ['F001', 'F002', 'F003', 'F004', 'F005']

    matches = fast_retriever.retrieve_matches_from_matrix(
        similarity_matrix, lost_ids, found_ids
    )

    print("\nFast retrieval results:")
    for lost_id, item_matches in matches.items():
        print(f"\n{lost_id}:")
        for match in item_matches:
            print(
                f"  Rank {match['rank']}: {match['found_id']} - {match['similarity']:.3f}")

    print("\n✓ Ranking and matching test complete!")
