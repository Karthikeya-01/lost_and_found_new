import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import normalize
from datetime import datetime


class SimilarityComputer:

    def __init__(self, weights=None):
        self.weights = weights or {'image': 0.6, 'text': 0.4}

    def cosine_sim(self, vec1, vec2):
        if vec1 is None or vec2 is None:
            return 0.0

        vec1 = vec1.reshape(1, -1)
        vec2 = vec2.reshape(1, -1)

        return cosine_similarity(vec1, vec2)[0, 0]

    def euclidean_sim(self, vec1, vec2):
        if vec1 is None or vec2 is None:
            return 0.0

        vec1 = vec1.reshape(1, -1)
        vec2 = vec2.reshape(1, -1)

        distance = euclidean_distances(vec1, vec2)[0, 0]

        # Convert to similarity
        similarity = 1 / (1 + distance)

        return similarity

    def normalized_euclidean_sim(self, vec1, vec2):
        if vec1 is None or vec2 is None:
            return 0.0

        # Normalize vectors
        vec1_norm = normalize(vec1.reshape(1, -1))[0]
        vec2_norm = normalize(vec2.reshape(1, -1))[0]

        # Compute distance
        distance = np.linalg.norm(vec1_norm - vec2_norm)

        # Convert to similarity (normalized distance is in [0, 2])
        similarity = 1 - (distance / 2)

        return max(0, similarity)  # Ensure non-negative

    def compute_image_similarity(self, img_feat1, img_feat2, method='cosine'):
        if method == 'cosine':
            return self.cosine_sim(img_feat1, img_feat2)
        elif method == 'euclidean':
            return self.euclidean_sim(img_feat1, img_feat2)
        elif method == 'normalized_euclidean':
            return self.normalized_euclidean_sim(img_feat1, img_feat2)
        else:
            raise ValueError(f"Unknown method: {method}")

    def compute_text_similarity(self, txt_feat1, txt_feat2, method='cosine'):
        if method == 'cosine':
            return self.cosine_sim(txt_feat1, txt_feat2)
        elif method == 'euclidean':
            return self.euclidean_sim(txt_feat1, txt_feat2)
        elif method == 'normalized_euclidean':
            return self.normalized_euclidean_sim(txt_feat1, txt_feat2)
        else:
            raise ValueError(f"Unknown method: {method}")

    def compute_combined_similarity(self, lost_item, found_item, image_method='cosine', text_method='cosine'):
        # Extract features
        img_feat_lost = lost_item.get('image_features')
        img_feat_found = found_item.get('image_features')
        txt_feat_lost = lost_item.get('text_features')
        txt_feat_found = found_item.get('text_features')

        # Compute individual similarities
        img_sim = 0.0
        txt_sim = 0.0

        if img_feat_lost is not None and img_feat_found is not None:
            img_sim = self.compute_image_similarity(
                img_feat_lost, img_feat_found, method=image_method
            )

        if txt_feat_lost is not None and txt_feat_found is not None:
            txt_sim = self.compute_text_similarity(
                txt_feat_lost, txt_feat_found, method=text_method
            )

        # Combined score
        # If only one type of feature available, use that
        if img_sim == 0.0 and txt_sim > 0.0:
            combined = txt_sim
        elif txt_sim == 0.0 and img_sim > 0.0:
            combined = img_sim
        else:
            # Weighted combination
            combined = (self.weights['image'] * img_sim +
                        self.weights['text'] * txt_sim)

        return {
            'image_similarity': float(img_sim),
            'text_similarity': float(txt_sim),
            'combined_similarity': float(combined),
            'weights_used': self.weights
        }

    def compute_category_match(self, category1, category2):
        return 1.0 if category1 == category2 else 0.0

    def compute_temporal_validity(self, lost_timestamp, found_timestamp):
        try:
            lost_date = datetime.fromisoformat(
                lost_timestamp.replace('Z', '+00:00'))
            found_date = datetime.fromisoformat(
                found_timestamp.replace('Z', '+00:00'))

            return 1.0 if found_date >= lost_date else 0.0
        except:
            # If can't parse, assume valid
            return 1.0

    def compute_match_score(self, lost_item, found_item,
                            require_category_match=True,
                            require_temporal_validity=True,
                            image_method='cosine',
                            text_method='cosine'):
        # Check category
        category_match = self.compute_category_match(
            lost_item['category'],
            found_item['category']
        )

        if require_category_match and category_match == 0.0:
            return {
                'valid': False,
                'reason': 'category_mismatch',
                'category_match': False,
                'combined_similarity': 0.0
            }

        # Check temporal validity
        temporal_valid = self.compute_temporal_validity(
            lost_item['timestamp'],
            found_item['timestamp']
        )

        if require_temporal_validity and temporal_valid == 0.0:
            return {
                'valid': False,
                'reason': 'temporal_invalid',
                'temporal_valid': False,
                'combined_similarity': 0.0
            }

        # Compute similarity
        similarities = self.compute_combined_similarity(
            lost_item, found_item,
            image_method=image_method,
            text_method=text_method
        )

        return {
            'valid': True,
            'category_match': category_match == 1.0,
            'temporal_valid': temporal_valid == 1.0,
            'lost_id': lost_item['id'],
            'found_id': found_item['id'],
            'lost_category': lost_item['category'],
            'found_category': found_item['category'],
            **similarities
        }

    def batch_compute_similarities(self, lost_items, found_items,
                                   require_category_match=True,
                                   require_temporal_validity=True,
                                   image_method='cosine',
                                   text_method='cosine'):
        all_matches = []

        for lost_item in lost_items:
            for found_item in found_items:
                match_score = self.compute_match_score(
                    lost_item, found_item,
                    require_category_match=require_category_match,
                    require_temporal_validity=require_temporal_validity,
                    image_method=image_method,
                    text_method=text_method
                )

                if match_score['valid']:
                    all_matches.append(match_score)

        return all_matches


class FastSimilarityComputer:

    def __init__(self, weights=None):
        self.weights = weights or {'image': 0.6, 'text': 0.4}

    def compute_similarity_matrix(self, features1, features2, method='cosine'):
        if method == 'cosine':
            # Cosine similarity matrix
            return cosine_similarity(features1, features2)

        elif method == 'euclidean':
            # Euclidean distance -> similarity
            distances = euclidean_distances(features1, features2)
            similarities = 1 / (1 + distances)
            return similarities

        else:
            raise ValueError(f"Unknown method: {method}")

    def compute_all_similarities(self, lost_matrix, found_matrix, image_method='cosine', text_method='cosine'):
        results = {}

        # Image similarities
        if 'image_features' in lost_matrix and 'image_features' in found_matrix:
            img_sim_matrix = self.compute_similarity_matrix(
                lost_matrix['image_features'],
                found_matrix['image_features'],
                method=image_method
            )
            results['image_similarity_matrix'] = img_sim_matrix
        else:
            results['image_similarity_matrix'] = None

        # Text similarities
        if 'text_features' in lost_matrix and 'text_features' in found_matrix:
            txt_sim_matrix = self.compute_similarity_matrix(
                lost_matrix['text_features'],
                found_matrix['text_features'],
                method=text_method
            )
            results['text_similarity_matrix'] = txt_sim_matrix
        else:
            results['text_similarity_matrix'] = None

        # Combined similarities
        if results['image_similarity_matrix'] is not None and results['text_similarity_matrix'] is not None:
            combined_matrix = (
                self.weights['image'] * results['image_similarity_matrix'] +
                self.weights['text'] * results['text_similarity_matrix']
            )
            results['combined_similarity_matrix'] = combined_matrix
        elif results['image_similarity_matrix'] is not None:
            results['combined_similarity_matrix'] = results['image_similarity_matrix']
        elif results['text_similarity_matrix'] is not None:
            results['combined_similarity_matrix'] = results['text_similarity_matrix']
        else:
            results['combined_similarity_matrix'] = None

        return results


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("SIMILARITY COMPUTATION TEST")
    print("="*80)

    # Create sample features
    lost_item = {
        'id': 'L001',
        'category': 'water_bottles',
        'timestamp': '2024-01-15T10:00:00',
        'image_features': np.random.rand(184),
        'text_features': np.random.rand(126)
    }

    found_item = {
        'id': 'F001',
        'category': 'water_bottles',
        'timestamp': '2024-01-16T14:00:00',
        'image_features': np.random.rand(184),
        'text_features': np.random.rand(126)
    }

    # Initialize computer
    computer = SimilarityComputer(weights={'image': 0.6, 'text': 0.4})

    # Compute similarity
    print("\nComputing similarity between items...")
    result = computer.compute_match_score(lost_item, found_item)

    print("\nResults:")
    print(f"  Valid match: {result['valid']}")
    print(f"  Category match: {result['category_match']}")
    print(f"  Temporal valid: {result['temporal_valid']}")
    print(f"  Image similarity: {result['image_similarity']:.4f}")
    print(f"  Text similarity: {result['text_similarity']:.4f}")
    print(f"  Combined similarity: {result['combined_similarity']:.4f}")

    # Test fast computation
    print("\n" + "-"*80)
    print("Testing fast matrix computation...")

    fast_computer = FastSimilarityComputer(weights={'image': 0.6, 'text': 0.4})

    # Create sample matrices
    lost_matrix = {
        'image_features': np.random.rand(5, 184),
        'text_features': np.random.rand(5, 126)
    }

    found_matrix = {
        'image_features': np.random.rand(10, 184),
        'text_features': np.random.rand(10, 126)
    }

    similarities = fast_computer.compute_all_similarities(
        lost_matrix, found_matrix)

    print(
        f"\nImage similarity matrix shape: {similarities['image_similarity_matrix'].shape}")
    print(
        f"Text similarity matrix shape: {similarities['text_similarity_matrix'].shape}")
    print(
        f"Combined similarity matrix shape: {similarities['combined_similarity_matrix'].shape}")

    print("\nSample similarities (first lost item vs all found items):")
    print(similarities['combined_similarity_matrix'][0])

    print("\n✓ Similarity computation test complete!")
