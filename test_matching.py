import sys
from pathlib import Path
import numpy as np

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from matching.similarity import SimilarityComputer


print("="*60)
print("TESTING SIMILARITY COMPUTATION")
print("="*60)

# Test similarity computation
computer = SimilarityComputer()

lost_item = {
    'id': 'L001',
    'category': 'bags',
    'timestamp': '2024-01-15T10:00:00',
    'image_features': np.random.rand(184),
    'text_features': np.random.rand(126)
}

found_item = {
    'id': 'F001',
    'category': 'bags',
    'timestamp': '2024-01-16T14:00:00',
    'image_features': np.random.rand(184),
    'text_features': np.random.rand(126)
}

result = computer.compute_match_score(lost_item, found_item)

print(f"\nResults:")
print(f"  Similarity: {result['combined_similarity']:.3f}")
print(f"  Image Similarity: {result['image_similarity']:.3f}")
print(f"  Text Similarity: {result['text_similarity']:.3f}")
print(f"  Valid Match: {result['valid']}")
print(f"  Category Match: {result['category_match']}")
print(f"  Temporal Valid: {result['temporal_valid']}")

print("\n✓ Test completed successfully!")
