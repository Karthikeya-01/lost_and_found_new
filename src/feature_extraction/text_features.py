"""
Text Feature Extraction Module
File: src/feature_extraction/text_features.py

This module extracts features from text descriptions:
- TF-IDF vectors
- Keyword extraction (colors, materials, brands, conditions)
- N-grams
- Text statistics
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter
import string


class TextFeatureExtractor:
    """Extract features from item descriptions"""

    def __init__(self, max_features=100):
        """
        Initialize text feature extractor
        
        Args:
            max_features: Maximum number of TF-IDF features
        """
        self.max_features = max_features
        self.tfidf_vectorizer = None
        self.is_fitted = False

        # Define keyword dictionaries
        self.color_keywords = {
            'red', 'blue', 'green', 'yellow', 'black', 'white', 'gray', 'grey',
            'brown', 'pink', 'purple', 'orange', 'cyan', 'magenta', 'beige',
            'navy', 'maroon', 'gold', 'silver', 'bronze', 'cream', 'tan'
        }

        self.material_keywords = {
            'leather', 'plastic', 'metal', 'wood', 'fabric', 'cotton', 'polyester',
            'nylon', 'rubber', 'silicon', 'glass', 'paper', 'cardboard', 'denim',
            'canvas', 'wool', 'synthetic', 'aluminum', 'steel', 'brass'
        }

        self.condition_keywords = {
            'new', 'old', 'used', 'worn', 'damaged', 'scratched', 'broken', 'torn',
            'pristine', 'excellent', 'good', 'fair', 'poor', 'vintage', 'antique'
        }

        self.size_keywords = {
            'small', 'medium', 'large', 'tiny', 'huge', 'big', 'little', 'mini',
            'compact', 'oversized', 'xl', 'xs', 'xxl'
        }

        self.brand_keywords = {
            'apple', 'samsung', 'nike', 'adidas', 'hp', 'dell', 'lenovo', 'asus',
            'sony', 'lg', 'microsoft', 'logitech', 'razer', 'casio', 'texas',
            'moleskine', 'leuchtturm', 'pilot', 'uni', 'sharpie'
        }

    def clean_text(self, text):
        """
        Clean and normalize text
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove special characters but keep spaces and hyphens
        text = re.sub(r'[^a-z0-9\s\-]', ' ', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def extract_keywords(self, text):
        """
        Extract specific keywords from text
        
        Args:
            text: Cleaned text string
            
        Returns:
            Dictionary with extracted keywords
        """
        words = set(text.lower().split())

        keywords = {
            'colors': list(words & self.color_keywords),
            'materials': list(words & self.material_keywords),
            'conditions': list(words & self.condition_keywords),
            'sizes': list(words & self.size_keywords),
            'brands': list(words & self.brand_keywords)
        }

        return keywords

    def keywords_to_features(self, keywords):
        """
        Convert keywords to binary feature vector
        
        Args:
            keywords: Dictionary of keyword lists
            
        Returns:
            Binary feature vector
        """
        features = []

        # Create features for presence of each keyword type
        for key in ['colors', 'materials', 'conditions', 'sizes', 'brands']:
            # Has any keyword of this type
            features.append(1 if len(keywords[key]) > 0 else 0)
            # Count of keywords
            features.append(len(keywords[key]))

        # One-hot encoding for specific common keywords
        # Colors
        for color in ['red', 'blue', 'black', 'white', 'green']:
            features.append(1 if color in keywords['colors'] else 0)

        # Materials
        for material in ['leather', 'plastic', 'metal']:
            features.append(1 if material in keywords['materials'] else 0)

        return np.array(features)

    def extract_text_statistics(self, text):
        """
        Extract statistical features from text
        
        Args:
            text: Raw text string
            
        Returns:
            Array of text statistics
        """
        if not text:
            return np.zeros(8)

        words = text.split()

        features = [
            len(text),                              # Character count
            len(words),                             # Word count
            # Avg word length
            np.mean([len(w) for w in words]) if words else 0,
            len([w for w in words if len(w) > 6]),  # Long words
            len([c for c in text if c.isdigit()]),  # Digit count
            len([c for c in text if c.isupper()]),  # Uppercase count
            # Exclamation/question marks
            text.count('!') + text.count('?'),
            len(set(words)) / len(words) if words else 0  # Unique word ratio
        ]

        return np.array(features)

    def fit_tfidf(self, descriptions):
        """
        Fit TF-IDF vectorizer on a corpus of descriptions
        
        Args:
            descriptions: List of text descriptions
        """
        # Clean descriptions
        cleaned = [self.clean_text(desc) for desc in descriptions]

        # Initialize and fit TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=2,            # Ignore terms that appear in less than 2 documents
            max_df=0.8,          # Ignore terms that appear in more than 80% of documents
            stop_words='english',
            sublinear_tf=True    # Apply sublinear tf scaling
        )

        self.tfidf_vectorizer.fit(cleaned)
        self.is_fitted = True

        print(
            f"✓ TF-IDF vectorizer fitted with {len(self.tfidf_vectorizer.vocabulary_)} terms")

    def extract_tfidf_features(self, text):
        """
        Extract TF-IDF features from text
        
        Args:
            text: Text description
            
        Returns:
            TF-IDF feature vector
        """
        if not self.is_fitted:
            raise ValueError(
                "TF-IDF vectorizer not fitted. Call fit_tfidf() first.")

        cleaned = self.clean_text(text)
        tfidf_vector = self.tfidf_vectorizer.transform([cleaned]).toarray()[0]

        return tfidf_vector

    def extract_ngrams(self, text, n=2):
        """
        Extract character n-grams (useful for typos and partial matches)
        
        Args:
            text: Text description
            n: N-gram size
            
        Returns:
            N-gram frequency vector
        """
        if not text:
            return np.zeros(50)

        # Generate character n-grams
        text = self.clean_text(text)
        ngrams = [text[i:i+n] for i in range(len(text)-n+1)]

        # Count frequencies
        ngram_counts = Counter(ngrams)

        # Get top 50 n-grams
        top_ngrams = dict(ngram_counts.most_common(50))

        # Create feature vector (normalized frequencies)
        total = sum(top_ngrams.values())
        features = np.array([v/total for v in list(top_ngrams.values())[:50]])

        # Pad if necessary
        if len(features) < 50:
            features = np.pad(features, (0, 50-len(features)), 'constant')

        return features

    def extract_all_features(self, text, category=None):
        """
        Extract all text features
        
        Args:
            text: Text description
            category: Item category (optional, for category-specific features)
            
        Returns:
            Dictionary with all features
        """
        if not text or not isinstance(text, str):
            text = ""

        # Clean text
        cleaned_text = self.clean_text(text)

        # Extract features
        features = {
            'text_stats': self.extract_text_statistics(text),
            'keywords': self.keywords_to_features(self.extract_keywords(cleaned_text)),
        }

        # Add TF-IDF if fitted
        if self.is_fitted:
            features['tfidf'] = self.extract_tfidf_features(text)

        # Flatten all features
        feature_vector = np.concatenate(
            [v.flatten() for v in features.values()])

        return {
            'features': features,
            'feature_vector': feature_vector,
            'feature_names': list(features.keys()),
            'feature_dims': {k: len(v.flatten()) for k, v in features.items()},
            'extracted_keywords': self.extract_keywords(cleaned_text)
        }

    def get_feature_vector_size(self, include_tfidf=True):
        """Get the total size of the feature vector"""
        sizes = {
            'text_stats': 8,
            'keywords': 18,  # 5*2 + 5 + 3
        }

        if include_tfidf:
            sizes['tfidf'] = self.max_features

        return sum(sizes.values())

    def get_top_terms(self, n=20):
        """
        Get top terms from TF-IDF vocabulary
        
        Args:
            n: Number of top terms to return
            
        Returns:
            List of top terms
        """
        if not self.is_fitted:
            return []

        # Get feature names
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        return list(feature_names[:n])

    def compare_texts(self, text1, text2):
        """
        Compare two texts and return similarity metrics
        
        Args:
            text1, text2: Text descriptions
            
        Returns:
            Dictionary with similarity scores
        """
        # Extract keywords
        keywords1 = self.extract_keywords(self.clean_text(text1))
        keywords2 = self.extract_keywords(self.clean_text(text2))

        # Calculate keyword overlap
        overlaps = {}
        for key in keywords1.keys():
            set1 = set(keywords1[key])
            set2 = set(keywords2[key])
            if len(set1) > 0 or len(set2) > 0:
                jaccard = len(set1 & set2) / len(set1 | set2)
            else:
                jaccard = 0
            overlaps[key] = jaccard

        # TF-IDF cosine similarity
        tfidf_sim = 0
        if self.is_fitted:
            vec1 = self.extract_tfidf_features(text1)
            vec2 = self.extract_tfidf_features(text2)

            # Cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 > 0 and norm2 > 0:
                tfidf_sim = dot_product / (norm1 * norm2)

        return {
            'keyword_overlaps': overlaps,
            'tfidf_similarity': tfidf_sim,
            'avg_keyword_overlap': np.mean(list(overlaps.values()))
        }


# Example usage and testing
if __name__ == "__main__":
    print("="*80)
    print("TEXT FEATURE EXTRACTOR TEST")
    print("="*80)

    # Sample descriptions
    sample_descriptions = [
        "Blue leather wallet with university ID card inside",
        "Black Nike backpack, medium size, slightly worn",
        "Red water bottle with a dent on the side",
        "Apple AirPods in white case, found near library",
        "Brown leather-bound textbook on calculus",
        "Silver calculator, Texas Instruments brand",
        "Black plastic glasses with scratched lens",
        "Metal keys on red keychain with car key",
        "Small pink wallet with floral pattern",
        "Large blue water bottle, stainless steel"
    ]

    # Initialize extractor
    extractor = TextFeatureExtractor(max_features=50)

    # Fit TF-IDF
    print("\nFitting TF-IDF on sample descriptions...")
    extractor.fit_tfidf(sample_descriptions)

    print(f"\nFeature Vector Size: {extractor.get_feature_vector_size()}")

    print("\n" + "-"*80)
    print("Top Terms in Vocabulary:")
    print("-"*80)
    top_terms = extractor.get_top_terms(20)
    print(", ".join(top_terms))

    # Test on a sample description
    test_desc = "Blue leather wallet with cards inside"
    print("\n" + "-"*80)
    print(f"Testing on: '{test_desc}'")
    print("-"*80)

    result = extractor.extract_all_features(test_desc)

    print("\nExtracted Features:")
    for name, dim in result['feature_dims'].items():
        print(f"  {name:.<25} {dim} dimensions")

    print(
        f"\n  Total feature vector: {len(result['feature_vector'])} dimensions")

    print("\nExtracted Keywords:")
    for key, values in result['extracted_keywords'].items():
        if values:
            print(f"  {key}: {values}")

    # Compare two similar descriptions
    print("\n" + "-"*80)
    print("Comparing Similar Descriptions:")
    print("-"*80)

    desc1 = "Blue leather wallet with ID card"
    desc2 = "Blue wallet made of leather"

    comparison = extractor.compare_texts(desc1, desc2)

    print(f"\nDescription 1: '{desc1}'")
    print(f"Description 2: '{desc2}'")
    print(f"\nTF-IDF Similarity: {comparison['tfidf_similarity']:.4f}")
    print(f"Average Keyword Overlap: {comparison['avg_keyword_overlap']:.4f}")
    print("\nKeyword Overlaps:")
    for key, value in comparison['keyword_overlaps'].items():
        if value > 0:
            print(f"  {key}: {value:.2f}")

    print("\n✓ Text feature extraction test complete!")
