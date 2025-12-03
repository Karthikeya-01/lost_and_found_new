"""
Feature Extraction Pipeline
File: notebooks/02_feature_extraction.ipynb (or scripts/extract_features.py)

This script processes the entire dataset and extracts features from all items.
Features are saved for later use in matching algorithms.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add project root to path - use absolute path for reliability
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from src.feature_extraction.text_features import TextFeatureExtractor
from src.feature_extraction.image_features import ImageFeatureExtractor
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


class FeatureExtractionPipeline:
    """Pipeline to extract and save features from all items"""

    def __init__(self, dataset_path, output_path):
        """
        Initialize pipeline
        
        Args:
            dataset_path: Path to dataset directory
            output_path: Path to save extracted features
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Initialize extractors
        self.image_extractor = ImageFeatureExtractor(target_size=(224, 224))
        self.text_extractor = TextFeatureExtractor(max_features=100)

        # Load metadata
        self.metadata = None
        self.df = None

    def load_metadata(self):
        """Load metadata from JSON file"""
        metadata_path = self.dataset_path / 'metadata_labelled.json'

        print(f"Loading metadata from: {metadata_path}")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.df = pd.DataFrame(self.metadata)
        print(f"✓ Loaded {len(self.df)} items")

        return self.df

    def fit_text_extractor(self):
        """Fit TF-IDF vectorizer on all descriptions"""
        print("\nFitting TF-IDF vectorizer...")

        # Get all descriptions (only labelled ones)
        descriptions = self.df[self.df['labelled']
                               == True]['description'].tolist()

        if len(descriptions) == 0:
            print("⚠ No descriptions found to fit TF-IDF")
            return

        self.text_extractor.fit_tfidf(descriptions)

        # Save fitted vectorizer
        vectorizer_path = self.output_path / 'tfidf_vectorizer.pkl'
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.text_extractor.tfidf_vectorizer, f)

        print(f"✓ Saved TF-IDF vectorizer to: {vectorizer_path}")

    def extract_features_for_item(self, item_data, include_sift=False):
        """
        Extract features for a single item
        
        Args:
            item_data: Dictionary with item metadata
            include_sift: Whether to include SIFT features
            
        Returns:
            Dictionary with extracted features
        """
        item_id = item_data['id']
        category = item_data['category']
        filename = item_data['filename']
        description = item_data.get('description', '')

        # Image path
        image_path = self.dataset_path / 'images' / category / filename

        features = {
            'id': item_id,
            'category': category,
            'filename': filename,
            'status': item_data['status'],
            'timestamp': item_data['timestamp'],
            'has_description': item_data['labelled']
        }

        # Extract image features
        try:
            img_result = self.image_extractor.extract_all_features(
                image_path,
                include_sift=include_sift
            )
            features['image_features'] = img_result['feature_vector']
            features['image_feature_dims'] = img_result['feature_dims']
        except Exception as e:
            # Skip silently - file might be missing
            features['image_features'] = None

        # Extract text features
        try:
            if description and item_data['labelled']:
                text_result = self.text_extractor.extract_all_features(
                    description,
                    category=category
                )
                features['text_features'] = text_result['feature_vector']
                features['text_feature_dims'] = text_result['feature_dims']
                features['extracted_keywords'] = text_result['extracted_keywords']
            else:
                # No description - use zeros
                features['text_features'] = np.zeros(
                    self.text_extractor.get_feature_vector_size())
                features['text_feature_dims'] = {}
                features['extracted_keywords'] = {}
        except Exception as e:
            print(f"⚠ Error extracting text features for {item_id}: {e}")
            features['text_features'] = None

        return features

    def extract_all_features(self, include_sift=False, save_individual=False):
        """
        Extract features for all items in dataset
        
        Args:
            include_sift: Whether to include SIFT features (slower)
            save_individual: Save individual feature files per item
            
        Returns:
            List of feature dictionaries
        """
        print(f"\n{'='*80}")
        print("EXTRACTING FEATURES FROM ALL ITEMS")
        print(f"{'='*80}")

        all_features = []
        errors = []
        skipped_missing_files = 0

        # Process each item
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Extracting features"):
            try:
                features = self.extract_features_for_item(
                    row.to_dict(),
                    include_sift=include_sift
                )
                all_features.append(features)

                # Track missing image files
                if features['image_features'] is None:
                    skipped_missing_files += 1

                # Save individual file if requested
                if save_individual:
                    item_file = self.output_path / f"item_{features['id']}.pkl"
                    with open(item_file, 'wb') as f:
                        pickle.dump(features, f)

            except Exception as e:
                errors.append({'id': row['id'], 'error': str(e)})
                # Skip silently - don't print errors for missing files

        print(
            f"\n✓ Successfully processed {len(all_features)} items")

        if skipped_missing_files > 0:
            print(f"  ℹ Skipped {skipped_missing_files} items with missing image files")

        if errors:
            print(f"⚠ Encountered {len(errors)} errors")
            error_file = self.output_path / 'extraction_errors.json'
            with open(error_file, 'w') as f:
                json.dump(errors, f, indent=2)
            print(f"  Error log saved to: {error_file}")

        return all_features

    def create_feature_matrices(self, all_features):
        """
        Create numpy matrices for efficient computation
        
        Args:
            all_features: List of feature dictionaries
            
        Returns:
            Dictionary with feature matrices and metadata
        """
        print("\nCreating feature matrices...")

        # Separate by status
        lost_features = [f for f in all_features if f['status'] == 'lost']
        found_features = [f for f in all_features if f['status'] == 'found']

        print(f"  Lost items: {len(lost_features)}")
        print(f"  Found items: {len(found_features)}")

        # Create matrices
        matrices = {
            'lost': {
                'ids': [f['id'] for f in lost_features],
                'categories': [f['category'] for f in lost_features],
                'image_features': np.array([f['image_features'] for f in lost_features
                                           if f['image_features'] is not None]),
                'text_features': np.array([f['text_features'] for f in lost_features
                                          if f['text_features'] is not None]),
                'timestamps': [f['timestamp'] for f in lost_features],
                'has_description': [f['has_description'] for f in lost_features]
            },
            'found': {
                'ids': [f['id'] for f in found_features],
                'categories': [f['category'] for f in found_features],
                'image_features': np.array([f['image_features'] for f in found_features
                                           if f['image_features'] is not None]),
                'text_features': np.array([f['text_features'] for f in found_features
                                          if f['text_features'] is not None]),
                'timestamps': [f['timestamp'] for f in found_features],
                'has_description': [f['has_description'] for f in found_features]
            }
        }

        # Add combined features (concatenate image + text)
        for status in ['lost', 'found']:
            img_features = matrices[status]['image_features']
            txt_features = matrices[status]['text_features']

            if len(img_features) > 0 and len(txt_features) > 0:
                matrices[status]['combined_features'] = np.concatenate(
                    [img_features, txt_features], axis=1
                )

        return matrices

    def save_features(self, all_features, matrices):
        """
        Save extracted features to disk
        
        Args:
            all_features: List of all feature dictionaries
            matrices: Feature matrices
        """
        print(f"\n{'='*80}")
        print("SAVING FEATURES")
        print(f"{'='*80}")

        # Save complete feature list
        features_file = self.output_path / 'all_features.pkl'
        with open(features_file, 'wb') as f:
            pickle.dump(all_features, f)
        print(f"✓ Saved all features to: {features_file}")

        # Save feature matrices
        matrices_file = self.output_path / 'feature_matrices.pkl'
        with open(matrices_file, 'wb') as f:
            pickle.dump(matrices, f)
        print(f"✓ Saved feature matrices to: {matrices_file}")

        # Save metadata DataFrame with feature info
        feature_df = pd.DataFrame([{
            'id': f['id'],
            'category': f['category'],
            'status': f['status'],
            'has_description': f['has_description'],
            'has_image_features': f['image_features'] is not None,
            'has_text_features': f['text_features'] is not None,
            'image_feature_dim': len(f['image_features']) if f['image_features'] is not None else 0,
            'text_feature_dim': len(f['text_features']) if f['text_features'] is not None else 0,
        } for f in all_features])

        df_file = self.output_path / 'feature_summary.csv'
        feature_df.to_csv(df_file, index=False)
        print(f"✓ Saved feature summary to: {df_file}")

        # Create statistics report
        self.create_statistics_report(all_features, matrices)

    def create_statistics_report(self, all_features, matrices):
        """Create a statistics report about extracted features"""

        report = []
        report.append("="*80)
        report.append("FEATURE EXTRACTION STATISTICS")
        report.append("="*80)
        report.append("")

        # Overall statistics
        report.append("Overall Statistics:")
        report.append(f"  Total items processed: {len(all_features)}")
        report.append(f"  Lost items: {len(matrices['lost']['ids'])}")
        report.append(f"  Found items: {len(matrices['found']['ids'])}")
        report.append("")

        # Feature dimensions
        sample = all_features[0]
        if sample['image_features'] is not None:
            report.append(
                f"  Image feature dimension: {len(sample['image_features'])}")
        if sample['text_features'] is not None:
            report.append(
                f"  Text feature dimension: {len(sample['text_features'])}")
        report.append("")

        # Category breakdown
        report.append("By Category:")
        for category in sorted(set([f['category'] for f in all_features])):
            cat_items = [f for f in all_features if f['category'] == category]
            cat_lost = len([f for f in cat_items if f['status'] == 'lost'])
            cat_found = len([f for f in cat_items if f['status'] == 'found'])
            report.append(
                f"  {category:.<20} Total: {len(cat_items):>3}  (Lost: {cat_lost:>3}, Found: {cat_found:>3})")
        report.append("")

        # Description availability
        with_desc = len([f for f in all_features if f['has_description']])
        report.append(
            f"Items with descriptions: {with_desc} ({with_desc/len(all_features)*100:.1f}%)")
        report.append("")

        # Save report
        report_text = "\n".join(report)
        print("\n" + report_text)

        report_file = self.output_path / 'extraction_report.txt'
        with open(report_file, 'w') as f:
            f.write(report_text)
        print(f"\n✓ Saved extraction report to: {report_file}")

    def run(self, include_sift=False):
        """
        Run the complete feature extraction pipeline
        
        Args:
            include_sift: Whether to include SIFT features
        """
        print("\n" + "="*80)
        print("FEATURE EXTRACTION PIPELINE")
        print("="*80)

        # Step 1: Load metadata
        self.load_metadata()

        # Step 2: Fit text extractor
        self.fit_text_extractor()

        # Step 3: Extract all features
        all_features = self.extract_all_features(include_sift=include_sift)

        # Step 4: Create matrices
        matrices = self.create_feature_matrices(all_features)

        # Step 5: Save everything
        self.save_features(all_features, matrices)

        print("\n" + "="*80)
        print("✓ FEATURE EXTRACTION PIPELINE COMPLETE!")
        print("="*80)

        return all_features, matrices


# Main execution
if __name__ == "__main__":
    # Configure paths - relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    DATASET_PATH = Path(os.path.join(project_root, 'dataset'))
    OUTPUT_PATH = Path(os.path.join(project_root, 'features'))

    # Run pipeline
    pipeline = FeatureExtractionPipeline(DATASET_PATH, OUTPUT_PATH)

    # Extract features (set include_sift=True if you want SIFT features)
    all_features, matrices = pipeline.run(include_sift=False)

    print("\n📂 Output files created:")
    print(f"  • all_features.pkl - Complete feature data for all items")
    print(f"  • feature_matrices.pkl - Numpy matrices for efficient computation")
    print(f"  • feature_summary.csv - Summary statistics")
    print(f"  • tfidf_vectorizer.pkl - Fitted TF-IDF vectorizer")
    print(f"  • extraction_report.txt - Detailed statistics report")

