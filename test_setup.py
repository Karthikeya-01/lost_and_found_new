import sys
from pathlib import Path
import json
import numpy as np
from PIL import Image

print("="*80)
print("CAMPUS LOST & FOUND - SETUP TEST")
print("="*80)

# Test 1: Check directory structure
print("\n[1/6] Checking directory structure...")
required_dirs = [
    'dataset/images',
    'dataset',
    'src/feature_extraction',
    'notebooks'
]

for dir_path in required_dirs:
    path = Path(dir_path)
    if path.exists():
        print(f"  ✓ {dir_path}")
    else:
        print(f"  ✗ {dir_path} - NOT FOUND")
        print(f"    Please create: mkdir -p {dir_path}")

# Test 2: Check metadata file
print("\n[2/6] Checking metadata file...")
metadata_path = Path('dataset/metadata_labelled.json')
if metadata_path.exists():
    print(f"  ✓ metadata_labelled.json found")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"  ✓ Loaded {len(metadata)} items")

    # Check structure
    sample = metadata[0]
    required_keys = ['id', 'filename', 'category',
                     'timestamp', 'status', 'description', 'labelled']
    for key in required_keys:
        if key in sample:
            print(f"  ✓ Key '{key}' present")
        else:
            print(f"  ✗ Key '{key}' missing")
else:
    print(f"  ✗ metadata_labelled.json NOT FOUND")
    sys.exit(1)

# Test 3: Check Python dependencies
print("\n[3/6] Checking Python dependencies...")
required_packages = {
    'cv2': 'opencv-python',
    'sklearn': 'scikit-learn',
    'PIL': 'Pillow',
    'numpy': 'numpy',
    'pandas': 'pandas',
    'skimage': 'scikit-image',
    'tqdm': 'tqdm'
}

missing_packages = []
for module, package in required_packages.items():
    try:
        __import__(module)
        print(f"  ✓ {package}")
    except ImportError:
        print(f"  ✗ {package} - NOT INSTALLED")
        missing_packages.append(package)

if missing_packages:
    print(f"\n  To install missing packages, run:")
    print(f"  pip install {' '.join(missing_packages)}")
    sys.exit(1)

# Test 4: Check image files
print("\n[4/6] Checking image files...")
categories = set([item['category'] for item in metadata])
print(f"  Categories found: {', '.join(sorted(categories))}")

total_images = 0
missing_images = 0

for category in categories:
    cat_path = Path('dataset/images') / category
    if cat_path.exists():
        images = list(cat_path.glob('*.jpg')) + list(cat_path.glob('*.png'))
        total_images += len(images)
        print(f"  ✓ {category}: {len(images)} images")
    else:
        print(f"  ✗ {category}: directory not found")

# Check for missing images from metadata
print("\n  Verifying image files from metadata...")
for item in metadata[:5]:  # Check first 5
    img_path = Path('dataset/images') / item['category'] / item['filename']
    if not img_path.exists():
        print(f"  ✗ Missing: {img_path}")
        missing_images += 1

if missing_images == 0:
    print(f"  ✓ All images found")
else:
    print(f"  ⚠ {missing_images} images missing")

# Test 5: Test image feature extraction
print("\n[5/6] Testing image feature extraction...")
try:
    sys.path.append('src')
    from feature_extraction.image_features import ImageFeatureExtractor

    extractor = ImageFeatureExtractor()

    # Find a sample image
    sample_item = metadata[0]
    sample_image = Path('dataset/images') / \
        sample_item['category'] / sample_item['filename']

    if sample_image.exists():
        print(f"  Testing on: {sample_image}")
        result = extractor.extract_all_features(
            sample_image, include_sift=False)

        print(f"  ✓ Extracted {len(result['feature_vector'])} image features")
        print(
            f"    - Color histogram: {result['feature_dims']['color_histogram']} dims")
        print(
            f"    - Dominant colors: {result['feature_dims']['dominant_colors']} dims")
        print(f"    - LBP texture: {result['feature_dims']['lbp']} dims")
        print(f"    - GLCM: {result['feature_dims']['glcm']} dims")
    else:
        print(f"  ✗ Sample image not found: {sample_image}")

except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Test text feature extraction
print("\n[6/6] Testing text feature extraction...")
try:
    from feature_extraction.text_features import TextFeatureExtractor

    text_extractor = TextFeatureExtractor(max_features=50)

    # Get sample descriptions
    descriptions = [item['description']
                    for item in metadata if item['labelled']][:10]

    if descriptions:
        print(f"  Fitting TF-IDF on {len(descriptions)} descriptions...")
        text_extractor.fit_tfidf(descriptions)

        sample_desc = descriptions[0]
        print(f"  Testing on: '{sample_desc[:50]}...'")

        result = text_extractor.extract_all_features(sample_desc)

        print(f"  ✓ Extracted {len(result['feature_vector'])} text features")
        print(f"    - TF-IDF: {result['feature_dims']['tfidf']} dims")
        print(f"    - Keywords: {result['feature_dims']['keywords']} dims")
        print(f"    - Text stats: {result['feature_dims']['text_stats']} dims")

        if result['extracted_keywords']:
            print(f"  ✓ Extracted keywords:")
            for key, values in result['extracted_keywords'].items():
                if values:
                    print(f"    - {key}: {values}")
    else:
        print(f"  ⚠ No descriptions found in metadata")

except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "="*80)
print("SETUP TEST COMPLETE")
print("="*80)

