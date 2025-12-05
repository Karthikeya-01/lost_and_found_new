"""
Ground Truth Generator for Campus Lost & Found
File: scripts/generate_ground_truth.py

Run this script to automatically generate ground truth matches from your metadata.
"""

import json
from pathlib import Path
import re
from difflib import SequenceMatcher
from collections import defaultdict
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def clean_text(text):
    """Clean and normalize text for comparison"""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    return text


def extract_keywords(description):
    """Extract important keywords from description"""
    desc = clean_text(description)
    
    # Color keywords
    colors = ['red', 'blue', 'black', 'white', 'green', 'yellow', 'orange', 
              'purple', 'pink', 'brown', 'grey', 'gray', 'navy', 'dark', 
              'light', 'tan', 'beige', 'maroon', 'olive', 'camouflage', 
              'camo', 'multicolor']
    
    # Brand keywords
    brands = ['nike', 'adidas', 'jansport', 'under armour', 'targus', 'head',
              'musto', 'jack wolfskin', 'hello kitty', 'disney', 'scooby',
              'strawberry shortcake', 'hedgren', 'majestic', 'texas instruments',
              'casio', 'apple', 'samsung', 'sony', 'hp', 'dell']
    
    # Material keywords
    materials = ['leather', 'fabric', 'canvas', 'nylon', 'denim', 'plastic',
                'metal', 'wood', 'glass', 'rubber', 'cotton', 'polyester']
    
    # Feature keywords
    features = ['backpack', 'bag', 'satchel', 'messenger', 'hiking', 'laptop',
                'zippered', 'compartment', 'pocket', 'strap', 'buckle', 'flap',
                'wallet', 'book', 'calculator', 'mouse', 'earphones', 'headphones',
                'glasses', 'sunglasses', 'keys', 'keychain', 'water bottle',
                'bottle', 'id card', 'card']
    
    # Size keywords
    sizes = ['small', 'large', 'medium', 'big', 'compact', 'mini']
    
    found_colors = [c for c in colors if c in desc]
    found_brands = [b for b in brands if b in desc]
    found_materials = [m for m in materials if m in desc]
    found_features = [f for f in features if f in desc]
    found_sizes = [s for s in sizes if s in desc]
    
    return {
        'colors': found_colors,
        'brands': found_brands,
        'materials': found_materials,
        'features': found_features,
        'sizes': found_sizes,
        'text': desc
    }


def compute_similarity(lost_item, found_item):
    """Compute similarity between lost and found items"""
    
    # Must be same category
    if lost_item['category'] != found_item['category']:
        return 0.0
    
    # Skip if no descriptions
    if not lost_item.get('description') or not found_item.get('description'):
        return 0.0
    
    # Extract features
    lost_kw = extract_keywords(lost_item['description'])
    found_kw = extract_keywords(found_item['description'])
    
    # Compute text similarity
    text_sim = SequenceMatcher(None, lost_kw['text'], found_kw['text']).ratio()
    
    # Color match (very important)
    color_score = 0.0
    if lost_kw['colors'] and found_kw['colors']:
        common_colors = set(lost_kw['colors']) & set(found_kw['colors'])
        if common_colors:
            color_score = len(common_colors) / max(len(lost_kw['colors']), len(found_kw['colors']))
        else:
            # Penalty for different colors
            color_score = -0.2
    
    # Brand match (extremely strong indicator)
    brand_score = 0.0
    if lost_kw['brands'] and found_kw['brands']:
        common_brands = set(lost_kw['brands']) & set(found_kw['brands'])
        if common_brands:
            brand_score = 1.0
        else:
            # Penalty for different brands
            brand_score = -0.3
    
    # Material match
    material_score = 0.0
    if lost_kw['materials'] and found_kw['materials']:
        common_materials = set(lost_kw['materials']) & set(found_kw['materials'])
        if common_materials:
            material_score = len(common_materials) / max(len(lost_kw['materials']), len(found_kw['materials']))
    
    # Feature overlap
    feature_score = 0.0
    if lost_kw['features'] and found_kw['features']:
        common_features = set(lost_kw['features']) & set(found_kw['features'])
        feature_score = len(common_features) / max(len(lost_kw['features']), len(found_kw['features']))
    
    # Size match
    size_score = 0.0
    if lost_kw['sizes'] and found_kw['sizes']:
        common_sizes = set(lost_kw['sizes']) & set(found_kw['sizes'])
        if common_sizes:
            size_score = 0.5
    
    # Combined score with weights
    final_score = (
        0.25 * text_sim +
        0.25 * max(0, color_score) +
        0.25 * max(0, brand_score) +
        0.10 * material_score +
        0.10 * feature_score +
        0.05 * size_score
    )
    
    # Boost for very similar text
    if text_sim > 0.7:
        final_score += 0.1
    
    # Ensure score is between 0 and 1
    final_score = max(0.0, min(1.0, final_score))
    
    return final_score


def generate_ground_truth(metadata, min_confidence=0.75, max_per_lost=1):
    """Generate ground truth matches"""
    
    # Separate lost and found items
    lost_items = [item for item in metadata if item['status'] == 'lost' and item.get('labelled')]
    found_items = [item for item in metadata if item['status'] == 'found' and item.get('labelled')]
    
    print(f"Processing {len(lost_items)} lost items and {len(found_items)} found items...")
    print(f"Minimum confidence threshold: {min_confidence}")
    print()
    
    ground_truth = {}
    match_details = []
    
    # Progress tracking
    processed = 0
    total = len(lost_items)
    
    for lost_item in lost_items:
        processed += 1
        if processed % 50 == 0:
            print(f"  Progress: {processed}/{total} ({processed/total*100:.1f}%)")
        
        best_match = None
        best_score = 0.0
        
        for found_item in found_items:
            score = compute_similarity(lost_item, found_item)
            
            if score > best_score:
                best_score = score
                best_match = found_item
        
        # Only add if above minimum confidence
        if best_match and best_score >= min_confidence:
            ground_truth[lost_item['id']] = best_match['id']
            
            match_details.append({
                'lost_id': lost_item['id'],
                'found_id': best_match['id'],
                'confidence': round(best_score, 3),
                'category': lost_item['category'],
                'lost_desc': lost_item['description'][:100] + '...' if len(lost_item['description']) > 100 else lost_item['description'],
                'found_desc': best_match['description'][:100] + '...' if len(best_match['description']) > 100 else best_match['description'],
                'quality': 'high' if best_score > 0.6 else 'medium' if best_score > 0.45 else 'reasonable'
            })
    
    return ground_truth, match_details


def main():
    """Main execution"""
    print("="*80)
    print("GROUND TRUTH GENERATOR")
    print("="*80)
    print()
    
    # Load metadata
    metadata_path = project_root / 'dataset' / 'metadata_labelled.json'
    
    if not metadata_path.exists():
        print(f"❌ Error: Metadata file not found at {metadata_path}")
        print("   Please make sure the file exists.")
        return
    
    print(f"Loading metadata from: {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"✓ Loaded {len(metadata)} items")
    print(f"  Lost items: {sum(1 for item in metadata if item['status']=='lost')}")
    print(f"  Found items: {sum(1 for item in metadata if item['status']=='found')}")
    print()
    
    # Generate ground truth with hybrid approach
    print("Generating ground truth matches...")
    print("Using hybrid moderate-comprehensive approach")
    print()
    
    ground_truth, details = generate_ground_truth(metadata, min_confidence=0.75)
    
    # Print statistics
    print()
    print(f"{'='*80}")
    print("GROUND TRUTH STATISTICS")
    print(f"{'='*80}")
    print(f"Total matches created: {len(ground_truth)}")
    
    lost_with_desc = sum(1 for item in metadata if item['status']=='lost' and item.get('labelled'))
    if lost_with_desc > 0:
        print(f"Coverage: {len(ground_truth)}/{lost_with_desc} lost items ({len(ground_truth)/lost_with_desc*100:.1f}%)")
    else:
        print(f"Coverage: {len(ground_truth)}/0 lost items (no lost items found)")
    
    # Quality breakdown
    quality_counts = defaultdict(int)
    for detail in details:
        quality_counts[detail['quality']] += 1
    
    print(f"\nQuality Breakdown:")
    print(f"  High confidence (>0.6): {quality_counts['high']} matches")
    print(f"  Medium confidence (0.45-0.6): {quality_counts['medium']} matches")
    print(f"  Reasonable confidence (0.35-0.45): {quality_counts['reasonable']} matches")
    
    # Category breakdown
    category_counts = defaultdict(int)
    for detail in details:
        category_counts[detail['category']] += 1
    
    print(f"\nMatches by Category:")
    for category, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {category}: {count}")
    
    # Show sample matches
    print(f"\n{'='*80}")
    print("SAMPLE MATCHES (Top 10 by confidence)")
    print(f"{'='*80}")
    
    sorted_details = sorted(details, key=lambda x: -x['confidence'])
    for i, match in enumerate(sorted_details[:10], 1):
        print(f"\n{i}. Confidence: {match['confidence']:.3f} ({match['quality'].upper()})")
        print(f"   Category: {match['category']}")
        print(f"   Lost ID: {match['lost_id']}")
        print(f"   Found ID: {match['found_id']}")
        print(f"   Lost: {match['lost_desc']}")
        print(f"   Found: {match['found_desc']}")
    
    # Save ground truth
    output_file = project_root / 'dataset' / 'ground_truth_75.json'
    with open(output_file, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✓ Ground truth saved to: {output_file}")
    print(f"{'='*80}")
    
    # Save detailed matches for review
    details_file = project_root / 'dataset' / 'ground_truth_details_75.json'
    with open(details_file, 'w') as f:
        json.dump(details, f, indent=2)
    
    print(f"✓ Match details saved to: {details_file}")
    print()
    print("You can now use this ground truth with:")
    print("  cd scripts")
    print("  python evaluate_all_models.py")
    print()
    print("To review the matches:")
    print("  Check ground_truth_details.json for confidence scores and match quality")


if __name__ == "__main__":
    main()
