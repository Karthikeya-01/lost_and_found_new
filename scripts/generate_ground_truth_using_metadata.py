"""
Ground Truth Generator
Creates ground truth matches from metadata using text similarity
"""

import json
from pathlib import Path
import re
from difflib import SequenceMatcher
from collections import defaultdict

# Your metadata (paste the full content here)
metadata = [
    # ... your full metadata ...
]

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
              'light', 'tan', 'beige', 'maroon', 'olive']

    # Brand keywords
    brands = ['nike', 'adidas', 'jansport', 'under armour', 'targus', 'head',
              'musto', 'jack wolfskin', 'hello kitty', 'disney']

    # Material keywords
    materials = ['leather', 'fabric', 'canvas', 'nylon', 'denim']

    # Feature keywords
    features = ['backpack', 'bag', 'satchel', 'messenger', 'hiking', 'laptop',
                'zippered', 'compartment', 'pocket', 'strap']

    found_colors = [c for c in colors if c in desc]
    found_brands = [b for b in brands if b in desc]
    found_materials = [m for m in materials if m in desc]
    found_features = [f for f in features if f in desc]

    return {
        'colors': found_colors,
        'brands': found_brands,
        'materials': found_materials,
        'features': found_features,
        'text': desc
    }

def compute_similarity(lost_item, found_item):
    """Compute similarity between lost and found items"""

    # Must be same category
    if lost_item['category'] != found_item['category']:
        return 0.0

    # Extract features
    lost_kw = extract_keywords(lost_item['description'])
    found_kw = extract_keywords(found_item['description'])

    # Compute text similarity
    text_sim = SequenceMatcher(None, lost_kw['text'], found_kw['text']).ratio()

    # Color match
    color_score = 0.0
    if lost_kw['colors'] and found_kw['colors']:
        common_colors = set(lost_kw['colors']) & set(found_kw['colors'])
        if common_colors:
            color_score = len(common_colors) / \
                              max(len(lost_kw['colors']),
                                  len(found_kw['colors']))

    # Brand match (very strong indicator)
    brand_score = 0.0
    if lost_kw['brands'] and found_kw['brands']:
        common_brands = set(lost_kw['brands']) & set(found_kw['brands'])
        if common_brands:
            brand_score = 1.0

    # Material match
    material_score = 0.0
    if lost_kw['materials'] and found_kw['materials']:
        common_materials = set(lost_kw['materials']) & set(
            found_kw['materials'])
        if common_materials:
            material_score = len(
                common_materials) / max(len(lost_kw['materials']), len(found_kw['materials']))

    # Feature overlap
    feature_score = 0.0
    if lost_kw['features'] and found_kw['features']:
        common_features = set(lost_kw['features']) & set(found_kw['features'])
        feature_score = len(common_features) / \
                            max(len(lost_kw['features']),
                                len(found_kw['features']))

    # Combined score with weights
    final_score = (
        0.30 * text_sim +
        0.25 * color_score +
        0.25 * brand_score +
        0.10 * material_score +
        0.10 * feature_score
    )

    return final_score

def generate_ground_truth(metadata, min_confidence=0.75, max_per_lost=1):
    """Generate ground truth matches"""
    
    # Separate lost and found items
    lost_items = [item for item in metadata if item['status'] == 'lost']
    found_items = [item for item in metadata if item['status'] == 'found']
    
    print(f"Processing {len(lost_items)} lost items and {len(found_items)} found items...")
    
    ground_truth = {}
    match_details = []
    
    for lost_item in lost_items:
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
                'lost_desc': lost_item['description'][:80] + '...',
                'found_desc': best_match['description'][:80] + '...',
                'quality': 'high' if best_score > 0.6 else 'medium' if best_score > 0.45 else 'reasonable'
            })
    
    return ground_truth, match_details

# Generate ground truth
print("Generating ground truth matches...")
print("Using hybrid moderate-comprehensive approach (min confidence: 0.35)")
print()

ground_truth, details = generate_ground_truth(metadata, min_confidence=0.75)

# Print statistics
print(f"\n{'='*80}")
print("GROUND TRUTH STATISTICS")
print(f"{'='*80}")
print(f"Total matches created: {len(ground_truth)}")
total_lost = sum(1 for item in metadata if item['status']=='lost')
if total_lost > 0:
    print(f"Coverage: {len(ground_truth)}/{total_lost} lost items ({len(ground_truth)/total_lost*100:.1f}%)")
else:
    print(f"Coverage: {len(ground_truth)}/0 lost items (no lost items found)")

# Quality breakdown
quality_counts = defaultdict(int)
for detail in details:
    quality_counts[detail['quality']] += 1

print(f"\nQuality Breakdown:")
print(f"  High confidence (>0.6): {quality_counts['high']}")
print(f"  Medium confidence (0.45-0.6): {quality_counts['medium']}")
print(f"  Reasonable confidence (0.35-0.45): {quality_counts['reasonable']}")

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
    print(f"\n{i}. Confidence: {match['confidence']:.3f} ({match['quality']})")
    print(f"   Category: {match['category']}")
    print(f"   Lost: {match['lost_desc']}")
    print(f"   Found: {match['found_desc']}")

# Save ground truth
output_file = 'ground_truth_75.json'
with open(output_file, 'w') as f:
    json.dump(ground_truth, f, indent=2)

print(f"\n{'='*80}")
print(f"✓ Ground truth saved to: {output_file}")
print(f"{'='*80}")

# Save detailed matches for review
details_file = 'ground_truth_details_75.json'
with open(details_file, 'w') as f:
    json.dump(details, f, indent=2)

print(f"✓ Match details saved to: {details_file}")

print(f"\nYou can now use this ground truth with:")
print(f"  python scripts/evaluate_all_models.py")
