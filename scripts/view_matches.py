"""
View Sample Matches
File: scripts/view_matches.py
"""

import json
import pickle
import os
from pathlib import Path

# Get project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Load matches
matches_file = Path(project_root) / 'features' / 'matches' / 'matches.json'
with open(matches_file, 'r') as f:
    matches = json.load(f)

# Load metadata to get descriptions
metadata_file = Path(project_root) / 'dataset' / 'metadata_labelled.json'
with open(metadata_file, 'r') as f:
    metadata = json.load(f)

# Create ID to metadata mapping
id_to_meta = {item['id']: item for item in metadata}

print("="*80)
print("SAMPLE MATCHES")
print("="*80)

# Show first 5 lost items with their matches
for i, (lost_id, item_matches) in enumerate(list(matches.items())[:5], 1):
    print(f"\n{i}. LOST ITEM: {lost_id}")

    # Get lost item info
    if lost_id in id_to_meta:
        lost_info = id_to_meta[lost_id]
        print(f"   Category: {lost_info['category']}")
        print(
            f"   Description: {lost_info.get('description', 'No description')[:80]}...")

    if not item_matches:
        print("   ⚠ No matches found")
        continue

    print(f"\n   Top {len(item_matches)} Matches:")
    print("   " + "-"*76)

    for match in item_matches:
        found_id = match['found_id']
        similarity = match['similarity']
        rank = match['rank']

        # Get found item info
        if found_id in id_to_meta:
            found_info = id_to_meta[found_id]
            print(
                f"\n   Rank {rank}: {found_id} (Similarity: {similarity:.3f})")
            print(f"   Category: {found_info['category']}")
            print(
                f"   Description: {found_info.get('description', 'No description')[:70]}...")
        else:
            print(
                f"\n   Rank {rank}: {found_id} (Similarity: {similarity:.3f})")

    print("\n" + "="*80)

print("\n✓ To see all matches, check: ../features/matches/matches.json")
