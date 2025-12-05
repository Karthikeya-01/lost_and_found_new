# """ V1 """"
# Campus Lost and Found - Streamlit Application
# File: app/streamlit_app.py

# Interactive web interface for the Lost and Found matching system.
# """

# import sys
# from pathlib import Path

# # Add project root to path BEFORE any other imports
# project_root = Path(__file__).parent.parent
# sys.path.insert(0, str(project_root))

# from src.matching.ranking import MatchRanker
# from src.matching.similarity import SimilarityComputer
# from src.feature_extraction.text_features import TextFeatureExtractor
# from src.feature_extraction.image_features import ImageFeatureExtractor
# import streamlit as st
# import pickle
# import json
# import numpy as np
# import pandas as pd
# from PIL import Image
# from datetime import datetime

# # Page configuration
# st.set_page_config(
#     page_title="Campus Lost & Found",
#     page_icon="🔍",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS
# st.markdown("""
#     <style>
#     .main-header {
#         font-size: 3rem;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .match-card {
#         border: 2px solid #e0e0e0;
#         border-radius: 10px;
#         padding: 1rem;
#         margin: 1rem 0;
#         background-color: #f9f9f9;
#     }
#     .high-confidence {
#         border-left: 5px solid #4CAF50;
#     }
#     .medium-confidence {
#         border-left: 5px solid #FF9800;
#     }
#     .low-confidence {
#         border-left: 5px solid #f44336;
#     }
#     .similarity-score {
#         font-size: 1.5rem;
#         font-weight: bold;
#         color: #1f77b4;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Initialize session state
# if 'matcher_initialized' not in st.session_state:
#     st.session_state.matcher_initialized = False
# if 'current_matches' not in st.session_state:
#     st.session_state.current_matches = None

# # Categories
# CATEGORIES = [
#     'bags', 'books', 'calculators', 'computer_mouse',
#     'earphones', 'glasses', 'id_cards', 'keys',
#     'wallets', 'water_bottles'
# ]


# @st.cache_resource
# def load_system():
#     """Load the matching system components"""
#     try:
#         # Load feature extractors
#         image_extractor = ImageFeatureExtractor()
#         text_extractor = TextFeatureExtractor()

#         # Get project root (parent of app directory)
#         project_root = Path(__file__).parent.parent

#         # Load fitted TF-IDF vectorizer
#         vectorizer_path = project_root / 'features' / 'tfidf_vectorizer.pkl'
#         with open(vectorizer_path, 'rb') as f:
#             text_extractor.tfidf_vectorizer = pickle.load(f)
#             text_extractor.is_fitted = True

#         # Load feature matrices
#         matrices_path = project_root / 'features' / 'feature_matrices.pkl'
#         with open(matrices_path, 'rb') as f:
#             feature_matrices = pickle.load(f)

#         # Load all features
#         features_path = project_root / 'features' / 'all_features.pkl'
#         with open(features_path, 'rb') as f:
#             all_features = pickle.load(f)

#         # Load metadata
#         metadata_path = project_root / 'dataset' / 'metadata_labelled.json'
#         with open(metadata_path, 'r') as f:
#             metadata = json.load(f)

#         # Create ID to metadata mapping
#         id_to_meta = {item['id']: item for item in metadata}

#         # Initialize similarity computer and ranker
#         similarity_computer = SimilarityComputer(
#             weights={'image': 0.6, 'text': 0.4})
#         ranker = MatchRanker(top_k=5, min_similarity=0.3)

#         return {
#             'image_extractor': image_extractor,
#             'text_extractor': text_extractor,
#             'similarity_computer': similarity_computer,
#             'ranker': ranker,
#             'feature_matrices': feature_matrices,
#             'all_features': all_features,
#             'id_to_meta': id_to_meta
#         }
#     except Exception as e:
#         st.error(f"Error loading system: {e}")
#         return None


# def extract_features_from_upload(image, description, category, system):
#     """Extract features from uploaded image and description"""
#     try:
#         # Save temporary image
#         project_root = Path(__file__).parent.parent
#         temp_path = project_root / 'temp_upload.jpg'
#         temp_path.parent.mkdir(exist_ok=True)
#         image.save(temp_path)

#         # Extract image features
#         img_result = system['image_extractor'].extract_all_features(temp_path)
#         img_features = img_result['feature_vector']

#         # Extract text features
#         txt_result = system['text_extractor'].extract_all_features(
#             description, category=category)
#         txt_features = txt_result['feature_vector']

#         # Clean up
#         temp_path.unlink()

#         return {
#             'image_features': img_features,
#             'text_features': txt_features,
#             'category': category,
#             'description': description,
#             'timestamp': datetime.now().isoformat(),
#             'id': f'NEW_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
#         }
#     except Exception as e:
#         st.error(f"Error extracting features: {e}")
#         return None


# def find_matches(new_item, system, status='lost'):
#     """Find matches for a new item"""
#     try:
#         # Determine which items to match against
#         if status == 'lost':
#             # Match against found items
#             target_features = [
#                 f for f in system['all_features'] if f['status'] == 'found']
#         else:
#             # Match against lost items
#             target_features = [
#                 f for f in system['all_features'] if f['status'] == 'lost']

#         # Compute matches
#         matches = []
#         for target_item in target_features:
#             match = system['similarity_computer'].compute_match_score(
#                 new_item if status == 'lost' else target_item,
#                 target_item if status == 'lost' else new_item,
#                 require_category_match=True,
#                 require_temporal_validity=False
#             )

#             if match['valid']:
#                 # Add metadata
#                 target_id = target_item['id']
#                 if target_id in system['id_to_meta']:
#                     meta = system['id_to_meta'][target_id]
#                     match['metadata'] = meta

#                 matches.append(match)

#         # Rank matches
#         top_matches = system['ranker'].get_top_k_matches(matches)

#         return top_matches
#     except Exception as e:
#         st.error(f"Error finding matches: {e}")
#         return []


# def display_match_card(match, rank, system):
#     """Display a match card"""
#     match_id = match.get('found_id') or match.get('lost_id')
#     similarity = match.get('combined_similarity', match.get('similarity', 0.0))

#     # Get confidence level
#     if similarity >= 0.7:
#         confidence = "High"
#         css_class = "high-confidence"
#     elif similarity >= 0.5:
#         confidence = "Medium"
#         css_class = "medium-confidence"
#     else:
#         confidence = "Low"
#         css_class = "low-confidence"

#     # Get metadata
#     meta = match.get('metadata', {})

#     with st.container():
#         st.markdown(
#             f'<div class="match-card {css_class}">', unsafe_allow_html=True)

#         col1, col2, col3 = st.columns([1, 3, 1])

#         with col1:
#             st.markdown(f"### Rank {rank}")
#             st.markdown(
#                 f'<div class="similarity-score">{similarity:.1%}</div>', unsafe_allow_html=True)
#             st.caption(f"{confidence} Confidence")

#         with col2:
#             st.markdown(f"**Item ID:** {match_id}")
#             st.markdown(f"**Category:** {meta.get('category', 'Unknown')}")

#             if meta.get('description'):
#                 st.markdown(f"**Description:** {meta['description']}")

#             if meta.get('timestamp'):
#                 st.markdown(f"**Date:** {meta['timestamp'][:10]}")

#             # Show similarity breakdown
#             with st.expander("📊 Similarity Breakdown"):
#                 if 'image_similarity' in match:
#                     st.write(f"Image Similarity: {match['image_similarity']:.1%}")
#                 if 'text_similarity' in match:
#                     st.write(f"Text Similarity: {match['text_similarity']:.1%}")
#                 if not ('image_similarity' in match or 'text_similarity' in match):
#                     st.write(f"Overall Similarity: {similarity:.1%}")

#         with col3:
#             # Show image if available
#             if meta.get('filename') and meta.get('category'):
#                 # Use absolute path from the project root
#                 img_path = Path(__file__).parent.parent / "dataset" / "images" / meta['category'] / meta['filename']
#                 try:
#                     if img_path.exists():
#                         from PIL import Image
#                         img = Image.open(img_path)
#                         st.image(img, width=150)
#                     else:
#                         st.caption("📷 Image unavailable")
#                 except Exception:
#                     # Silently skip if image cannot be loaded
#                     st.caption("📷 Image unavailable")

#         st.markdown('</div>', unsafe_allow_html=True)


# def main():
#     # Header
#     st.markdown('<h1 class="main-header">🔍 Campus Lost & Found</h1>',
#                 unsafe_allow_html=True)
#     st.markdown("---")

#     # Load system
#     with st.spinner("Loading matching system..."):
#         system = load_system()

#     if system is None:
#         st.error("Failed to load the matching system. Please check the setup.")
#         st.stop()

#     st.success("✓ System loaded successfully!")

#     # Sidebar navigation
#     st.sidebar.title("Navigation")
#     page = st.sidebar.radio(
#         "Choose an action:",
#         ["🏠 Home", "😢 Report Lost Item", "🎉 Report Found Item",
#             "🔎 Browse Existing Matches", "📊 Statistics"]
#     )

#     # Home Page
#     if page == "🏠 Home":
#         st.header("Welcome to Campus Lost & Found!")

#         col1, col2 = st.columns(2)

#         with col1:
#             st.subheader("😢 Lost Something?")
#             st.write("""
#             1. Upload a photo of your lost item
#             2. Provide a description
#             3. Our AI will find potential matches
#             4. Browse through suggested items
#             """)
#             if st.button("Report Lost Item", width='stretch'):
#                 st.session_state.page = "😢 Report Lost Item"
#                 st.rerun()

#         with col2:
#             st.subheader("🎉 Found Something?")
#             st.write("""
#             1. Upload a photo of the found item
#             2. Describe what you found
#             3. Help us match it to owners
#             4. Make someone's day!
#             """)
#             if st.button("Report Found Item", width='stretch'):
#                 st.session_state.page = "🎉 Report Found Item"
#                 st.rerun()

#         st.markdown("---")

#         # Statistics
#         st.subheader("📈 Current Statistics")
#         col1, col2, col3 = st.columns(3)

#         with col1:
#             lost_count = len(
#                 [f for f in system['all_features'] if f['status'] == 'lost'])
#             st.metric("Lost Items", lost_count)

#         with col2:
#             found_count = len(
#                 [f for f in system['all_features'] if f['status'] == 'found'])
#             st.metric("Found Items", found_count)

#         with col3:
#             categories_count = len(CATEGORIES)
#             st.metric("Categories", categories_count)

#     # Report Lost Item
#     elif page == "😢 Report Lost Item":
#         st.header("Report a Lost Item")

#         with st.form("lost_item_form"):
#             col1, col2 = st.columns(2)

#             with col1:
#                 uploaded_image = st.file_uploader(
#                     "Upload an image of your lost item",
#                     type=['jpg', 'jpeg', 'png'],
#                     help="Upload a clear photo of the item you lost"
#                 )

#                 if uploaded_image:
#                     image = Image.open(uploaded_image)
#                     st.image(image, caption="Uploaded Image",
#                              width='stretch')

#             with col2:
#                 category = st.selectbox(
#                     "Category",
#                     CATEGORIES,
#                     help="Select the category that best matches your item"
#                 )

#                 description = st.text_area(
#                     "Description",
#                     placeholder="Describe your item: color, brand, distinguishing features, where you lost it...",
#                     height=150,
#                     help="The more details you provide, the better we can match!"
#                 )

#                 contact_info = st.text_input(
#                     "Contact Information (Optional)",
#                     placeholder="Email or phone number"
#                 )

#             submitted = st.form_submit_button(
#                 "🔍 Find Matches", width='stretch')

#         if submitted:
#             if not uploaded_image:
#                 st.error("Please upload an image of your lost item.")
#             elif not description:
#                 st.error("Please provide a description of your lost item.")
#             else:
#                 with st.spinner("Extracting features and finding matches..."):
#                     # Extract features
#                     image = Image.open(uploaded_image)
#                     new_item = extract_features_from_upload(
#                         image, description, category, system)

#                     if new_item:
#                         # Find matches
#                         matches = find_matches(new_item, system, status='lost')

#                         st.session_state.current_matches = matches

#                         # Display results
#                         st.success(
#                             f"✓ Found {len(matches)} potential matches!")

#                         if matches:
#                             st.markdown("---")
#                             st.subheader("🎯 Potential Matches")
#                             st.write(
#                                 "Here are items that might match what you lost:")

#                             for i, match in enumerate(matches, 1):
#                                 display_match_card(match, i, system)
#                         else:
#                             st.info(
#                                 "No matches found yet. We'll keep looking as new items are reported!")

#     # Report Found Item
#     elif page == "🎉 Report Found Item":
#         st.header("Report a Found Item")

#         with st.form("found_item_form"):
#             col1, col2 = st.columns(2)

#             with col1:
#                 uploaded_image = st.file_uploader(
#                     "Upload an image of the found item",
#                     type=['jpg', 'jpeg', 'png'],
#                     help="Upload a clear photo of the item you found"
#                 )

#                 if uploaded_image:
#                     image = Image.open(uploaded_image)
#                     st.image(image, caption="Uploaded Image", width='stretch')

#             with col2:
#                 category = st.selectbox(
#                     "Category",
#                     CATEGORIES,
#                     help="Select the category that best matches the item"
#                 )

#                 description = st.text_area(
#                     "Description",
#                     placeholder="Describe the item: color, brand, condition, where you found it...",
#                     height=150,
#                     help="Provide details to help match with the owner"
#                 )

#                 location = st.text_input(
#                     "Where did you find it?",
#                     placeholder="Building, room, or location"
#                 )

#             submitted = st.form_submit_button(
#                 "🔍 Find Potential Owners", width='stretch')

#         if submitted:
#             if not uploaded_image:
#                 st.error("Please upload an image of the found item.")
#             elif not description:
#                 st.error("Please provide a description of the found item.")
#             else:
#                 with st.spinner("Extracting features and finding matches..."):
#                     # Extract features
#                     image = Image.open(uploaded_image)
#                     new_item = extract_features_from_upload(
#                         image, description, category, system)

#                     if new_item:
#                         # Find matches
#                         matches = find_matches(
#                             new_item, system, status='found')

#                         st.session_state.current_matches = matches

#                         # Display results
#                         st.success(f"✓ Found {len(matches)} potential owners!")

#                         if matches:
#                             st.markdown("---")
#                             st.subheader("🎯 Potential Owners")
#                             st.write(
#                                 "These people might be looking for this item:")

#                             for i, match in enumerate(matches, 1):
#                                 display_match_card(match, i, system)
#                         else:
#                             st.info(
#                                 "No owners found yet. The item will be available for matching as people report lost items!")

#     # Browse Existing Matches
#     elif page == "🔎 Browse Existing Matches":
#         st.header("Browse Existing Matches")

#         # Load existing matches
#         project_root = Path(__file__).parent.parent
#         matches_file = project_root / 'features' / 'matches' / 'matches.json'
#         if matches_file.exists():
#             with open(matches_file, 'r') as f:
#                 all_matches = json.load(f)

#             # Filter options
#             col1, col2 = st.columns(2)

#             with col1:
#                 filter_category = st.selectbox(
#                     "Filter by category",
#                     ["All"] + CATEGORIES
#                 )

#             with col2:
#                 min_similarity = st.slider(
#                     "Minimum similarity",
#                     0.0, 1.0, 0.3, 0.05
#                 )

#             st.markdown("---")

#             # Display matches
#             displayed = 0
#             for lost_id, item_matches in all_matches.items():
#                 # Get lost item metadata
#                 if lost_id not in system['id_to_meta']:
#                     continue

#                 lost_meta = system['id_to_meta'][lost_id]

#                 # Apply filters
#                 if filter_category != "All" and lost_meta['category'] != filter_category:
#                     continue

#                 # Filter by similarity
#                 filtered_matches = [
#                     m for m in item_matches if m['similarity'] >= min_similarity]

#                 if not filtered_matches:
#                     continue

#                 displayed += 1

#                 # Display lost item info
#                 st.subheader(f"Lost Item: {lost_id}")
#                 col1, col2 = st.columns([1, 3])

#                 with col1:
#                     project_root = Path(__file__).parent.parent
#                     img_path = project_root / "dataset" / "images" / lost_meta['category'] / lost_meta['filename']
#                     try:
#                         if img_path.exists():
#                             from PIL import Image as PILImage
#                             img = PILImage.open(img_path)
#                             st.image(img, width=150)
#                         else:
#                             st.caption("Image not available")
#                     except Exception:
#                         # Skip silently if image cannot be loaded
#                         pass

#                 with col2:
#                     st.write(f"**Category:** {lost_meta['category']}")
#                     if lost_meta.get('description'):
#                         st.write(
#                             f"**Description:** {lost_meta['description']}")
#                     st.write(f"**Matches found:** {len(filtered_matches)}")

#                 # Display matches
#                 with st.expander(f"View {len(filtered_matches)} matches", expanded=False):
#                     for i, match in enumerate(filtered_matches, 1):
#                         found_id = match['found_id']
#                         if found_id in system['id_to_meta']:
#                             match['metadata'] = system['id_to_meta'][found_id]
#                             display_match_card(match, i, system)

#                 st.markdown("---")

#             if displayed == 0:
#                 st.info("No matches found with the current filters.")
#         else:
#             st.warning(
#                 "No existing matches found. Please run the matching pipeline first.")

#     # Statistics
#     elif page == "📊 Statistics":
#         st.header("System Statistics")

#         # Overall stats
#         st.subheader("📈 Overview")
#         col1, col2, col3, col4 = st.columns(4)

#         with col1:
#             total_items = len(system['all_features'])
#             st.metric("Total Items", total_items)

#         with col2:
#             lost_count = len(
#                 [f for f in system['all_features'] if f['status'] == 'lost'])
#             st.metric("Lost Items", lost_count)

#         with col3:
#             found_count = len(
#                 [f for f in system['all_features'] if f['status'] == 'found'])
#             st.metric("Found Items", found_count)

#         with col4:
#             with_desc = len(
#                 [f for f in system['all_features'] if f['has_description']])
#             st.metric("With Descriptions", f"{with_desc/total_items*100:.0f}%")

#         st.markdown("---")

#         # Category breakdown
#         st.subheader("📦 Items by Category")

#         category_data = {}
#         for feature in system['all_features']:
#             cat = feature['category']
#             status = feature['status']

#             if cat not in category_data:
#                 category_data[cat] = {'lost': 0, 'found': 0}

#             category_data[cat][status] += 1

#         # Create DataFrame
#         df = pd.DataFrame(category_data).T
#         df['total'] = df['lost'] + df['found']
#         df = df.sort_values('total', ascending=False)

#         st.bar_chart(df[['lost', 'found']])
#         st.dataframe(df, width='stretch')


# if __name__ == "__main__":
#     main()


# """
# V2
# Campus Lost and Found - Streamlit Application
# File: app/streamlit_app.py

# Interactive web interface for the Lost and Found matching system.
# """

# from src.matching.ranking import MatchRanker
# from src.matching.similarity import SimilarityComputer
# from src.feature_extraction.text_features import TextFeatureExtractor
# from src.feature_extraction.image_features import ImageFeatureExtractor
# import streamlit as st
# import pickle
# import json
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from PIL import Image
# import sys
# import os
# from datetime import datetime

# # Get the project root directory (parent of app directory)
# current_dir = Path(__file__).parent
# project_root = current_dir.parent

# # Add project root to path if not already there
# if str(project_root) not in sys.path:
#     sys.path.insert(0, str(project_root))

# # Now import from src

# # Page configuration
# st.set_page_config(
#     page_title="Campus Lost & Found",
#     page_icon="🔍",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS
# st.markdown("""
#     <style>
#     .main-header {
#         font-size: 3rem;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .match-card {
#         border: 2px solid #e0e0e0;
#         border-radius: 10px;
#         padding: 1rem;
#         margin: 1rem 0;
#         background-color: #f9f9f9;
#     }
#     .high-confidence {
#         border-left: 5px solid #4CAF50;
#     }
#     .medium-confidence {
#         border-left: 5px solid #FF9800;
#     }
#     .low-confidence {
#         border-left: 5px solid #f44336;
#     }
#     .similarity-score {
#         font-size: 1.5rem;
#         font-weight: bold;
#         color: #1f77b4;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Initialize session state
# if 'matcher_initialized' not in st.session_state:
#     st.session_state.matcher_initialized = False
# if 'current_matches' not in st.session_state:
#     st.session_state.current_matches = None

# # Categories
# CATEGORIES = [
#     'bags', 'books', 'calculators', 'computer_mouse',
#     'earphones', 'glasses', 'id_cards', 'keys',
#     'wallets', 'water_bottles'
# ]


# @st.cache_resource
# def load_system():
#     """Load the matching system components"""
#     try:
#         # Get project root
#         current_dir = Path(__file__).parent
#         project_root = current_dir.parent

#         # Load feature extractors
#         image_extractor = ImageFeatureExtractor()
#         text_extractor = TextFeatureExtractor()

#         # Load fitted TF-IDF vectorizer
#         vectorizer_path = project_root / 'features' / 'tfidf_vectorizer.pkl'
#         with open(vectorizer_path, 'rb') as f:
#             text_extractor.tfidf_vectorizer = pickle.load(f)
#             text_extractor.is_fitted = True

#         # Load feature matrices
#         matrices_path = project_root / 'features' / 'feature_matrices.pkl'
#         with open(matrices_path, 'rb') as f:
#             feature_matrices = pickle.load(f)

#         # Load all features
#         features_path = project_root / 'features' / 'all_features.pkl'
#         with open(features_path, 'rb') as f:
#             all_features = pickle.load(f)

#         # Load metadata
#         metadata_path = project_root / 'dataset' / 'metadata_labelled.json'
#         with open(metadata_path, 'r') as f:
#             metadata = json.load(f)

#         # Create ID to metadata mapping
#         id_to_meta = {item['id']: item for item in metadata}

#         # Initialize similarity computer and ranker
#         similarity_computer = SimilarityComputer(
#             weights={'image': 0.6, 'text': 0.4})
#         ranker = MatchRanker(top_k=5, min_similarity=0.3)

#         return {
#             'image_extractor': image_extractor,
#             'text_extractor': text_extractor,
#             'similarity_computer': similarity_computer,
#             'ranker': ranker,
#             'feature_matrices': feature_matrices,
#             'all_features': all_features,
#             'id_to_meta': id_to_meta,
#             'project_root': project_root
#         }
#     except Exception as e:
#         st.error(f"Error loading system: {e}")
#         st.error(f"Current directory: {Path.cwd()}")
#         st.error(f"Project root: {project_root}")
#         return None


# def extract_features_from_upload(image, description, category, system):
#     """Extract features from uploaded image and description"""
#     try:
#         project_root = system['project_root']

#         # Save temporary image
#         temp_path = project_root / 'temp_upload.jpg'
#         image.save(temp_path)

#         # Extract image features
#         img_result = system['image_extractor'].extract_all_features(temp_path)
#         img_features = img_result['feature_vector']

#         # Extract text features
#         txt_result = system['text_extractor'].extract_all_features(
#             description, category=category)
#         txt_features = txt_result['feature_vector']

#         # Clean up
#         temp_path.unlink()

#         return {
#             'image_features': img_features,
#             'text_features': txt_features,
#             'category': category,
#             'description': description,
#             'timestamp': datetime.now().isoformat(),
#             'id': f'NEW_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
#         }
#     except Exception as e:
#         st.error(f"Error extracting features: {e}")
#         return None


# def find_matches(new_item, system, status='lost'):
#     """Find matches for a new item"""
#     try:
#         # Determine which items to match against
#         if status == 'lost':
#             # Match against found items
#             target_features = [
#                 f for f in system['all_features'] if f['status'] == 'found']
#         else:
#             # Match against lost items
#             target_features = [
#                 f for f in system['all_features'] if f['status'] == 'lost']

#         # Compute matches
#         matches = []
#         for target_item in target_features:
#             match = system['similarity_computer'].compute_match_score(
#                 new_item if status == 'lost' else target_item,
#                 target_item if status == 'lost' else new_item,
#                 require_category_match=True,
#                 require_temporal_validity=False
#             )

#             if match['valid']:
#                 # Add metadata
#                 target_id = target_item['id']
#                 if target_id in system['id_to_meta']:
#                     meta = system['id_to_meta'][target_id]
#                     match['metadata'] = meta

#                 matches.append(match)

#         # Rank matches
#         top_matches = system['ranker'].get_top_k_matches(matches)

#         return top_matches
#     except Exception as e:
#         st.error(f"Error finding matches: {e}")
#         return []


# def display_match_card(match, rank, system, show_image=True):
#     """Display a match card"""
#     match_id = match.get('found_id') or match.get('lost_id')
#     similarity = match.get('combined_similarity', match.get('similarity', 0.0))
#     project_root = system['project_root']

#     # Get confidence level
#     if similarity >= 0.7:
#         confidence = "High"
#         css_class = "high-confidence"
#     elif similarity >= 0.5:
#         confidence = "Medium"
#         css_class = "medium-confidence"
#     else:
#         confidence = "Low"
#         css_class = "low-confidence"

#     # Get metadata
#     meta = match.get('metadata', {})

#     with st.container():
#         st.markdown(
#             f'<div class="match-card {css_class}">', unsafe_allow_html=True)

#         col1, col2, col3 = st.columns([1, 3, 1])

#         with col1:
#             st.markdown(f"### Rank {rank}")
#             st.markdown(
#                 f'<div class="similarity-score">{similarity:.1%}</div>', unsafe_allow_html=True)
#             st.caption(f"{confidence} Confidence")

#         with col2:
#             st.markdown(f"**Item ID:** {match_id}")
#             st.markdown(f"**Category:** {meta.get('category', 'Unknown')}")

#             if meta.get('description'):
#                 # Truncate long descriptions
#                 desc = meta['description']
#                 if len(desc) > 150:
#                     st.markdown(f"**Description:** {desc[:150]}...")
#                 else:
#                     st.markdown(f"**Description:** {desc}")

#             if meta.get('timestamp'):
#                 st.markdown(f"**Date:** {meta['timestamp'][:10]}")

#             # Show similarity breakdown
#             with st.expander("📊 Similarity Breakdown"):
#                 if 'image_similarity' in match:
#                     st.write(f"Image Similarity: {match['image_similarity']:.1%}")
#                 if 'text_similarity' in match:
#                     st.write(f"Text Similarity: {match['text_similarity']:.1%}")
#                 if 'image_similarity' not in match and 'text_similarity' not in match:
#                     st.write(f"Overall Similarity: {similarity:.1%}")

#         with col3:
#             # Show image if available and enabled
#             if show_image and meta.get('filename') and meta.get('category'):
#                 img_path = project_root / 'dataset' / 'images' / \
#                     meta['category'] / meta['filename']
#                 if img_path.exists():
#                     try:
#                         # Use smaller width and catch any image loading errors
#                         st.image(str(img_path), width=120)
#                     except Exception as e:
#                         st.caption("🖼️ Image preview unavailable")
#             elif not show_image:
#                 st.caption("🖼️ Click to view image")

#         st.markdown('</div>', unsafe_allow_html=True)


# def main():
#     # Header
#     st.markdown('<h1 class="main-header">🔍 Campus Lost & Found</h1>',
#                 unsafe_allow_html=True)
#     st.markdown("---")

#     # Load system
#     with st.spinner("Loading matching system..."):
#         system = load_system()

#     if system is None:
#         st.error("Failed to load the matching system. Please check the setup.")
#         st.stop()

#     st.success("✓ System loaded successfully!")

#     # Sidebar navigation
#     st.sidebar.title("Navigation")
#     page = st.sidebar.radio(
#         "Choose an action:",
#         ["🏠 Home", "😢 Report Lost Item", "🎉 Report Found Item",
#             "🔎 Browse Existing Matches", "📊 Statistics"]
#     )

#     # Home Page
#     if page == "🏠 Home":
#         st.header("Welcome to Campus Lost & Found!")

#         col1, col2 = st.columns(2)

#         with col1:
#             st.subheader("😢 Lost Something?")
#             st.write("""
#             1. Upload a photo of your lost item
#             2. Provide a description
#             3. Our AI will find potential matches
#             4. Browse through suggested items
#             """)
#             if st.button("Report Lost Item", width='stretch'):
#                 st.session_state.page = "😢 Report Lost Item"
#                 st.rerun()

#         with col2:
#             st.subheader("🎉 Found Something?")
#             st.write("""
#             1. Upload a photo of the found item
#             2. Describe what you found
#             3. Help us match it to owners
#             4. Make someone's day!
#             """)
#             if st.button("Report Found Item", width='stretch'):
#                 st.session_state.page = "🎉 Report Found Item"
#                 st.rerun()

#         st.markdown("---")

#         # Statistics
#         st.subheader("📈 Current Statistics")
#         col1, col2, col3 = st.columns(3)

#         with col1:
#             lost_count = len(
#                 [f for f in system['all_features'] if f['status'] == 'lost'])
#             st.metric("Lost Items", lost_count)

#         with col2:
#             found_count = len(
#                 [f for f in system['all_features'] if f['status'] == 'found'])
#             st.metric("Found Items", found_count)

#         with col3:
#             categories_count = len(CATEGORIES)
#             st.metric("Categories", categories_count)

#     # Report Lost Item
#     elif page == "😢 Report Lost Item":
#         st.header("Report a Lost Item")

#         with st.form("lost_item_form"):
#             col1, col2 = st.columns(2)

#             with col1:
#                 uploaded_image = st.file_uploader(
#                     "Upload an image of your lost item",
#                     type=['jpg', 'jpeg', 'png'],
#                     help="Upload a clear photo of the item you lost"
#                 )

#                 if uploaded_image:
#                     image = Image.open(uploaded_image)
#                     st.image(image, caption="Uploaded Image",
#                              width='stretch')

#             with col2:
#                 category = st.selectbox(
#                     "Category",
#                     CATEGORIES,
#                     help="Select the category that best matches your item"
#                 )

#                 description = st.text_area(
#                     "Description",
#                     placeholder="Describe your item: color, brand, distinguishing features, where you lost it...",
#                     height=150,
#                     help="The more details you provide, the better we can match!"
#                 )

#                 contact_info = st.text_input(
#                     "Contact Information (Optional)",
#                     placeholder="Email or phone number"
#                 )

#             submitted = st.form_submit_button(
#                 "🔍 Find Matches", width='stretch')

#         if submitted:
#             if not uploaded_image:
#                 st.error("Please upload an image of your lost item.")
#             elif not description:
#                 st.error("Please provide a description of your lost item.")
#             else:
#                 with st.spinner("Extracting features and finding matches..."):
#                     # Extract features
#                     image = Image.open(uploaded_image)
#                     new_item = extract_features_from_upload(
#                         image, description, category, system)

#                     if new_item:
#                         # Find matches
#                         matches = find_matches(new_item, system, status='lost')

#                         st.session_state.current_matches = matches

#                         # Display results
#                         st.success(
#                             f"✓ Found {len(matches)} potential matches!")

#                         if matches:
#                             st.markdown("---")
#                             st.subheader("🎯 Potential Matches")
#                             st.write(
#                                 "Here are items that might match what you lost:")

#                             for i, match in enumerate(matches, 1):
#                                 display_match_card(match, i, system)
#                         else:
#                             st.info(
#                                 "No matches found yet. We'll keep looking as new items are reported!")

#     # Report Found Item
#     elif page == "🎉 Report Found Item":
#         st.header("Report a Found Item")

#         with st.form("found_item_form"):
#             col1, col2 = st.columns(2)

#             with col1:
#                 uploaded_image = st.file_uploader(
#                     "Upload an image of the found item",
#                     type=['jpg', 'jpeg', 'png'],
#                     help="Upload a clear photo of the item you found"
#                 )

#                 if uploaded_image:
#                     image = Image.open(uploaded_image)
#                     st.image(image, caption="Uploaded Image",
#                              width='stretch')

#             with col2:
#                 category = st.selectbox(
#                     "Category",
#                     CATEGORIES,
#                     help="Select the category that best matches the item"
#                 )

#                 description = st.text_area(
#                     "Description",
#                     placeholder="Describe the item: color, brand, condition, where you found it...",
#                     height=150,
#                     help="Provide details to help match with the owner"
#                 )

#                 location = st.text_input(
#                     "Where did you find it?",
#                     placeholder="Building, room, or location"
#                 )

#             submitted = st.form_submit_button(
#                 "🔍 Find Potential Owners", width='stretch')

#         if submitted:
#             if not uploaded_image:
#                 st.error("Please upload an image of the found item.")
#             elif not description:
#                 st.error("Please provide a description of the found item.")
#             else:
#                 with st.spinner("Extracting features and finding matches..."):
#                     # Extract features
#                     image = Image.open(uploaded_image)
#                     new_item = extract_features_from_upload(
#                         image, description, category, system)

#                     if new_item:
#                         # Find matches
#                         matches = find_matches(
#                             new_item, system, status='found')

#                         st.session_state.current_matches = matches

#                         # Display results
#                         st.success(f"✓ Found {len(matches)} potential owners!")

#                         if matches:
#                             st.markdown("---")
#                             st.subheader("🎯 Potential Owners")
#                             st.write(
#                                 "These people might be looking for this item:")

#                             for i, match in enumerate(matches, 1):
#                                 display_match_card(match, i, system)
#                         else:
#                             st.info(
#                                 "No owners found yet. The item will be available for matching as people report lost items!")

#     # Browse Existing Matches
#     elif page == "🔎 Browse Existing Matches":
#         st.header("Browse Existing Matches")

#         # Performance tip
#         st.info("💡 **Performance Tip:** Uncheck 'Show images' for faster browsing with large datasets. Use pagination to view items in smaller batches.")

#         # Get project root
#         project_root = system['project_root']

#         # Load existing matches
#         matches_file = project_root / 'features' / 'matches' / 'matches.json'
#         if matches_file.exists():
#             with open(matches_file, 'r') as f:
#                 all_matches = json.load(f)

#             # Filter options
#             col1, col2, col3, col4 = st.columns(4)

#             with col1:
#                 filter_category = st.selectbox(
#                     "Filter by category",
#                     ["All"] + CATEGORIES
#                 )

#             with col2:
#                 min_similarity = st.slider(
#                     "Minimum similarity",
#                     0.0, 1.0, 0.3, 0.05
#                 )

#             with col3:
#                 items_per_page = st.selectbox(
#                     "Items per page",
#                     [5, 10, 20, 50],
#                     index=1  # Default to 10
#                 )

#             with col4:
#                 show_images = st.checkbox(
#                     "Show images",
#                     value=False,
#                     help="Uncheck for faster loading"
#                 )

#             st.markdown("---")

#             # Collect all filtered items first
#             filtered_items = []
#             for lost_id, item_matches in all_matches.items():
#                 # Get lost item metadata
#                 if lost_id not in system['id_to_meta']:
#                     continue

#                 lost_meta = system['id_to_meta'][lost_id]

#                 # Apply filters
#                 if filter_category != "All" and lost_meta['category'] != filter_category:
#                     continue

#                 # Filter by similarity
#                 filtered_matches = [
#                     m for m in item_matches if m['similarity'] >= min_similarity]

#                 if not filtered_matches:
#                     continue

#                 filtered_items.append({
#                     'lost_id': lost_id,
#                     'lost_meta': lost_meta,
#                     'matches': filtered_matches
#                 })

#             # Pagination
#             if filtered_items:
#                 total_items = len(filtered_items)
#                 total_pages = (total_items + items_per_page -
#                                1) // items_per_page

#                 # Initialize session state for page number
#                 if 'browse_page' not in st.session_state:
#                     st.session_state.browse_page = 1

#                 # Pagination controls at top
#                 col1, col2, col3, col4 = st.columns([2, 1, 1, 2])

#                 with col1:
#                     st.write(
#                         f"Showing **{total_items}** items ({total_pages} pages)")

#                 with col2:
#                     if st.button("⬅️ Previous", disabled=(st.session_state.browse_page == 1)):
#                         st.session_state.browse_page -= 1
#                         st.rerun()

#                 with col3:
#                     if st.button("Next ➡️", disabled=(st.session_state.browse_page >= total_pages)):
#                         st.session_state.browse_page += 1
#                         st.rerun()

#                 with col4:
#                     # Page selector
#                     selected_page = st.selectbox(
#                         "Go to page:",
#                         range(1, total_pages + 1),
#                         index=st.session_state.browse_page - 1,
#                         label_visibility="collapsed"
#                     )
#                     if selected_page != st.session_state.browse_page:
#                         st.session_state.browse_page = selected_page
#                         st.rerun()

#                 st.markdown("---")

#                 # Calculate slice for current page
#                 start_idx = (st.session_state.browse_page - 1) * items_per_page
#                 end_idx = min(start_idx + items_per_page, total_items)
#                 current_page_items = filtered_items[start_idx:end_idx]

#                 # Display items for current page
#                 for idx, item in enumerate(current_page_items, start=start_idx + 1):
#                     lost_id = item['lost_id']
#                     lost_meta = item['lost_meta']
#                     filtered_matches = item['matches']

#                     # Display lost item info
#                     st.subheader(f"{idx}. Lost Item: {lost_id}")
#                     col1, col2 = st.columns([1, 3])

#                     with col1:
#                         if show_images:
#                             img_path = project_root / 'dataset' / 'images' / \
#                                 lost_meta['category'] / lost_meta['filename']
#                             if img_path.exists():
#                                 try:
#                                     st.image(str(img_path), width=150)
#                                 except:
#                                     st.warning("⚠️ Image error")
#                         else:
#                             st.info(f"🖼️ Image\n({lost_meta['category']})")

#                     with col2:
#                         st.write(f"**Category:** {lost_meta['category']}")
#                         if lost_meta.get('description'):
#                             st.write(
#                                 f"**Description:** {lost_meta['description'][:100]}...")
#                         st.write(f"**Matches found:** {len(filtered_matches)}")

#                     # Display matches in expander
#                     with st.expander(f"🔍 View {len(filtered_matches)} matches", expanded=False):
#                         for i, match in enumerate(filtered_matches, 1):
#                             found_id = match['found_id']
#                             if found_id in system['id_to_meta']:
#                                 match['metadata'] = system['id_to_meta'][found_id]
#                                 display_match_card(
#                                     match, i, system, show_image=show_images)

#                     st.markdown("---")

#                 # Pagination controls at bottom
#                 st.markdown("###")
#                 col1, col2, col3 = st.columns([2, 2, 2])

#                 with col1:
#                     if st.button("⬅️ Previous Page", key="prev_bottom", disabled=(st.session_state.browse_page == 1)):
#                         st.session_state.browse_page -= 1
#                         st.rerun()

#                 with col2:
#                     st.markdown(
#                         f"<div style='text-align: center'>Page {st.session_state.browse_page} of {total_pages}</div>", unsafe_allow_html=True)

#                 with col3:
#                     if st.button("Next Page ➡️", key="next_bottom", disabled=(st.session_state.browse_page >= total_pages)):
#                         st.session_state.browse_page += 1
#                         st.rerun()

#             else:
#                 st.info("No matches found with the current filters.")
#         else:
#             st.warning(
#                 "No existing matches found. Please run the matching pipeline first.")

#     # Statistics
#     elif page == "📊 Statistics":
#         st.header("System Statistics")

#         # Overall stats
#         st.subheader("📈 Overview")
#         col1, col2, col3, col4 = st.columns(4)

#         with col1:
#             total_items = len(system['all_features'])
#             st.metric("Total Items", total_items)

#         with col2:
#             lost_count = len(
#                 [f for f in system['all_features'] if f['status'] == 'lost'])
#             st.metric("Lost Items", lost_count)

#         with col3:
#             found_count = len(
#                 [f for f in system['all_features'] if f['status'] == 'found'])
#             st.metric("Found Items", found_count)

#         with col4:
#             with_desc = len(
#                 [f for f in system['all_features'] if f['has_description']])
#             st.metric("With Descriptions", f"{with_desc/total_items*100:.0f}%")

#         st.markdown("---")

#         # Category breakdown
#         st.subheader("📦 Items by Category")

#         category_data = {}
#         for feature in system['all_features']:
#             cat = feature['category']
#             status = feature['status']

#             if cat not in category_data:
#                 category_data[cat] = {'lost': 0, 'found': 0}

#             category_data[cat][status] += 1

#         # Create DataFrame
#         df = pd.DataFrame(category_data).T
#         df['total'] = df['lost'] + df['found']
#         df = df.sort_values('total', ascending=False)

#         st.bar_chart(df[['lost', 'found']])
#         st.dataframe(df, width='stretch')


# if __name__ == "__main__":
#     main()


#######################################################################

####################### V3 #################################

# """
# Campus Lost and Found - Streamlit Application
# File: app/streamlit_app.py

# Interactive web interface for the Lost and Found matching system.
# """

# from src.matching.ranking import MatchRanker
# from src.matching.similarity import SimilarityComputer
# from src.feature_extraction.text_features import TextFeatureExtractor
# from src.feature_extraction.image_features import ImageFeatureExtractor
# import streamlit as st
# import pickle
# import json
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from PIL import Image
# import sys
# import os
# from datetime import datetime

# # Get the project root directory (parent of app directory)
# current_dir = Path(__file__).parent
# project_root = current_dir.parent

# # Add project root to path if not already there
# if str(project_root) not in sys.path:
#     sys.path.insert(0, str(project_root))

# # Now import from src

# # Page configuration
# st.set_page_config(
#     page_title="Campus Lost & Found",
#     page_icon="🔍",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS
# st.markdown("""
#     <style>
#     .main-header {
#         font-size: 3rem;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .match-card {
#         border: 2px solid #e0e0e0;
#         border-radius: 10px;
#         padding: 1rem;
#         margin: 1rem 0;
#         background-color: #f9f9f9;
#     }
#     .high-confidence {
#         border-left: 5px solid #4CAF50;
#     }
#     .medium-confidence {
#         border-left: 5px solid #FF9800;
#     }
#     .low-confidence {
#         border-left: 5px solid #f44336;
#     }
#     .similarity-score {
#         font-size: 1.5rem;
#         font-weight: bold;
#         color: #1f77b4;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Initialize session state
# if 'matcher_initialized' not in st.session_state:
#     st.session_state.matcher_initialized = False
# if 'current_matches' not in st.session_state:
#     st.session_state.current_matches = None
# if 'navigate_to' not in st.session_state:
#     st.session_state.navigate_to = None

# # Categories
# CATEGORIES = [
#     'bags', 'books', 'calculators', 'computer_mouse',
#     'earphones', 'glasses', 'id_cards', 'keys',
#     'wallets', 'water_bottles'
# ]


# @st.cache_resource
# def load_system():
#     """Load the matching system components"""
#     try:
#         # Get project root
#         current_dir = Path(__file__).parent
#         project_root = current_dir.parent

#         # Load feature extractors
#         image_extractor = ImageFeatureExtractor()
#         text_extractor = TextFeatureExtractor()

#         # Load fitted TF-IDF vectorizer
#         vectorizer_path = project_root / 'features' / 'tfidf_vectorizer.pkl'
#         with open(vectorizer_path, 'rb') as f:
#             text_extractor.tfidf_vectorizer = pickle.load(f)
#             text_extractor.is_fitted = True

#         # Load feature matrices
#         matrices_path = project_root / 'features' / 'feature_matrices.pkl'
#         with open(matrices_path, 'rb') as f:
#             feature_matrices = pickle.load(f)

#         # Load all features
#         features_path = project_root / 'features' / 'all_features.pkl'
#         with open(features_path, 'rb') as f:
#             all_features = pickle.load(f)

#         # Load metadata
#         metadata_path = project_root / 'dataset' / 'metadata_labelled.json'
#         with open(metadata_path, 'r') as f:
#             metadata = json.load(f)

#         # Create ID to metadata mapping
#         id_to_meta = {item['id']: item for item in metadata}

#         # Initialize similarity computer and ranker
#         similarity_computer = SimilarityComputer(
#             weights={'image': 0.6, 'text': 0.4})
#         ranker = MatchRanker(top_k=5, min_similarity=0.3)

#         return {
#             'image_extractor': image_extractor,
#             'text_extractor': text_extractor,
#             'similarity_computer': similarity_computer,
#             'ranker': ranker,
#             'feature_matrices': feature_matrices,
#             'all_features': all_features,
#             'id_to_meta': id_to_meta,
#             'project_root': project_root
#         }
#     except Exception as e:
#         st.error(f"Error loading system: {e}")
#         st.error(f"Current directory: {Path.cwd()}")
#         st.error(f"Project root: {project_root}")
#         return None


# def extract_features_from_upload(image, description, category, system):
#     """Extract features from uploaded image and description"""
#     try:
#         project_root = system['project_root']

#         # Save temporary image
#         temp_path = project_root / 'temp_upload.jpg'
#         image.save(temp_path)

#         # Extract image features
#         img_result = system['image_extractor'].extract_all_features(temp_path)
#         img_features = img_result['feature_vector']

#         # Extract text features
#         txt_result = system['text_extractor'].extract_all_features(
#             description, category=category)
#         txt_features = txt_result['feature_vector']

#         # Clean up
#         temp_path.unlink()

#         return {
#             'image_features': img_features,
#             'text_features': txt_features,
#             'category': category,
#             'description': description,
#             'timestamp': datetime.now().isoformat(),
#             'id': f'NEW_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
#         }
#     except Exception as e:
#         st.error(f"Error extracting features: {e}")
#         return None


# def find_matches(new_item, system, status='lost'):
#     """Find matches for a new item"""
#     try:
#         # Determine which items to match against
#         if status == 'lost':
#             # Match against found items
#             target_features = [
#                 f for f in system['all_features'] if f['status'] == 'found']
#         else:
#             # Match against lost items
#             target_features = [
#                 f for f in system['all_features'] if f['status'] == 'lost']

#         # Compute matches
#         matches = []
#         for target_item in target_features:
#             match = system['similarity_computer'].compute_match_score(
#                 new_item if status == 'lost' else target_item,
#                 target_item if status == 'lost' else new_item,
#                 require_category_match=True,
#                 require_temporal_validity=False
#             )

#             if match['valid']:
#                 # Add metadata
#                 target_id = target_item['id']
#                 if target_id in system['id_to_meta']:
#                     meta = system['id_to_meta'][target_id]
#                     match['metadata'] = meta

#                 matches.append(match)

#         # Rank matches
#         top_matches = system['ranker'].get_top_k_matches(matches)

#         return top_matches
#     except Exception as e:
#         st.error(f"Error finding matches: {e}")
#         return []


# def display_match_card(match, rank, system, show_image=True):
#     """Display a match card"""
#     match_id = match.get('found_id') or match.get('lost_id')
#     similarity = match.get('combined_similarity', match.get('similarity', 0.0))
#     project_root = system['project_root']

#     # Get confidence level
#     if similarity >= 0.7:
#         confidence = "High"
#         css_class = "high-confidence"
#     elif similarity >= 0.5:
#         confidence = "Medium"
#         css_class = "medium-confidence"
#     else:
#         confidence = "Low"
#         css_class = "low-confidence"

#     # Get metadata
#     meta = match.get('metadata', {})

#     with st.container():
#         st.markdown(
#             f'<div class="match-card {css_class}">', unsafe_allow_html=True)

#         col1, col2, col3 = st.columns([1, 3, 1])

#         with col1:
#             st.markdown(f"### Rank {rank}")
#             st.markdown(
#                 f'<div class="similarity-score">{similarity:.1%}</div>', unsafe_allow_html=True)
#             st.caption(f"{confidence} Confidence")

#         with col2:
#             st.markdown(f"**Item ID:** {match_id}")
#             st.markdown(f"**Category:** {meta.get('category', 'Unknown')}")

#             if meta.get('description'):
#                 # Truncate long descriptions
#                 desc = meta['description']
#                 if len(desc) > 150:
#                     st.markdown(f"**Description:** {desc[:150]}...")
#                 else:
#                     st.markdown(f"**Description:** {desc}")

#             if meta.get('timestamp'):
#                 st.markdown(f"**Date:** {meta['timestamp'][:10]}")

#             # Show similarity breakdown
#             with st.expander("📊 Similarity Breakdown"):
#                 if 'image_similarity' in match:
#                     st.write(f"Image Similarity: {match['image_similarity']:.1%}")
#                 if 'text_similarity' in match:
#                     st.write(f"Text Similarity: {match['text_similarity']:.1%}")
#                 if 'image_similarity' not in match and 'text_similarity' not in match:
#                     st.write(f"Overall Similarity: {similarity:.1%}")

#         with col3:
#             # Show image if available and enabled
#             if show_image and meta.get('filename') and meta.get('category'):
#                 img_path = project_root / 'dataset' / 'images' / \
#                     meta['category'] / meta['filename']
#                 if img_path.exists():
#                     try:
#                         # Use smaller width and catch any image loading errors
#                         st.image(str(img_path), width=120)
#                     except Exception as e:
#                         st.caption("🖼️ Image preview unavailable")
#             elif not show_image:
#                 st.caption("🖼️ Click to view image")

#         st.markdown('</div>', unsafe_allow_html=True)


# def main():
#     # Header
#     st.markdown('<h1 class="main-header">🔍 Campus Lost & Found</h1>',
#                 unsafe_allow_html=True)
#     st.markdown("---")

#     # Load system
#     with st.spinner("Loading matching system..."):
#         system = load_system()

#     if system is None:
#         st.error("Failed to load the matching system. Please check the setup.")
#         st.stop()

#     st.success("✓ System loaded successfully!")

#     # Sidebar navigation
#     st.sidebar.title("Navigation")

#     # Check if navigation was triggered from home page buttons
#     if 'navigate_to' in st.session_state and st.session_state.navigate_to:
#         default_page = st.session_state.navigate_to
#         st.session_state.navigate_to = None  # Reset after using
#     else:
#         default_page = "🏠 Home"

#     # Get index of default page
#     pages = ["🏠 Home", "😢 Report Lost Item", "🎉 Report Found Item",
#              "🔎 Browse Existing Matches", "📊 Statistics"]
#     default_index = pages.index(default_page) if default_page in pages else 0

#     page = st.sidebar.radio(
#         "Choose an action:",
#         pages,
#         index=default_index
#     )

#     # Home Page
#     if page == "🏠 Home":
#         st.header("Welcome to Campus Lost & Found!")

#         col1, col2 = st.columns(2)

#         with col1:
#             st.subheader("😢 Lost Something?")
#             st.write("""
#             1. Upload a photo of your lost item
#             2. Provide a description
#             3. Our AI will find potential matches
#             4. Browse through suggested items
#             """)
#             if st.button("Report Lost Item", width='stretch', key="home_lost_btn"):
#                 st.session_state.navigate_to = "😢 Report Lost Item"
#                 st.rerun()

#         with col2:
#             st.subheader("🎉 Found Something?")
#             st.write("""
#             1. Upload a photo of the found item
#             2. Describe what you found
#             3. Help us match it to owners
#             4. Make someone's day!
#             """)
#             if st.button("Report Found Item", width='stretch', key="home_found_btn"):
#                 st.session_state.navigate_to = "🎉 Report Found Item"
#                 st.rerun()

#         st.markdown("---")

#         # Statistics
#         st.subheader("📈 Current Statistics")
#         col1, col2, col3 = st.columns(3)

#         with col1:
#             lost_count = len(
#                 [f for f in system['all_features'] if f['status'] == 'lost'])
#             st.metric("Lost Items", lost_count)

#         with col2:
#             found_count = len(
#                 [f for f in system['all_features'] if f['status'] == 'found'])
#             st.metric("Found Items", found_count)

#         with col3:
#             categories_count = len(CATEGORIES)
#             st.metric("Categories", categories_count)

#     # Report Lost Item
#     elif page == "😢 Report Lost Item":
#         st.header("Report a Lost Item")

#         with st.form("lost_item_form"):
#             col1, col2 = st.columns(2)

#             with col1:
#                 uploaded_image = st.file_uploader(
#                     "Upload an image of your lost item",
#                     type=['jpg', 'jpeg', 'png'],
#                     help="Upload a clear photo of the item you lost"
#                 )

#                 if uploaded_image:
#                     image = Image.open(uploaded_image)
#                     st.image(image, caption="Uploaded Image",
#                              width='stretch')

#             with col2:
#                 category = st.selectbox(
#                     "Category",
#                     CATEGORIES,
#                     help="Select the category that best matches your item"
#                 )

#                 description = st.text_area(
#                     "Description",
#                     placeholder="Describe your item: color, brand, distinguishing features, where you lost it...",
#                     height=150,
#                     help="The more details you provide, the better we can match!"
#                 )

#                 contact_info = st.text_input(
#                     "Contact Information (Optional)",
#                     placeholder="Email or phone number"
#                 )

#             submitted = st.form_submit_button(
#                 "🔍 Find Matches", width='stretch')

#         if submitted:
#             if not uploaded_image:
#                 st.error("Please upload an image of your lost item.")
#             elif not description:
#                 st.error("Please provide a description of your lost item.")
#             else:
#                 with st.spinner("Extracting features and finding matches..."):
#                     # Extract features
#                     image = Image.open(uploaded_image)
#                     new_item = extract_features_from_upload(
#                         image, description, category, system)

#                     if new_item:
#                         # Find matches
#                         matches = find_matches(new_item, system, status='lost')

#                         st.session_state.current_matches = matches

#                         # Display results
#                         st.success(
#                             f"✓ Found {len(matches)} potential matches!")

#                         if matches:
#                             st.markdown("---")
#                             st.subheader("🎯 Potential Matches")
#                             st.write(
#                                 "Here are items that might match what you lost:")

#                             for i, match in enumerate(matches, 1):
#                                 display_match_card(match, i, system)
#                         else:
#                             st.info(
#                                 "No matches found yet. We'll keep looking as new items are reported!")

#     # Report Found Item
#     elif page == "🎉 Report Found Item":
#         st.header("Report a Found Item")

#         with st.form("found_item_form"):
#             col1, col2 = st.columns(2)

#             with col1:
#                 uploaded_image = st.file_uploader(
#                     "Upload an image of the found item",
#                     type=['jpg', 'jpeg', 'png'],
#                     help="Upload a clear photo of the item you found"
#                 )

#                 if uploaded_image:
#                     image = Image.open(uploaded_image)
#                     st.image(image, caption="Uploaded Image",
#                              width='stretch')

#             with col2:
#                 category = st.selectbox(
#                     "Category",
#                     CATEGORIES,
#                     help="Select the category that best matches the item"
#                 )

#                 description = st.text_area(
#                     "Description",
#                     placeholder="Describe the item: color, brand, condition, where you found it...",
#                     height=150,
#                     help="Provide details to help match with the owner"
#                 )

#                 location = st.text_input(
#                     "Where did you find it?",
#                     placeholder="Building, room, or location"
#                 )

#             submitted = st.form_submit_button(
#                 "🔍 Find Potential Owners", width='stretch')

#         if submitted:
#             if not uploaded_image:
#                 st.error("Please upload an image of the found item.")
#             elif not description:
#                 st.error("Please provide a description of the found item.")
#             else:
#                 with st.spinner("Extracting features and finding matches..."):
#                     # Extract features
#                     image = Image.open(uploaded_image)
#                     new_item = extract_features_from_upload(
#                         image, description, category, system)

#                     if new_item:
#                         # Find matches
#                         matches = find_matches(
#                             new_item, system, status='found')

#                         st.session_state.current_matches = matches

#                         # Display results
#                         st.success(f"✓ Found {len(matches)} potential owners!")

#                         if matches:
#                             st.markdown("---")
#                             st.subheader("🎯 Potential Owners")
#                             st.write(
#                                 "These people might be looking for this item:")

#                             for i, match in enumerate(matches, 1):
#                                 display_match_card(match, i, system)
#                         else:
#                             st.info(
#                                 "No owners found yet. The item will be available for matching as people report lost items!")

#     # Browse Existing Matches
#     elif page == "🔎 Browse Existing Matches":
#         st.header("Browse Existing Matches")

#         # Performance tip
#         st.info("💡 **Performance Tip:** Uncheck 'Show images' for faster browsing with large datasets. Use pagination to view items in smaller batches.")

#         # Get project root
#         project_root = system['project_root']

#         # Load existing matches
#         matches_file = project_root / 'features' / 'matches' / 'matches.json'
#         if matches_file.exists():
#             with open(matches_file, 'r') as f:
#                 all_matches = json.load(f)

#             # Filter options
#             col1, col2, col3, col4 = st.columns(4)

#             with col1:
#                 filter_category = st.selectbox(
#                     "Filter by category",
#                     ["All"] + CATEGORIES
#                 )

#             with col2:
#                 min_similarity = st.slider(
#                     "Minimum similarity",
#                     0.0, 1.0, 0.3, 0.05
#                 )

#             with col3:
#                 items_per_page = st.selectbox(
#                     "Items per page",
#                     [5, 10, 20, 50],
#                     index=1  # Default to 10
#                 )

#             with col4:
#                 show_images = st.checkbox(
#                     "Show images",
#                     value=False,
#                     help="Uncheck for faster loading"
#                 )

#             st.markdown("---")

#             # Collect all filtered items first
#             filtered_items = []
#             for lost_id, item_matches in all_matches.items():
#                 # Get lost item metadata
#                 if lost_id not in system['id_to_meta']:
#                     continue

#                 lost_meta = system['id_to_meta'][lost_id]

#                 # Apply filters
#                 if filter_category != "All" and lost_meta['category'] != filter_category:
#                     continue

#                 # Filter by similarity
#                 filtered_matches = [
#                     m for m in item_matches if m['similarity'] >= min_similarity]

#                 if not filtered_matches:
#                     continue

#                 filtered_items.append({
#                     'lost_id': lost_id,
#                     'lost_meta': lost_meta,
#                     'matches': filtered_matches
#                 })

#             # Pagination
#             if filtered_items:
#                 total_items = len(filtered_items)
#                 total_pages = (total_items + items_per_page -
#                                1) // items_per_page

#                 # Initialize session state for page number
#                 if 'browse_page' not in st.session_state:
#                     st.session_state.browse_page = 1

#                 # Pagination controls at top
#                 col1, col2, col3, col4 = st.columns([2, 1, 1, 2])

#                 with col1:
#                     st.write(
#                         f"Showing **{total_items}** items ({total_pages} pages)")

#                 with col2:
#                     if st.button("⬅️ Previous", disabled=(st.session_state.browse_page == 1)):
#                         st.session_state.browse_page -= 1
#                         st.rerun()

#                 with col3:
#                     if st.button("Next ➡️", disabled=(st.session_state.browse_page >= total_pages)):
#                         st.session_state.browse_page += 1
#                         st.rerun()

#                 with col4:
#                     # Page selector
#                     selected_page = st.selectbox(
#                         "Go to page:",
#                         range(1, total_pages + 1),
#                         index=st.session_state.browse_page - 1,
#                         label_visibility="collapsed"
#                     )
#                     if selected_page != st.session_state.browse_page:
#                         st.session_state.browse_page = selected_page
#                         st.rerun()

#                 st.markdown("---")

#                 # Calculate slice for current page
#                 start_idx = (st.session_state.browse_page - 1) * items_per_page
#                 end_idx = min(start_idx + items_per_page, total_items)
#                 current_page_items = filtered_items[start_idx:end_idx]

#                 # Display items for current page
#                 for idx, item in enumerate(current_page_items, start=start_idx + 1):
#                     lost_id = item['lost_id']
#                     lost_meta = item['lost_meta']
#                     filtered_matches = item['matches']

#                     # Display lost item info
#                     st.subheader(f"{idx}. Lost Item: {lost_id}")
#                     col1, col2 = st.columns([1, 3])

#                     with col1:
#                         if show_images:
#                             img_path = project_root / 'dataset' / 'images' / \
#                                 lost_meta['category'] / lost_meta['filename']
#                             if img_path.exists():
#                                 try:
#                                     st.image(str(img_path), width=150)
#                                 except:
#                                     st.warning("⚠️ Image error")
#                         else:
#                             st.info(f"🖼️ Image\n({lost_meta['category']})")

#                     with col2:
#                         st.write(f"**Category:** {lost_meta['category']}")
#                         if lost_meta.get('description'):
#                             st.write(
#                                 f"**Description:** {lost_meta['description'][:100]}...")
#                         st.write(f"**Matches found:** {len(filtered_matches)}")

#                     # Display matches in expander
#                     with st.expander(f"🔍 View {len(filtered_matches)} matches", expanded=False):
#                         for i, match in enumerate(filtered_matches, 1):
#                             found_id = match['found_id']
#                             if found_id in system['id_to_meta']:
#                                 match['metadata'] = system['id_to_meta'][found_id]
#                                 display_match_card(
#                                     match, i, system, show_image=show_images)

#                     st.markdown("---")

#                 # Pagination controls at bottom
#                 st.markdown("###")
#                 col1, col2, col3 = st.columns([2, 2, 2])

#                 with col1:
#                     if st.button("⬅️ Previous Page", key="prev_bottom", disabled=(st.session_state.browse_page == 1)):
#                         st.session_state.browse_page -= 1
#                         st.rerun()

#                 with col2:
#                     st.markdown(
#                         f"<div style='text-align: center'>Page {st.session_state.browse_page} of {total_pages}</div>", unsafe_allow_html=True)

#                 with col3:
#                     if st.button("Next Page ➡️", key="next_bottom", disabled=(st.session_state.browse_page >= total_pages)):
#                         st.session_state.browse_page += 1
#                         st.rerun()

#             else:
#                 st.info("No matches found with the current filters.")
#         else:
#             st.warning(
#                 "No existing matches found. Please run the matching pipeline first.")

#     # Statistics
#     elif page == "📊 Statistics":
#         st.header("System Statistics")

#         # Overall stats
#         st.subheader("📈 Overview")
#         col1, col2, col3, col4 = st.columns(4)

#         with col1:
#             total_items = len(system['all_features'])
#             st.metric("Total Items", total_items)

#         with col2:
#             lost_count = len(
#                 [f for f in system['all_features'] if f['status'] == 'lost'])
#             st.metric("Lost Items", lost_count)

#         with col3:
#             found_count = len(
#                 [f for f in system['all_features'] if f['status'] == 'found'])
#             st.metric("Found Items", found_count)

#         with col4:
#             with_desc = len(
#                 [f for f in system['all_features'] if f['has_description']])
#             st.metric("With Descriptions", f"{with_desc/total_items*100:.0f}%")

#         st.markdown("---")

#         # Category breakdown
#         st.subheader("📦 Items by Category")

#         category_data = {}
#         for feature in system['all_features']:
#             cat = feature['category']
#             status = feature['status']

#             if cat not in category_data:
#                 category_data[cat] = {'lost': 0, 'found': 0}

#             category_data[cat][status] += 1

#         # Create DataFrame
#         df = pd.DataFrame(category_data).T
#         df['total'] = df['lost'] + df['found']
#         df = df.sort_values('total', ascending=False)

#         st.bar_chart(df[['lost', 'found']])
#         st.dataframe(df, width='stretch')


# if __name__ == "__main__":
#     main()


###################################################################
########################## V4 #################################

# """
# Campus Lost and Found - Streamlit Application
# File: app/streamlit_app.py

# Interactive web interface for the Lost and Found matching system.
# """

# from src.matching.ml_models import KNNMatcher, SVMMatcher
# from src.matching.ranking import MatchRanker
# from src.matching.similarity import SimilarityComputer
# from src.feature_extraction.text_features import TextFeatureExtractor
# from src.feature_extraction.image_features import ImageFeatureExtractor
# import streamlit as st
# import pickle
# import json
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from PIL import Image
# import sys
# import os
# from datetime import datetime

# # Get the project root directory (parent of app directory)
# current_dir = Path(__file__).parent
# project_root = current_dir.parent

# # Add project root to path if not already there
# if str(project_root) not in sys.path:
#     sys.path.insert(0, str(project_root))

# # Now import from src

# # Page configuration
# st.set_page_config(
#     page_title="Campus Lost & Found",
#     page_icon="🔍",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS
# st.markdown("""
#     <style>
#     .main-header {
#         font-size: 3rem;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .match-card {
#         border: 2px solid #e0e0e0;
#         border-radius: 10px;
#         padding: 1rem;
#         margin: 1rem 0;
#         background-color: #f9f9f9;
#     }
#     .high-confidence {
#         border-left: 5px solid #4CAF50;
#     }
#     .medium-confidence {
#         border-left: 5px solid #FF9800;
#     }
#     .low-confidence {
#         border-left: 5px solid #f44336;
#     }
#     .similarity-score {
#         font-size: 1.5rem;
#         font-weight: bold;
#         color: #1f77b4;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Initialize session state
# if 'matcher_initialized' not in st.session_state:
#     st.session_state.matcher_initialized = False
# if 'current_matches' not in st.session_state:
#     st.session_state.current_matches = None
# if 'navigate_to' not in st.session_state:
#     st.session_state.navigate_to = None

# # Categories
# CATEGORIES = [
#     'bags', 'books', 'calculators', 'computer_mouse',
#     'earphones', 'glasses', 'id_cards', 'keys',
#     'wallets', 'water_bottles'
# ]


# @st.cache_resource
# def load_system():
#     """Load the matching system components"""
#     try:
#         # Get project root
#         current_dir = Path(__file__).parent
#         project_root = current_dir.parent

#         # Load feature extractors
#         image_extractor = ImageFeatureExtractor()
#         text_extractor = TextFeatureExtractor()

#         # Load fitted TF-IDF vectorizer
#         vectorizer_path = project_root / 'features' / 'tfidf_vectorizer.pkl'
#         with open(vectorizer_path, 'rb') as f:
#             text_extractor.tfidf_vectorizer = pickle.load(f)
#             text_extractor.is_fitted = True

#         # Load feature matrices
#         matrices_path = project_root / 'features' / 'feature_matrices.pkl'
#         with open(matrices_path, 'rb') as f:
#             feature_matrices = pickle.load(f)

#         # Load all features
#         features_path = project_root / 'features' / 'all_features.pkl'
#         with open(features_path, 'rb') as f:
#             all_features = pickle.load(f)

#         # Load metadata
#         metadata_path = project_root / 'dataset' / 'metadata_labelled.json'
#         with open(metadata_path, 'r') as f:
#             metadata = json.load(f)

#         # Create ID to metadata mapping
#         id_to_meta = {item['id']: item for item in metadata}

#         # Initialize similarity computer and ranker
#         similarity_computer = SimilarityComputer(
#             weights={'image': 0.6, 'text': 0.4})
#         ranker = MatchRanker(top_k=5, min_similarity=0.3)

#         # Try to load ML models (KNN and SVM)
#         knn_matcher = None
#         svm_matcher = None

#         models_dir = project_root / 'models'

#         # Load KNN
#         knn_path = models_dir / 'knn_matcher.pkl'
#         if knn_path.exists():
#             try:
#                 knn_matcher = KNNMatcher.load(knn_path)
#                 st.sidebar.success("✓ KNN model loaded")
#             except Exception as e:
#                 st.sidebar.warning(f"⚠ KNN model not available: {str(e)[:50]}")

#         # Load SVM
#         svm_path = models_dir / 'svm_matcher.pkl'
#         if svm_path.exists():
#             try:
#                 svm_matcher = SVMMatcher.load(svm_path)
#                 st.sidebar.success("✓ SVM model loaded")
#             except Exception as e:
#                 st.sidebar.warning(f"⚠ SVM model not available: {str(e)[:50]}")

#         return {
#             'image_extractor': image_extractor,
#             'text_extractor': text_extractor,
#             'similarity_computer': similarity_computer,
#             'ranker': ranker,
#             'knn_matcher': knn_matcher,
#             'svm_matcher': svm_matcher,
#             'feature_matrices': feature_matrices,
#             'all_features': all_features,
#             'id_to_meta': id_to_meta,
#             'project_root': project_root
#         }
#     except Exception as e:
#         st.error(f"Error loading system: {e}")
#         st.error(f"Current directory: {Path.cwd()}")
#         st.error(f"Project root: {project_root}")
#         return None


# def extract_features_from_upload(image, description, category, system):
#     """Extract features from uploaded image and description"""
#     try:
#         project_root = system['project_root']

#         # Save temporary image
#         temp_path = project_root / 'temp_upload.jpg'
#         image.save(temp_path)

#         # Extract image features
#         img_result = system['image_extractor'].extract_all_features(temp_path)
#         img_features = img_result['feature_vector']

#         # Extract text features
#         txt_result = system['text_extractor'].extract_all_features(
#             description, category=category)
#         txt_features = txt_result['feature_vector']

#         # Clean up
#         temp_path.unlink()

#         return {
#             'image_features': img_features,
#             'text_features': txt_features,
#             'category': category,
#             'description': description,
#             'timestamp': datetime.now().isoformat(),
#             'id': f'NEW_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
#         }
#     except Exception as e:
#         st.error(f"Error extracting features: {e}")
#         return None


# def find_matches(new_item, system, status='lost', model_type='similarity'):
#     """Find matches for a new item using selected model"""
#     try:
#         # Determine which items to match against
#         if status == 'lost':
#             # Match against found items
#             target_features_list = [
#                 f for f in system['all_features'] if f['status'] == 'found']
#         else:
#             # Match against lost items
#             target_features_list = [
#                 f for f in system['all_features'] if f['status'] == 'lost']

#         if model_type == 'similarity':
#             # Original similarity-based matching
#             matches = []
#             for target_item in target_features_list:
#                 match = system['similarity_computer'].compute_match_score(
#                     new_item if status == 'lost' else target_item,
#                     target_item if status == 'lost' else new_item,
#                     require_category_match=True,
#                     require_temporal_validity=False
#                 )

#                 if match['valid']:
#                     # Add metadata
#                     target_id = target_item['id']
#                     if target_id in system['id_to_meta']:
#                         meta = system['id_to_meta'][target_id]
#                         match['metadata'] = meta

#                     # Ensure all keys exist
#                     if 'combined_similarity' not in match:
#                         match['combined_similarity'] = match.get(
#                             'similarity', 0.0)
#                     if 'image_similarity' not in match:
#                         match['image_similarity'] = match.get(
#                             'combined_similarity', 0.0)
#                     if 'text_similarity' not in match:
#                         match['text_similarity'] = match.get(
#                             'combined_similarity', 0.0)

#                     matches.append(match)

#             # Rank matches
#             top_matches = system['ranker'].get_top_k_matches(matches)
#             return top_matches

#         elif model_type == 'knn':
#             # KNN-based matching
#             if system['knn_matcher'] is None:
#                 st.error("KNN model not available. Please train the model first.")
#                 return []

#             # Get combined features
#             new_features = np.concatenate(
#                 [new_item['image_features'], new_item['text_features']])

#             # Predict using KNN
#             knn_matches = system['knn_matcher'].predict(
#                 new_features,
#                 new_item['category'],
#                 filter_category=True
#             )

#             # Add metadata and convert format
#             matches = []
#             for match in knn_matches:
#                 found_id = match['found_id']
#                 if found_id in system['id_to_meta']:
#                     match['metadata'] = system['id_to_meta'][found_id]
#                     # Ensure consistent keys
#                     similarity = match.get('similarity', 0.0)
#                     match['combined_similarity'] = similarity
#                     # KNN doesn't separate these
#                     match['image_similarity'] = similarity
#                     match['text_similarity'] = similarity
#                     matches.append(match)

#             return matches

#         elif model_type == 'svm':
#             # SVM-based matching
#             if system['svm_matcher'] is None:
#                 st.error("SVM model not available. Please train the model first.")
#                 return []

#             # Get features
#             new_features = np.concatenate(
#                 [new_item['image_features'], new_item['text_features']])

#             # Get found item features and IDs
#             found_features = system['feature_matrices']['found']['combined_features']
#             found_ids = system['feature_matrices']['found']['ids']
#             found_categories = system['feature_matrices']['found']['categories']

#             # Predict using SVM
#             svm_matches = system['svm_matcher'].predict(
#                 new_features,
#                 found_features,
#                 found_ids,
#                 new_item['category'],
#                 found_categories,
#                 filter_category=True,
#                 top_k=5
#             )

#             # Add metadata and convert format
#             matches = []
#             for match in svm_matches:
#                 found_id = match['found_id']
#                 if found_id in system['id_to_meta']:
#                     match['metadata'] = system['id_to_meta'][found_id]
#                     # Ensure consistent keys
#                     similarity = match.get('similarity', 0.0)
#                     match['combined_similarity'] = similarity
#                     # SVM doesn't separate these
#                     match['image_similarity'] = similarity
#                     match['text_similarity'] = similarity
#                     matches.append(match)

#             return matches

#         else:
#             st.error(f"Unknown model type: {model_type}")
#             return []

#     except Exception as e:
#         st.error(f"Error finding matches: {e}")
#         import traceback
#         st.error(traceback.format_exc())
#         return []


# def display_match_card(match, rank, system, show_image=True):
#     """Display a match card"""
#     match_id = match.get('found_id') or match.get('lost_id')
#     similarity = match.get('combined_similarity', match.get('similarity', 0.0))
#     project_root = system['project_root']

#     # Get confidence level
#     if similarity >= 0.7:
#         confidence = "High"
#         css_class = "high-confidence"
#     elif similarity >= 0.5:
#         confidence = "Medium"
#         css_class = "medium-confidence"
#     else:
#         confidence = "Low"
#         css_class = "low-confidence"

#     # Get metadata
#     meta = match.get('metadata', {})

#     with st.container():
#         st.markdown(
#             f'<div class="match-card {css_class}">', unsafe_allow_html=True)

#         col1, col2, col3 = st.columns([1, 3, 1])

#         with col1:
#             st.markdown(f"### Rank {rank}")
#             st.markdown(
#                 f'<div class="similarity-score">{similarity:.1%}</div>', unsafe_allow_html=True)
#             st.caption(f"{confidence} Confidence")

#         with col2:
#             st.markdown(f"**Item ID:** {match_id}")
#             st.markdown(f"**Category:** {meta.get('category', 'Unknown')}")

#             if meta.get('description'):
#                 # Truncate long descriptions
#                 desc = meta['description']
#                 if len(desc) > 150:
#                     st.markdown(f"**Description:** {desc[:150]}...")
#                 else:
#                     st.markdown(f"**Description:** {desc}")

#             if meta.get('timestamp'):
#                 st.markdown(f"**Date:** {meta['timestamp'][:10]}")

#             # Show similarity breakdown
#             with st.expander("📊 Similarity Breakdown"):
#                 if 'image_similarity' in match:
#                     st.write(
#                         f"Image Similarity: {match['image_similarity']:.1%}")
#                 if 'text_similarity' in match:
#                     st.write(
#                         f"Text Similarity: {match['text_similarity']:.1%}")
#                 if 'image_similarity' not in match and 'text_similarity' not in match:
#                     st.write(f"Overall Similarity: {similarity:.1%}")

#         with col3:
#             # Show image if available and enabled
#             if show_image and meta.get('filename') and meta.get('category'):
#                 img_path = project_root / 'dataset' / 'images' / \
#                     meta['category'] / meta['filename']
#                 if img_path.exists():
#                     try:
#                         # Use smaller width and catch any image loading errors
#                         st.image(str(img_path), width=120)
#                     except Exception as e:
#                         st.caption("🖼️ Image preview unavailable")
#             elif not show_image:
#                 st.caption("🖼️ Click to view image")

#         st.markdown('</div>', unsafe_allow_html=True)


# def main():
#     # Header
#     st.markdown('<h1 class="main-header">🔍 Campus Lost & Found</h1>',
#                 unsafe_allow_html=True)
#     st.markdown("---")

#     # Load system
#     with st.spinner("Loading matching system..."):
#         system = load_system()

#     if system is None:
#         st.error("Failed to load the matching system. Please check the setup.")
#         st.stop()

#     st.success("✓ System loaded successfully!")

#     # Sidebar navigation
#     st.sidebar.title("Navigation")

#     # Check if navigation was triggered from home page buttons
#     if 'navigate_to' in st.session_state and st.session_state.navigate_to:
#         default_page = st.session_state.navigate_to
#         st.session_state.navigate_to = None  # Reset after using
#     else:
#         default_page = "🏠 Home"

#     # Get index of default page
#     pages = ["🏠 Home", "😢 Report Lost Item", "🎉 Report Found Item",
#              "🔎 Browse Existing Matches", "📊 Statistics"]
#     default_index = pages.index(default_page) if default_page in pages else 0

#     page = st.sidebar.radio(
#         "Choose an action:",
#         pages,
#         index=default_index
#     )

#     # Model selection
#     st.sidebar.markdown("---")
#     st.sidebar.subheader("⚙️ Model Settings")

#     # Check which models are available
#     available_models = ["Similarity-based"]
#     if system.get('knn_matcher') is not None:
#         available_models.append("KNN")
#     if system.get('svm_matcher') is not None:
#         available_models.append("SVM")

#     model_choice = st.sidebar.selectbox(
#         "Matching Algorithm:",
#         available_models,
#         help="""
#         • Similarity-based: Fast, weighted image+text features
#         • KNN: Nearest neighbors in feature space
#         • SVM: Trained classifier for match prediction
#         """
#     )

#     # Map to internal model type
#     model_type_map = {
#         "Similarity-based": "similarity",
#         "KNN": "knn",
#         "SVM": "svm"
#     }
#     model_type = model_type_map[model_choice]

#     # Store in session state
#     if 'selected_model' not in st.session_state:
#         st.session_state.selected_model = model_type
#     else:
#         st.session_state.selected_model = model_type

#     # Show model info
#     with st.sidebar.expander("ℹ️ About Models"):
#         if model_choice == "Similarity-based":
#             st.write("**Cosine similarity** on combined image and text features.")
#             st.write("✅ Fast")
#             st.write("✅ Interpretable")
#             st.write("✅ No training required")
#         elif model_choice == "KNN":
#             st.write(
#                 "**K-Nearest Neighbors** finds similar items in feature space.")
#             st.write("✅ Simple")
#             st.write("✅ Non-parametric")
#             st.write("⚠️ Requires training")
#         elif model_choice == "SVM":
#             st.write("**Support Vector Machine** learns to classify matches.")
#             st.write("✅ Powerful")
#             st.write("✅ Probability scores")
#             st.write("⚠️ Requires labeled data")

#     # Training status
#     if len(available_models) < 3:
#         st.sidebar.info(
#             f"💡 Train additional models to unlock more options. Currently available: {len(available_models)}/3")

#     # Home Page
#     if page == "🏠 Home":
#         st.header("Welcome to Campus Lost & Found!")

#         col1, col2 = st.columns(2)

#         with col1:
#             st.subheader("😢 Lost Something?")
#             st.write("""
#             1. Upload a photo of your lost item
#             2. Provide a description
#             3. Our AI will find potential matches
#             4. Browse through suggested items
#             """)
#             if st.button("Report Lost Item", width='stretch', key="home_lost_btn"):
#                 st.session_state.navigate_to = "😢 Report Lost Item"
#                 st.rerun()

#         with col2:
#             st.subheader("🎉 Found Something?")
#             st.write("""
#             1. Upload a photo of the found item
#             2. Describe what you found
#             3. Help us match it to owners
#             4. Make someone's day!
#             """)
#             if st.button("Report Found Item", width='stretch', key="home_found_btn"):
#                 st.session_state.navigate_to = "🎉 Report Found Item"
#                 st.rerun()

#         st.markdown("---")

#         # Statistics
#         st.subheader("📈 Current Statistics")
#         col1, col2, col3 = st.columns(3)

#         with col1:
#             lost_count = len(
#                 [f for f in system['all_features'] if f['status'] == 'lost'])
#             st.metric("Lost Items", lost_count)

#         with col2:
#             found_count = len(
#                 [f for f in system['all_features'] if f['status'] == 'found'])
#             st.metric("Found Items", found_count)

#         with col3:
#             categories_count = len(CATEGORIES)
#             st.metric("Categories", categories_count)

#     # Report Lost Item
#     elif page == "😢 Report Lost Item":
#         st.header("Report a Lost Item")

#         with st.form("lost_item_form"):
#             col1, col2 = st.columns(2)

#             with col1:
#                 uploaded_image = st.file_uploader(
#                     "Upload an image of your lost item",
#                     type=['jpg', 'jpeg', 'png'],
#                     help="Upload a clear photo of the item you lost"
#                 )

#                 if uploaded_image:
#                     image = Image.open(uploaded_image)
#                     st.image(image, caption="Uploaded Image",
#                              width='stretch')

#             with col2:
#                 category = st.selectbox(
#                     "Category",
#                     CATEGORIES,
#                     help="Select the category that best matches your item"
#                 )

#                 description = st.text_area(
#                     "Description",
#                     placeholder="Describe your item: color, brand, distinguishing features, where you lost it...",
#                     height=150,
#                     help="The more details you provide, the better we can match!"
#                 )

#                 contact_info = st.text_input(
#                     "Contact Information (Optional)",
#                     placeholder="Email or phone number"
#                 )

#             submitted = st.form_submit_button(
#                 "🔍 Find Matches", width='stretch')

#         if submitted:
#             if not uploaded_image:
#                 st.error("Please upload an image of your lost item.")
#             elif not description:
#                 st.error("Please provide a description of your lost item.")
#             else:
#                 with st.spinner("Extracting features and finding matches..."):
#                     # Extract features
#                     image = Image.open(uploaded_image)
#                     new_item = extract_features_from_upload(
#                         image, description, category, system)

#                     if new_item:
#                         # Find matches using selected model
#                         matches = find_matches(
#                             new_item, system, status='lost', model_type=st.session_state.selected_model)

#                         st.session_state.current_matches = matches

#                         # Display results
#                         st.success(
#                             f"✓ Found {len(matches)} potential matches using **{model_choice}** algorithm!")

#                         if matches:
#                             st.markdown("---")
#                             st.subheader("🎯 Potential Matches")
#                             st.write(
#                                 "Here are items that might match what you lost:")

#                             for i, match in enumerate(matches, 1):
#                                 display_match_card(match, i, system)
#                         else:
#                             st.info(
#                                 "No matches found yet. We'll keep looking as new items are reported!")

#     # Report Found Item
#     elif page == "🎉 Report Found Item":
#         st.header("Report a Found Item")

#         with st.form("found_item_form"):
#             col1, col2 = st.columns(2)

#             with col1:
#                 uploaded_image = st.file_uploader(
#                     "Upload an image of the found item",
#                     type=['jpg', 'jpeg', 'png'],
#                     help="Upload a clear photo of the item you found"
#                 )

#                 if uploaded_image:
#                     image = Image.open(uploaded_image)
#                     st.image(image, caption="Uploaded Image",
#                              width='stretch')

#             with col2:
#                 category = st.selectbox(
#                     "Category",
#                     CATEGORIES,
#                     help="Select the category that best matches the item"
#                 )

#                 description = st.text_area(
#                     "Description",
#                     placeholder="Describe the item: color, brand, condition, where you found it...",
#                     height=150,
#                     help="Provide details to help match with the owner"
#                 )

#                 location = st.text_input(
#                     "Where did you find it?",
#                     placeholder="Building, room, or location"
#                 )

#             submitted = st.form_submit_button(
#                 "🔍 Find Potential Owners", width='stretch')

#         if submitted:
#             if not uploaded_image:
#                 st.error("Please upload an image of the found item.")
#             elif not description:
#                 st.error("Please provide a description of the found item.")
#             else:
#                 with st.spinner("Extracting features and finding matches..."):
#                     # Extract features
#                     image = Image.open(uploaded_image)
#                     new_item = extract_features_from_upload(
#                         image, description, category, system)

#                     if new_item:
#                         # Find matches using selected model
#                         matches = find_matches(
#                             new_item, system, status='found', model_type=st.session_state.selected_model)

#                         st.session_state.current_matches = matches

#                         # Display results
#                         st.success(
#                             f"✓ Found {len(matches)} potential owners using **{model_choice}** algorithm!")

#                         if matches:
#                             st.markdown("---")
#                             st.subheader("🎯 Potential Owners")
#                             st.write(
#                                 "These people might be looking for this item:")

#                             for i, match in enumerate(matches, 1):
#                                 display_match_card(match, i, system)
#                         else:
#                             st.info(
#                                 "No owners found yet. The item will be available for matching as people report lost items!")

#     # Browse Existing Matches
#     elif page == "🔎 Browse Existing Matches":
#         st.header("Browse Existing Matches")

#         # Performance tip
#         st.info("💡 **Performance Tip:** Uncheck 'Show images' for faster browsing with large datasets. Use pagination to view items in smaller batches.")

#         # Get project root
#         project_root = system['project_root']

#         # Load existing matches
#         matches_file = project_root / 'features' / 'matches' / 'matches.json'
#         if matches_file.exists():
#             with open(matches_file, 'r') as f:
#                 all_matches = json.load(f)

#             # Filter options
#             col1, col2, col3, col4 = st.columns(4)

#             with col1:
#                 filter_category = st.selectbox(
#                     "Filter by category",
#                     ["All"] + CATEGORIES
#                 )

#             with col2:
#                 min_similarity = st.slider(
#                     "Minimum similarity",
#                     0.0, 1.0, 0.3, 0.05
#                 )

#             with col3:
#                 items_per_page = st.selectbox(
#                     "Items per page",
#                     [5, 10, 20, 50],
#                     index=1  # Default to 10
#                 )

#             with col4:
#                 show_images = st.checkbox(
#                     "Show images",
#                     value=False,
#                     help="Uncheck for faster loading"
#                 )

#             st.markdown("---")

#             # Collect all filtered items first
#             filtered_items = []
#             for lost_id, item_matches in all_matches.items():
#                 # Get lost item metadata
#                 if lost_id not in system['id_to_meta']:
#                     continue

#                 lost_meta = system['id_to_meta'][lost_id]

#                 # Apply filters
#                 if filter_category != "All" and lost_meta['category'] != filter_category:
#                     continue

#                 # Filter by similarity
#                 filtered_matches = [
#                     m for m in item_matches if m['similarity'] >= min_similarity]

#                 if not filtered_matches:
#                     continue

#                 filtered_items.append({
#                     'lost_id': lost_id,
#                     'lost_meta': lost_meta,
#                     'matches': filtered_matches
#                 })

#             # Pagination
#             if filtered_items:
#                 total_items = len(filtered_items)
#                 total_pages = (total_items + items_per_page -
#                                1) // items_per_page

#                 # Initialize session state for page number
#                 if 'browse_page' not in st.session_state:
#                     st.session_state.browse_page = 1

#                 # Pagination controls at top
#                 col1, col2, col3, col4 = st.columns([2, 1, 1, 2])

#                 with col1:
#                     st.write(
#                         f"Showing **{total_items}** items ({total_pages} pages)")

#                 with col2:
#                     if st.button("⬅️ Previous", disabled=(st.session_state.browse_page == 1)):
#                         st.session_state.browse_page -= 1
#                         st.rerun()

#                 with col3:
#                     if st.button("Next ➡️", disabled=(st.session_state.browse_page >= total_pages)):
#                         st.session_state.browse_page += 1
#                         st.rerun()

#                 with col4:
#                     # Page selector
#                     selected_page = st.selectbox(
#                         "Go to page:",
#                         range(1, total_pages + 1),
#                         index=st.session_state.browse_page - 1,
#                         label_visibility="collapsed"
#                     )
#                     if selected_page != st.session_state.browse_page:
#                         st.session_state.browse_page = selected_page
#                         st.rerun()

#                 st.markdown("---")

#                 # Calculate slice for current page
#                 start_idx = (st.session_state.browse_page - 1) * items_per_page
#                 end_idx = min(start_idx + items_per_page, total_items)
#                 current_page_items = filtered_items[start_idx:end_idx]

#                 # Display items for current page
#                 for idx, item in enumerate(current_page_items, start=start_idx + 1):
#                     lost_id = item['lost_id']
#                     lost_meta = item['lost_meta']
#                     filtered_matches = item['matches']

#                     # Display lost item info
#                     st.subheader(f"{idx}. Lost Item: {lost_id}")
#                     col1, col2 = st.columns([1, 3])

#                     with col1:
#                         if show_images:
#                             img_path = project_root / 'dataset' / 'images' / \
#                                 lost_meta['category'] / lost_meta['filename']
#                             if img_path.exists():
#                                 try:
#                                     st.image(str(img_path), width=150)
#                                 except:
#                                     st.warning("⚠️ Image error")
#                         else:
#                             st.info(f"🖼️ Image\n({lost_meta['category']})")

#                     with col2:
#                         st.write(f"**Category:** {lost_meta['category']}")
#                         if lost_meta.get('description'):
#                             st.write(
#                                 f"**Description:** {lost_meta['description'][:100]}...")
#                         st.write(f"**Matches found:** {len(filtered_matches)}")

#                     # Display matches in expander
#                     with st.expander(f"🔍 View {len(filtered_matches)} matches", expanded=False):
#                         for i, match in enumerate(filtered_matches, 1):
#                             found_id = match['found_id']
#                             if found_id in system['id_to_meta']:
#                                 match['metadata'] = system['id_to_meta'][found_id]
#                                 display_match_card(
#                                     match, i, system, show_image=show_images)

#                     st.markdown("---")

#                 # Pagination controls at bottom
#                 st.markdown("###")
#                 col1, col2, col3 = st.columns([2, 2, 2])

#                 with col1:
#                     if st.button("⬅️ Previous Page", key="prev_bottom", disabled=(st.session_state.browse_page == 1)):
#                         st.session_state.browse_page -= 1
#                         st.rerun()

#                 with col2:
#                     st.markdown(
#                         f"<div style='text-align: center'>Page {st.session_state.browse_page} of {total_pages}</div>", unsafe_allow_html=True)

#                 with col3:
#                     if st.button("Next Page ➡️", key="next_bottom", disabled=(st.session_state.browse_page >= total_pages)):
#                         st.session_state.browse_page += 1
#                         st.rerun()

#             else:
#                 st.info("No matches found with the current filters.")
#         else:
#             st.warning(
#                 "No existing matches found. Please run the matching pipeline first.")

#     # Statistics
#     elif page == "📊 Statistics":
#         st.header("System Statistics")

#         # Overall stats
#         st.subheader("📈 Overview")
#         col1, col2, col3, col4 = st.columns(4)

#         with col1:
#             total_items = len(system['all_features'])
#             st.metric("Total Items", total_items)

#         with col2:
#             lost_count = len(
#                 [f for f in system['all_features'] if f['status'] == 'lost'])
#             st.metric("Lost Items", lost_count)

#         with col3:
#             found_count = len(
#                 [f for f in system['all_features'] if f['status'] == 'found'])
#             st.metric("Found Items", found_count)

#         with col4:
#             with_desc = len(
#                 [f for f in system['all_features'] if f['has_description']])
#             st.metric("With Descriptions", f"{with_desc/total_items*100:.0f}%")

#         st.markdown("---")

#         # Category breakdown
#         st.subheader("📦 Items by Category")

#         category_data = {}
#         for feature in system['all_features']:
#             cat = feature['category']
#             status = feature['status']

#             if cat not in category_data:
#                 category_data[cat] = {'lost': 0, 'found': 0}

#             category_data[cat][status] += 1

#         # Create DataFrame
#         df = pd.DataFrame(category_data).T
#         df['total'] = df['lost'] + df['found']
#         df = df.sort_values('total', ascending=False)

#         st.bar_chart(df[['lost', 'found']])
#         st.dataframe(df, width='stretch')


# if __name__ == "__main__":
#     main()


########################################################################################
#################################### V5 ###############################################

"""
Campus Lost and Found - Streamlit Application
File: app/streamlit_app.py

Interactive web interface for the Lost and Found matching system.
"""

# Add project root to path FIRST, before any imports
import sys
from pathlib import Path

# Get the project root directory (parent of app directory)
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now import everything else
import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd
from PIL import Image
import os
from datetime import datetime

# Import from src after path is set up
from src.feature_extraction.image_features import ImageFeatureExtractor
from src.feature_extraction.text_features import TextFeatureExtractor
from src.matching.similarity import SimilarityComputer
from src.matching.ranking import MatchRanker
from src.matching.ml_models import KNNMatcher, SVMMatcher

# Page configuration
st.set_page_config(
    page_title="Campus Lost & Found",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .match-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f9f9f9;
    }
    .high-confidence {
        border-left: 5px solid #4CAF50;
    }
    .medium-confidence {
        border-left: 5px solid #FF9800;
    }
    .low-confidence {
        border-left: 5px solid #f44336;
    }
    .similarity-score {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'matcher_initialized' not in st.session_state:
    st.session_state.matcher_initialized = False
if 'current_matches' not in st.session_state:
    st.session_state.current_matches = None
if 'navigate_to' not in st.session_state:
    st.session_state.navigate_to = None

# Categories
CATEGORIES = [
    'bags', 'books', 'calculators', 'computer_mouse', 
    'earphones', 'glasses', 'id_cards', 'keys', 
    'wallets', 'water_bottles'
]


@st.cache_resource
def load_system():
    """Load the matching system components"""
    try:
        # Get project root
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        
        # Load feature extractors
        image_extractor = ImageFeatureExtractor()
        text_extractor = TextFeatureExtractor()
        
        # Load fitted TF-IDF vectorizer
        vectorizer_path = project_root / 'features' / 'tfidf_vectorizer.pkl'
        with open(vectorizer_path, 'rb') as f:
            text_extractor.tfidf_vectorizer = pickle.load(f)
            text_extractor.is_fitted = True
        
        # Load feature matrices
        matrices_path = project_root / 'features' / 'feature_matrices.pkl'
        with open(matrices_path, 'rb') as f:
            feature_matrices = pickle.load(f)
        
        # Load all features
        features_path = project_root / 'features' / 'all_features.pkl'
        with open(features_path, 'rb') as f:
            all_features = pickle.load(f)
        
        # Load metadata
        metadata_path = project_root / 'dataset' / 'metadata_labelled.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create ID to metadata mapping
        id_to_meta = {item['id']: item for item in metadata}
        
        # Initialize similarity computer and ranker
        similarity_computer = SimilarityComputer(weights={'image': 0.6, 'text': 0.4})
        ranker = MatchRanker(top_k=5, min_similarity=0.3)
        
        # Try to load ML models (KNN and SVM)
        knn_matcher = None
        svm_matcher = None
        
        models_dir = project_root / 'models'
        
        # Load KNN
        knn_path = models_dir / 'knn_matcher.pkl'
        if knn_path.exists():
            try:
                knn_matcher = KNNMatcher.load(knn_path)
                st.sidebar.success("✓ KNN model loaded")
            except Exception as e:
                st.sidebar.warning(f"⚠ KNN model not available: {str(e)[:50]}")
        
        # Load SVM
        svm_path = models_dir / 'svm_matcher.pkl'
        if svm_path.exists():
            try:
                svm_matcher = SVMMatcher.load(svm_path)
                st.sidebar.success("✓ SVM model loaded")
            except Exception as e:
                st.sidebar.warning(f"⚠ SVM model not available: {str(e)[:50]}")
        
        return {
            'image_extractor': image_extractor,
            'text_extractor': text_extractor,
            'similarity_computer': similarity_computer,
            'ranker': ranker,
            'knn_matcher': knn_matcher,
            'svm_matcher': svm_matcher,
            'feature_matrices': feature_matrices,
            'all_features': all_features,
            'id_to_meta': id_to_meta,
            'project_root': project_root
        }
    except Exception as e:
        st.error(f"Error loading system: {e}")
        st.error(f"Current directory: {Path.cwd()}")
        st.error(f"Project root: {project_root}")
        return None


def extract_features_from_upload(image, description, category, system):
    """Extract features from uploaded image and description"""
    try:
        project_root = system['project_root']
        
        # Save temporary image
        temp_path = project_root / 'temp_upload.jpg'
        image.save(temp_path)
        
        # Extract image features
        img_result = system['image_extractor'].extract_all_features(temp_path)
        img_features = img_result['feature_vector']
        
        # Extract text features
        txt_result = system['text_extractor'].extract_all_features(description, category=category)
        txt_features = txt_result['feature_vector']
        
        # Clean up
        temp_path.unlink()
        
        return {
            'image_features': img_features,
            'text_features': txt_features,
            'category': category,
            'description': description,
            'timestamp': datetime.now().isoformat(),
            'id': f'NEW_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        }
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None


def find_matches(new_item, system, status='lost', model_type='similarity'):
    """Find matches for a new item using selected model"""
    try:
        # Determine which items to match against
        if status == 'lost':
            # Match against found items
            target_features_list = [f for f in system['all_features'] if f['status'] == 'found']
        else:
            # Match against lost items
            target_features_list = [f for f in system['all_features'] if f['status'] == 'lost']
        
        if model_type == 'similarity':
            # Original similarity-based matching
            matches = []
            for target_item in target_features_list:
                match = system['similarity_computer'].compute_match_score(
                    new_item if status == 'lost' else target_item,
                    target_item if status == 'lost' else new_item,
                    require_category_match=True,
                    require_temporal_validity=False
                )
                
                if match['valid']:
                    # Add metadata
                    target_id = target_item['id']
                    if target_id in system['id_to_meta']:
                        meta = system['id_to_meta'][target_id]
                        match['metadata'] = meta
                    
                    # Ensure all keys exist
                    if 'combined_similarity' not in match:
                        match['combined_similarity'] = match.get('similarity', 0.0)
                    if 'image_similarity' not in match:
                        match['image_similarity'] = match.get('combined_similarity', 0.0)
                    if 'text_similarity' not in match:
                        match['text_similarity'] = match.get('combined_similarity', 0.0)
                    
                    matches.append(match)
            
            # Rank matches
            top_matches = system['ranker'].get_top_k_matches(matches)
            return top_matches
        
        elif model_type == 'knn':
            # KNN-based matching
            if system['knn_matcher'] is None:
                st.error("KNN model not available. Please train the model first.")
                return []
            
            # Get combined features
            new_features = np.concatenate([new_item['image_features'], new_item['text_features']])
            
            # Predict using KNN
            knn_matches = system['knn_matcher'].predict(
                new_features,
                new_item['category'],
                filter_category=True
            )
            
            # Add metadata and convert format
            matches = []
            for match in knn_matches:
                found_id = match['found_id']
                if found_id in system['id_to_meta']:
                    match['metadata'] = system['id_to_meta'][found_id]
                    # Ensure consistent keys
                    similarity = match.get('similarity', 0.0)
                    match['combined_similarity'] = similarity
                    match['image_similarity'] = similarity  # KNN doesn't separate these
                    match['text_similarity'] = similarity
                    matches.append(match)
            
            return matches
        
        elif model_type == 'svm':
            # SVM-based matching
            if system['svm_matcher'] is None:
                st.error("SVM model not available. Please train the model first.")
                return []
            
            # Get features
            new_features = np.concatenate([new_item['image_features'], new_item['text_features']])
            
            # Get found item features and IDs
            found_features = system['feature_matrices']['found']['combined_features']
            found_ids = system['feature_matrices']['found']['ids']
            found_categories = system['feature_matrices']['found']['categories']
            
            # Predict using SVM
            svm_matches = system['svm_matcher'].predict(
                new_features,
                found_features,
                found_ids,
                new_item['category'],
                found_categories,
                filter_category=True,
                top_k=5
            )
            
            # Add metadata and convert format
            matches = []
            for match in svm_matches:
                found_id = match['found_id']
                if found_id in system['id_to_meta']:
                    match['metadata'] = system['id_to_meta'][found_id]
                    # Ensure consistent keys
                    similarity = match.get('similarity', 0.0)
                    match['combined_similarity'] = similarity
                    match['image_similarity'] = similarity  # SVM doesn't separate these
                    match['text_similarity'] = similarity
                    matches.append(match)
            
            return matches
        
        else:
            st.error(f"Unknown model type: {model_type}")
            return []
            
    except Exception as e:
        st.error(f"Error finding matches: {e}")
        import traceback
        st.error(traceback.format_exc())
        return []


def display_match_card(match, rank, system, show_image=True):
    """Display a match card"""
    match_id = match.get('found_id') or match.get('lost_id')
    similarity = match.get('combined_similarity', match.get('similarity', 0.0))
    project_root = system['project_root']
    
    # Get confidence level
    if similarity >= 0.7:
        confidence = "High"
        css_class = "high-confidence"
    elif similarity >= 0.5:
        confidence = "Medium"
        css_class = "medium-confidence"
    else:
        confidence = "Low"
        css_class = "low-confidence"
    
    # Get metadata
    meta = match.get('metadata', {})
    
    with st.container():
        st.markdown(f'<div class="match-card {css_class}">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            st.markdown(f"### Rank {rank}")
            st.markdown(f'<div class="similarity-score">{similarity:.1%}</div>', unsafe_allow_html=True)
            st.caption(f"{confidence} Confidence")
        
        with col2:
            st.markdown(f"**Item ID:** {match_id}")
            st.markdown(f"**Category:** {meta.get('category', 'Unknown')}")
            
            if meta.get('description'):
                # Truncate long descriptions
                desc = meta['description']
                if len(desc) > 150:
                    st.markdown(f"**Description:** {desc[:150]}...")
                else:
                    st.markdown(f"**Description:** {desc}")
            
            if meta.get('timestamp'):
                st.markdown(f"**Date:** {meta['timestamp'][:10]}")
            
            # Show similarity breakdown
            with st.expander("📊 Similarity Breakdown"):
                if 'image_similarity' in match:
                    st.write(f"Image Similarity: {match['image_similarity']:.1%}")
                if 'text_similarity' in match:
                    st.write(f"Text Similarity: {match['text_similarity']:.1%}")
                if 'image_similarity' not in match and 'text_similarity' not in match:
                    st.write(f"Overall Similarity: {similarity:.1%}")
        
        with col3:
            # Show image if available and enabled
            if show_image and meta.get('filename') and meta.get('category'):
                img_path = project_root / 'dataset' / 'images' / meta['category'] / meta['filename']
                if img_path.exists():
                    try:
                        # Use smaller width and catch any image loading errors
                        st.image(str(img_path), width=120)
                    except Exception as e:
                        st.caption("🖼️ Image preview unavailable")
            elif not show_image:
                st.caption("🖼️ Click to view image")
        
        st.markdown('</div>', unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Campus Lost & Found</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load system
    with st.spinner("Loading matching system..."):
        system = load_system()
    
    if system is None:
        st.error("Failed to load the matching system. Please check the setup.")
        st.stop()
    
    st.success("✓ System loaded successfully!")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Check if navigation was triggered from home page buttons
    if 'navigate_to' in st.session_state and st.session_state.navigate_to:
        default_page = st.session_state.navigate_to
        st.session_state.navigate_to = None  # Reset after using
    else:
        default_page = "🏠 Home"
    
    # Get index of default page
    pages = ["🏠 Home", "😢 Report Lost Item", "🎉 Report Found Item", "🔎 Browse Existing Matches", "📊 Statistics"]
    default_index = pages.index(default_page) if default_page in pages else 0
    
    page = st.sidebar.radio(
        "Choose an action:",
        pages,
        index=default_index
    )
    
    # Model selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Model Settings")
    
    # Check which models are available
    available_models = ["Similarity-based"]
    if system.get('knn_matcher') is not None:
        available_models.append("KNN")
    if system.get('svm_matcher') is not None:
        available_models.append("SVM")
    
    model_choice = st.sidebar.selectbox(
        "Matching Algorithm:",
        available_models,
        help="""
        • Similarity-based: Fast, weighted image+text features
        • KNN: Nearest neighbors in feature space
        • SVM: Trained classifier for match prediction
        """
    )
    
    # Map to internal model type
    model_type_map = {
        "Similarity-based": "similarity",
        "KNN": "knn",
        "SVM": "svm"
    }
    model_type = model_type_map[model_choice]
    
    # Store in session state
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = model_type
    else:
        st.session_state.selected_model = model_type
    
    # Show model info
    with st.sidebar.expander("ℹ️ About Models"):
        if model_choice == "Similarity-based":
            st.write("**Cosine similarity** on combined image and text features.")
            st.write("✅ Fast")
            st.write("✅ Interpretable")
            st.write("✅ No training required")
        elif model_choice == "KNN":
            st.write("**K-Nearest Neighbors** finds similar items in feature space.")
            st.write("✅ Simple")
            st.write("✅ Non-parametric")
            st.write("⚠️ Requires training")
        elif model_choice == "SVM":
            st.write("**Support Vector Machine** learns to classify matches.")
            st.write("✅ Powerful")
            st.write("✅ Probability scores")
            st.write("⚠️ Requires labeled data")
    
    # Training status
    if len(available_models) < 3:
        st.sidebar.info(f"💡 Train additional models to unlock more options. Currently available: {len(available_models)}/3")
    
    # Home Page
    if page == "🏠 Home":
        st.header("Welcome to Campus Lost & Found!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("😢 Lost Something?")
            st.write("""
            1. Upload a photo of your lost item
            2. Provide a description
            3. Our AI will find potential matches
            4. Browse through suggested items
            """)
            if st.button("Report Lost Item", use_container_width=True, key="home_lost_btn"):
                st.session_state.navigate_to = "😢 Report Lost Item"
                st.rerun()
        
        with col2:
            st.subheader("🎉 Found Something?")
            st.write("""
            1. Upload a photo of the found item
            2. Describe what you found
            3. Help us match it to owners
            4. Make someone's day!
            """)
            if st.button("Report Found Item", use_container_width=True, key="home_found_btn"):
                st.session_state.navigate_to = "🎉 Report Found Item"
                st.rerun()
        
        st.markdown("---")
        
        # Statistics
        st.subheader("📈 Current Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            lost_count = len([f for f in system['all_features'] if f['status'] == 'lost'])
            st.metric("Lost Items", lost_count)
        
        with col2:
            found_count = len([f for f in system['all_features'] if f['status'] == 'found'])
            st.metric("Found Items", found_count)
        
        with col3:
            categories_count = len(CATEGORIES)
            st.metric("Categories", categories_count)
    
    # Report Lost Item
    elif page == "😢 Report Lost Item":
        st.header("Report a Lost Item")
        
        with st.form("lost_item_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                uploaded_image = st.file_uploader(
                    "Upload an image of your lost item",
                    type=['jpg', 'jpeg', 'png'],
                    help="Upload a clear photo of the item you lost"
                )
                
                if uploaded_image:
                    image = Image.open(uploaded_image)
                    st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                category = st.selectbox(
                    "Category",
                    CATEGORIES,
                    help="Select the category that best matches your item"
                )
                
                description = st.text_area(
                    "Description",
                    placeholder="Describe your item: color, brand, distinguishing features, where you lost it...",
                    height=150,
                    help="The more details you provide, the better we can match!"
                )
                
                contact_info = st.text_input(
                    "Contact Information (Optional)",
                    placeholder="Email or phone number"
                )
            
            submitted = st.form_submit_button("🔍 Find Matches", use_container_width=True)
        
        if submitted:
            if not uploaded_image:
                st.error("Please upload an image of your lost item.")
            elif not description:
                st.error("Please provide a description of your lost item.")
            else:
                with st.spinner("Extracting features and finding matches..."):
                    # Extract features
                    image = Image.open(uploaded_image)
                    new_item = extract_features_from_upload(image, description, category, system)
                    
                    if new_item:
                        # Find matches using selected model
                        matches = find_matches(new_item, system, status='lost', model_type=st.session_state.selected_model)
                        
                        st.session_state.current_matches = matches
                        
                        # Display results
                        st.success(f"✓ Found {len(matches)} potential matches using **{model_choice}** algorithm!")
                        
                        if matches:
                            st.markdown("---")
                            st.subheader("🎯 Potential Matches")
                            st.write("Here are items that might match what you lost:")
                            
                            for i, match in enumerate(matches, 1):
                                display_match_card(match, i, system)
                        else:
                            st.info("No matches found yet. We'll keep looking as new items are reported!")
    
    # Report Found Item
    elif page == "🎉 Report Found Item":
        st.header("Report a Found Item")
        
        with st.form("found_item_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                uploaded_image = st.file_uploader(
                    "Upload an image of the found item",
                    type=['jpg', 'jpeg', 'png'],
                    help="Upload a clear photo of the item you found"
                )
                
                if uploaded_image:
                    image = Image.open(uploaded_image)
                    st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                category = st.selectbox(
                    "Category",
                    CATEGORIES,
                    help="Select the category that best matches the item"
                )
                
                description = st.text_area(
                    "Description",
                    placeholder="Describe the item: color, brand, condition, where you found it...",
                    height=150,
                    help="Provide details to help match with the owner"
                )
                
                location = st.text_input(
                    "Where did you find it?",
                    placeholder="Building, room, or location"
                )
            
            submitted = st.form_submit_button("🔍 Find Potential Owners", use_container_width=True)
        
        if submitted:
            if not uploaded_image:
                st.error("Please upload an image of the found item.")
            elif not description:
                st.error("Please provide a description of the found item.")
            else:
                with st.spinner("Extracting features and finding matches..."):
                    # Extract features
                    image = Image.open(uploaded_image)
                    new_item = extract_features_from_upload(image, description, category, system)
                    
                    if new_item:
                        # Find matches using selected model
                        matches = find_matches(new_item, system, status='found', model_type=st.session_state.selected_model)
                        
                        st.session_state.current_matches = matches
                        
                        # Display results
                        st.success(f"✓ Found {len(matches)} potential owners using **{model_choice}** algorithm!")
                        
                        if matches:
                            st.markdown("---")
                            st.subheader("🎯 Potential Owners")
                            st.write("These people might be looking for this item:")
                            
                            for i, match in enumerate(matches, 1):
                                display_match_card(match, i, system)
                        else:
                            st.info("No owners found yet. The item will be available for matching as people report lost items!")
    
    # Browse Existing Matches
    elif page == "🔎 Browse Existing Matches":
        st.header("Browse Existing Matches")
        
        # Performance tip
        st.info("💡 **Performance Tip:** Uncheck 'Show images' for faster browsing with large datasets. Use pagination to view items in smaller batches.")
        
        # Get project root
        project_root = system['project_root']
        
        # Model selection for browsing
        st.subheader("⚙️ Select Matching Algorithm")
        col_model, col_info = st.columns([2, 3])
        
        with col_model:
            # Check which models are available
            browse_models = ["Precomputed (Similarity-based)"]
            if system.get('knn_matcher') is not None:
                browse_models.append("Recompute with KNN")
            if system.get('svm_matcher') is not None:
                browse_models.append("Recompute with SVM")
            
            browse_model_choice = st.selectbox(
                "View matches using:",
                browse_models,
                help="Precomputed: Fast, uses saved matches. Recompute: Recalculates matches with selected model."
            )
        
        with col_info:
            if browse_model_choice == "Precomputed (Similarity-based)":
                st.info("📊 Showing pre-computed matches from the matching pipeline (fastest)")
            elif "KNN" in browse_model_choice:
                st.warning("🔄 Will recompute matches using KNN (may take a few seconds)")
            elif "SVM" in browse_model_choice:
                st.warning("🔄 Will recompute matches using SVM (may take a few seconds)")
        
        # Determine which matching to use
        use_precomputed = browse_model_choice == "Precomputed (Similarity-based)"
        recompute_model = None
        if "KNN" in browse_model_choice:
            recompute_model = "knn"
        elif "SVM" in browse_model_choice:
            recompute_model = "svm"
        
        # Load existing matches or prepare for recomputation
        matches_file = project_root / 'features' / 'matches' / 'matches.json'
        if matches_file.exists() and use_precomputed:
            # Load precomputed matches
            with open(matches_file, 'r') as f:
                all_matches = json.load(f)
            
            # Filter options
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                filter_category = st.selectbox(
                    "Filter by category",
                    ["All"] + CATEGORIES
                )
            
            with col2:
                min_similarity = st.slider(
                    "Minimum similarity",
                    0.0, 1.0, 0.3, 0.05
                )
            
            with col3:
                items_per_page = st.selectbox(
                    "Items per page",
                    [5, 10, 20, 50],
                    index=1  # Default to 10
                )
            
            with col4:
                show_images = st.checkbox(
                    "Show images",
                    value=False,
                    help="Uncheck for faster loading"
                )
            
            st.markdown("---")
            
            # Collect all filtered items first
            filtered_items = []
            for lost_id, item_matches in all_matches.items():
                # Get lost item metadata
                if lost_id not in system['id_to_meta']:
                    continue
                
                lost_meta = system['id_to_meta'][lost_id]
                
                # Apply filters
                if filter_category != "All" and lost_meta['category'] != filter_category:
                    continue
                
                # Filter by similarity
                filtered_matches = [m for m in item_matches if m['similarity'] >= min_similarity]
                
                if not filtered_matches:
                    continue
                
                filtered_items.append({
                    'lost_id': lost_id,
                    'lost_meta': lost_meta,
                    'matches': filtered_matches
                })
            
            # Pagination
            if filtered_items:
                total_items = len(filtered_items)
                total_pages = (total_items + items_per_page - 1) // items_per_page
                
                # Initialize session state for page number
                if 'browse_page' not in st.session_state:
                    st.session_state.browse_page = 1
                
                # Pagination controls at top
                col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
                
                with col1:
                    st.write(f"Showing **{total_items}** items ({total_pages} pages)")
                
                with col2:
                    if st.button("⬅️ Previous", disabled=(st.session_state.browse_page == 1)):
                        st.session_state.browse_page -= 1
                        st.rerun()
                
                with col3:
                    if st.button("Next ➡️", disabled=(st.session_state.browse_page >= total_pages)):
                        st.session_state.browse_page += 1
                        st.rerun()
                
                with col4:
                    # Page selector
                    selected_page = st.selectbox(
                        "Go to page:",
                        range(1, total_pages + 1),
                        index=st.session_state.browse_page - 1,
                        label_visibility="collapsed"
                    )
                    if selected_page != st.session_state.browse_page:
                        st.session_state.browse_page = selected_page
                        st.rerun()
                
                st.markdown("---")
                
                # Calculate slice for current page
                start_idx = (st.session_state.browse_page - 1) * items_per_page
                end_idx = min(start_idx + items_per_page, total_items)
                current_page_items = filtered_items[start_idx:end_idx]
                
                # Display items for current page
                for idx, item in enumerate(current_page_items, start=start_idx + 1):
                    lost_id = item['lost_id']
                    lost_meta = item['lost_meta']
                    filtered_matches = item['matches']
                    
                    # Display lost item info
                    st.subheader(f"{idx}. Lost Item: {lost_id}")
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        if show_images:
                            img_path = project_root / 'dataset' / 'images' / lost_meta['category'] / lost_meta['filename']
                            if img_path.exists():
                                try:
                                    st.image(str(img_path), width=150)
                                except:
                                    st.warning("⚠️ Image error")
                        else:
                            st.info(f"🖼️ Image\n({lost_meta['category']})")
                    
                    with col2:
                        st.write(f"**Category:** {lost_meta['category']}")
                        if lost_meta.get('description'):
                            st.write(f"**Description:** {lost_meta['description'][:100]}...")
                        st.write(f"**Matches found:** {len(filtered_matches)}")
                    
                    # Display matches in expander
                    with st.expander(f"🔍 View {len(filtered_matches)} matches", expanded=False):
                        for i, match in enumerate(filtered_matches, 1):
                            found_id = match['found_id']
                            if found_id in system['id_to_meta']:
                                # Create match card data
                                match_card = {
                                    'found_id': found_id,
                                    'similarity': match['similarity'],
                                    'combined_similarity': match['similarity'],
                                    'rank': match.get('rank', i),
                                    'metadata': system['id_to_meta'][found_id]
                                }
                                display_match_card(match_card, i, system, show_image=show_images)
                    
                    st.markdown("---")
                
                # Pagination controls at bottom
                st.markdown("###")
                col1, col2, col3 = st.columns([2, 2, 2])
                
                with col1:
                    if st.button("⬅️ Previous Page", key="prev_bottom", disabled=(st.session_state.browse_page == 1)):
                        st.session_state.browse_page -= 1
                        st.rerun()
                
                with col2:
                    st.markdown(f"<div style='text-align: center'>Page {st.session_state.browse_page} of {total_pages}</div>", unsafe_allow_html=True)
                
                with col3:
                    if st.button("Next Page ➡️", key="next_bottom", disabled=(st.session_state.browse_page >= total_pages)):
                        st.session_state.browse_page += 1
                        st.rerun()
            
            else:
                st.info("No matches found with the current filters.")
        
        elif recompute_model:
            # Recompute matches using selected model
            st.info(f"🔄 Recomputing matches using **{browse_model_choice}**...")
            
            # Filter options (same as above but before computation)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                filter_category = st.selectbox(
                    "Filter by category",
                    ["All"] + CATEGORIES
                )
            
            with col2:
                min_similarity = st.slider(
                    "Minimum similarity",
                    0.0, 1.0, 0.3, 0.05
                )
            
            with col3:
                items_per_page = st.selectbox(
                    "Items per page",
                    [5, 10, 20, 50],
                    index=1
                )
            
            with col4:
                show_images = st.checkbox(
                    "Show images",
                    value=False,
                    help="Uncheck for faster loading"
                )
            
            st.markdown("---")
            
            # Get lost items features
            lost_features_list = [f for f in system['all_features'] if f['status'] == 'lost']
            found_features_list = [f for f in system['all_features'] if f['status'] == 'found']
            
            if filter_category != "All":
                lost_features_list = [f for f in lost_features_list if f['category'] == filter_category]
            
            # Recompute matches
            filtered_items = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, lost_item in enumerate(lost_features_list):
                status_text.text(f"Computing matches for item {idx+1}/{len(lost_features_list)}...")
                progress_bar.progress((idx + 1) / len(lost_features_list))
                
                lost_id = lost_item['id']
                lost_meta = system['id_to_meta'].get(lost_id, {})
                
                # Compute matches using selected model
                if recompute_model == "knn":
                    # KNN matching
                    new_features = np.concatenate([lost_item['image_features'], lost_item['text_features']])
                    
                    matches = system['knn_matcher'].predict(
                        new_features,
                        lost_item['category'],
                        filter_category=True
                    )
                    
                    # Add metadata and format
                    formatted_matches = []
                    for match in matches:
                        found_id = match['found_id']
                        if found_id in system['id_to_meta']:
                            formatted_match = {
                                'rank': match['rank'],
                                'found_id': found_id,
                                'similarity': match['similarity']
                            }
                            if formatted_match['similarity'] >= min_similarity:
                                formatted_matches.append(formatted_match)
                    
                elif recompute_model == "svm":
                    # SVM matching
                    new_features = np.concatenate([lost_item['image_features'], lost_item['text_features']])
                    
                    found_features = system['feature_matrices']['found']['combined_features']
                    found_ids = system['feature_matrices']['found']['ids']
                    found_categories = system['feature_matrices']['found']['categories']
                    
                    matches = system['svm_matcher'].predict(
                        new_features,
                        found_features,
                        found_ids,
                        lost_item['category'],
                        found_categories,
                        filter_category=True,
                        top_k=10
                    )
                    
                    # Format matches
                    formatted_matches = []
                    for match in matches:
                        formatted_match = {
                            'rank': match['rank'],
                            'found_id': match['found_id'],
                            'similarity': match['similarity']
                        }
                        if formatted_match['similarity'] >= min_similarity:
                            formatted_matches.append(formatted_match)
                
                if formatted_matches:
                    filtered_items.append({
                        'lost_id': lost_id,
                        'lost_meta': lost_meta,
                        'matches': formatted_matches
                    })
            
            progress_bar.empty()
            status_text.empty()
            st.success(f"✓ Computed matches for {len(filtered_items)} lost items")
            
            # Pagination (same structure as precomputed)
            if filtered_items:
                total_items = len(filtered_items)
                total_pages = (total_items + items_per_page - 1) // items_per_page
                
                # Initialize session state for page number
                if 'browse_page' not in st.session_state:
                    st.session_state.browse_page = 1
                
                # Pagination controls at top
                col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
                
                with col1:
                    st.write(f"Showing **{total_items}** items ({total_pages} pages)")
                
                with col2:
                    if st.button("⬅️ Previous", key="recomp_prev_top", disabled=(st.session_state.browse_page == 1)):
                        st.session_state.browse_page -= 1
                        st.rerun()
                
                with col3:
                    if st.button("Next ➡️", key="recomp_next_top", disabled=(st.session_state.browse_page >= total_pages)):
                        st.session_state.browse_page += 1
                        st.rerun()
                
                with col4:
                    # Page selector
                    selected_page = st.selectbox(
                        "Go to page:",
                        range(1, total_pages + 1),
                        index=st.session_state.browse_page - 1,
                        label_visibility="collapsed",
                        key="recomp_page_select"
                    )
                    if selected_page != st.session_state.browse_page:
                        st.session_state.browse_page = selected_page
                        st.rerun()
                
                st.markdown("---")
                
                # Calculate slice for current page
                start_idx = (st.session_state.browse_page - 1) * items_per_page
                end_idx = min(start_idx + items_per_page, total_items)
                current_page_items = filtered_items[start_idx:end_idx]
                
                # Display items for current page
                for idx, item in enumerate(current_page_items, start=start_idx + 1):
                    lost_id = item['lost_id']
                    lost_meta = item['lost_meta']
                    filtered_matches = item['matches']
                    
                    # Display lost item info
                    st.subheader(f"{idx}. Lost Item: {lost_id}")
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        if show_images:
                            img_path = project_root / 'dataset' / 'images' / lost_meta['category'] / lost_meta['filename']
                            if img_path.exists():
                                try:
                                    st.image(str(img_path), width=150)
                                except:
                                    st.warning("⚠️ Image error")
                        else:
                            st.info(f"🖼️ Image\n({lost_meta['category']})")
                    
                    with col2:
                        st.write(f"**Category:** {lost_meta['category']}")
                        if lost_meta.get('description'):
                            st.write(f"**Description:** {lost_meta['description'][:100]}...")
                        st.write(f"**Matches found:** {len(filtered_matches)}")
                    
                    # Display matches in expander
                    with st.expander(f"🔍 View {len(filtered_matches)} matches", expanded=False):
                        for i, match in enumerate(filtered_matches, 1):
                            found_id = match['found_id']
                            if found_id in system['id_to_meta']:
                                # Create match card data
                                match_card = {
                                    'found_id': found_id,
                                    'similarity': match['similarity'],
                                    'combined_similarity': match['similarity'],
                                    'rank': match.get('rank', i),
                                    'metadata': system['id_to_meta'][found_id]
                                }
                                display_match_card(match_card, i, system, show_image=show_images)
                    
                    st.markdown("---")
                
                # Pagination controls at bottom
                st.markdown("###")
                col1, col2, col3 = st.columns([2, 2, 2])
                
                with col1:
                    if st.button("⬅️ Previous Page", key="recomp_prev_bottom", disabled=(st.session_state.browse_page == 1)):
                        st.session_state.browse_page -= 1
                        st.rerun()
                
                with col2:
                    st.markdown(f"<div style='text-align: center'>Page {st.session_state.browse_page} of {total_pages}</div>", unsafe_allow_html=True)
                
                with col3:
                    if st.button("Next Page ➡️", key="recomp_next_bottom", disabled=(st.session_state.browse_page >= total_pages)):
                        st.session_state.browse_page += 1
                        st.rerun()
            
            else:
                st.info("No matches found with the current filters.")
        
        else:
                st.warning("No existing matches found. Please run the matching pipeline first.")
                total_items = len(filtered_items)
                total_pages = (total_items + items_per_page - 1) // items_per_page
                
                # Initialize session state for page number
                if 'browse_page' not in st.session_state:
                    st.session_state.browse_page = 1
                
                # Pagination controls at top
                col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
                
                with col1:
                    st.write(f"Showing **{total_items}** items ({total_pages} pages)")
                
                with col2:
                    if st.button("⬅️ Previous", disabled=(st.session_state.browse_page == 1)):
                        st.session_state.browse_page -= 1
                        st.rerun()
                
                with col3:
                    if st.button("Next ➡️", disabled=(st.session_state.browse_page >= total_pages)):
                        st.session_state.browse_page += 1
                        st.rerun()
                
                with col4:
                    # Page selector
                    selected_page = st.selectbox(
                        "Go to page:",
                        range(1, total_pages + 1),
                        index=st.session_state.browse_page - 1,
                        label_visibility="collapsed"
                    )
                    if selected_page != st.session_state.browse_page:
                        st.session_state.browse_page = selected_page
                        st.rerun()
                
                st.markdown("---")
                
                # Calculate slice for current page
                start_idx = (st.session_state.browse_page - 1) * items_per_page
                end_idx = min(start_idx + items_per_page, total_items)
                current_page_items = filtered_items[start_idx:end_idx]
                
                # Display items for current page
                for idx, item in enumerate(current_page_items, start=start_idx + 1):
                    lost_id = item['lost_id']
                    lost_meta = item['lost_meta']
                    filtered_matches = item['matches']
                    
                    # Display lost item info
                    st.subheader(f"{idx}. Lost Item: {lost_id}")
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        if show_images:
                            img_path = project_root / 'dataset' / 'images' / lost_meta['category'] / lost_meta['filename']
                            if img_path.exists():
                                try:
                                    st.image(str(img_path), width=150)
                                except:
                                    st.warning("⚠️ Image error")
                        else:
                            st.info(f"🖼️ Image\n({lost_meta['category']})")
                    
                    with col2:
                        st.write(f"**Category:** {lost_meta['category']}")
                        if lost_meta.get('description'):
                            st.write(f"**Description:** {lost_meta['description'][:100]}...")
                        st.write(f"**Matches found:** {len(filtered_matches)}")
                    
                    # Display matches in expander
                    with st.expander(f"🔍 View {len(filtered_matches)} matches", expanded=False):
                        for i, match in enumerate(filtered_matches, 1):
                            found_id = match['found_id']
                            if found_id in system['id_to_meta']:
                                # Create match card data
                                match_card = {
                                    'found_id': found_id,
                                    'similarity': match['similarity'],
                                    'combined_similarity': match['similarity'],
                                    'rank': match.get('rank', i),
                                    'metadata': system['id_to_meta'][found_id]
                                }
                                display_match_card(match_card, i, system, show_image=show_images)
                    
                    st.markdown("---")
                
                # Pagination controls at bottom
                st.markdown("###")
                col1, col2, col3 = st.columns([2, 2, 2])
                
                with col1:
                    if st.button("⬅️ Previous Page", key="prev_bottom", disabled=(st.session_state.browse_page == 1)):
                        st.session_state.browse_page -= 1
                        st.rerun()
                
                with col2:
                    st.markdown(f"<div style='text-align: center'>Page {st.session_state.browse_page} of {total_pages}</div>", unsafe_allow_html=True)
                
                with col3:
                    if st.button("Next Page ➡️", key="next_bottom", disabled=(st.session_state.browse_page >= total_pages)):
                        st.session_state.browse_page += 1
                        st.rerun()
            
    
    # Statistics
    elif page == "📊 Statistics":
        st.header("System Statistics")
        
        # Overall stats
        st.subheader("📈 Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_items = len(system['all_features'])
            st.metric("Total Items", total_items)
        
        with col2:
            lost_count = len([f for f in system['all_features'] if f['status'] == 'lost'])
            st.metric("Lost Items", lost_count)
        
        with col3:
            found_count = len([f for f in system['all_features'] if f['status'] == 'found'])
            st.metric("Found Items", found_count)
        
        with col4:
            with_desc = len([f for f in system['all_features'] if f['has_description']])
            st.metric("With Descriptions", f"{with_desc/total_items*100:.0f}%")
        
        st.markdown("---")
        
        # Category breakdown
        st.subheader("📦 Items by Category")
        
        category_data = {}
        for feature in system['all_features']:
            cat = feature['category']
            status = feature['status']
            
            if cat not in category_data:
                category_data[cat] = {'lost': 0, 'found': 0}
            
            category_data[cat][status] += 1
        
        # Create DataFrame
        df = pd.DataFrame(category_data).T
        df['total'] = df['lost'] + df['found']
        df = df.sort_values('total', ascending=False)
        
        st.bar_chart(df[['lost', 'found']])
        st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()