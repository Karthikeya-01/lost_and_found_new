import sys
from pathlib import Path

print(f"Current working directory: {Path.cwd()}")
print(f"Script location: {Path(__file__).parent}")

# Add project root to path (same as streamlit_app.py)
project_root = Path(__file__).parent.parent
print(f"Project root: {project_root}")
print(f"Project root exists: {project_root.exists()}")
print(f"Project root absolute: {project_root.absolute()}")

sys.path.insert(0, str(project_root))
print(f"sys.path[0]: {sys.path[0]}")

try:
    from src.matching.ranking import MatchRanker
    from src.matching.similarity import SimilarityComputer
    from src.feature_extraction.text_features import TextFeatureExtractor
    from src.feature_extraction.image_features import ImageFeatureExtractor
    print("SUCCESS: All imports successful!")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
