import sys
from pathlib import Path

print(f"Current working directory: {Path.cwd()}")
print(f"Script location: {Path(__file__).parent}")

# Add project root to path
project_root = Path(__file__).parent
print(f"Project root: {project_root}")
print(f"Project root exists: {project_root.exists()}")
print(f"Project root absolute: {project_root.absolute()}")

sys.path.insert(0, str(project_root))
print(f"sys.path[0]: {sys.path[0]}")

try:
    from src.matching.ranking import MatchRanker
    print("SUCCESS: MatchRanker imported successfully!")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
