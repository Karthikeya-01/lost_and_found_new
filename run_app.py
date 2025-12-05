import sys
from pathlib import Path
import os

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set PYTHONPATH environment variable so subprocess inherits it
env = os.environ.copy()
pythonpath = str(project_root)
if 'PYTHONPATH' in env:
    env['PYTHONPATH'] = pythonpath + os.pathsep + env['PYTHONPATH']
else:
    env['PYTHONPATH'] = pythonpath

# Now run streamlit with the modified environment
import subprocess
subprocess.run([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"], env=env)
