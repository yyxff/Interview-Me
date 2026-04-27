"""
Common path and sys.path setup for all eval scripts.

Add this at the top of any eval script:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
SHARED_DIR = EXPERIMENTS_DIR / "shared"

SCORE_NODE_DIR = EXPERIMENTS_DIR / "score_node"
AGENT_EVAL_DIR = EXPERIMENTS_DIR / "agent_eval"
