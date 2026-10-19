"""Expose the standalone CPU runtime safety suite to standard test discovery."""
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
from test_launch_runtime import OwnedCheckpointTests, RuntimePatchTests  # noqa: E402,F401
