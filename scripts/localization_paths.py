"""Resolve the localization workspace and its adjacent SLAM workspace."""
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SLAM_ROOT = Path(os.environ.get('SLAM_LIG_ROOT', ROOT.parent / 'slam_lig')).expanduser().resolve()
