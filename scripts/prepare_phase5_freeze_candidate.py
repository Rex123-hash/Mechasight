"""Prepare candidate metadata only; never evaluate or authorize protected work."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment.fusion_freeze import prepare

if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False))
