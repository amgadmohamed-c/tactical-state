"""Command-line entry point for the Tactical State feature dataset."""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

# Allow the repository-root script to work before the package is installed.
_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_root.parent))
if "Transitions" not in sys.modules:
    package = types.ModuleType("Transitions")
    package.__path__ = [str(_root)]
    sys.modules["Transitions"] = package

from Transitions.analytics.tactical_state import build_dataset
from Transitions.io.paths import EPV_GRID_PATH, PROCESSED_DIR, PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one feature vector per possession sequence.")
    parser.add_argument("match_id", nargs="?", help="Optional processed match ID for debugging.")
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "tactical_state_features.parquet"))
    parser.add_argument("--epv-grid", default=str(EPV_GRID_PATH))
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.processed_dir, args.output, args.match_id, args.epv_grid), indent=2))


if __name__ == "__main__":
    main()
