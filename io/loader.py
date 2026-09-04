"""Simple loading helpers used by the package scaffolding."""

from __future__ import annotations

import bz2
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .field_keys import PLAYER_ID_KEYS as _LOADER_PLAYER_ID_KEYS
from .paths import match_dir
from ..domain import Match
from ..core.logger import get_logger


logger = get_logger(__name__)


def load_json(path: str | Path) -> Any:
    """Load a JSON document from disk."""

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl_bz2(path: str | Path) -> list[Any]:
    """Load a compressed JSONL file into memory."""

    rows: list[Any] = []
    with bz2.open(path, "rt") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def load_processed_match(match_id: str | int, processed_dir: str | Path, include_formations: bool = True) -> Match:
    """Load a processed match bundle into a shared Match container."""

    folder = match_dir(match_id, processed_dir)
    metadata_path = folder / "metadata.json"
    tracking_path = folder / "tracking.jsonl.bz2"
    formations_path = folder / "formations.csv"

    if not metadata_path.exists() or not tracking_path.exists():
        raise FileNotFoundError(
            f"Missing metadata.json / tracking.jsonl.bz2 under {folder}. Run preprocessing first."
        )

    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    pitch_length = metadata["pitch"]["length"]
    pitch_width = metadata["pitch"]["width"]

    home_ids, away_ids = set(), set()
    n_frames = 0
    try:
        with bz2.open(tracking_path, "rt") as handle:
            for line in handle:
                frame = json.loads(line)
                n_frames += 1
                for p in frame.get("homePlayers", []):
                    pid = _get_first(p, _LOADER_PLAYER_ID_KEYS)
                    if pid is not None:
                        home_ids.add(pid)
                for p in frame.get("awayPlayers", []):
                    pid = _get_first(p, _LOADER_PLAYER_ID_KEYS)
                    if pid is not None:
                        away_ids.add(pid)
    except EOFError:
        logger.warning(
            "%s is truncated; loading the %d frames decoded before EOF.",
            tracking_path,
            n_frames,
        )

    def _sort_key(v):
        try:
            return (0, int(v))
        except (TypeError, ValueError):
            return (1, str(v))

    home_ids = sorted(home_ids, key=_sort_key)
    away_ids = sorted(away_ids, key=_sort_key)
    home_idx = {pid: i for i, pid in enumerate(home_ids)}
    away_idx = {pid: i for i, pid in enumerate(away_ids)}

    periods = np.zeros(n_frames, dtype=np.int16)
    elapsed = np.zeros(n_frames, dtype=np.float32)
    home_xy = np.full((n_frames, len(home_ids), 2), np.nan, dtype=np.float32)
    away_xy = np.full((n_frames, len(away_ids), 2), np.nan, dtype=np.float32)
    ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float32)

    try:
        with bz2.open(tracking_path, "rt") as handle:
            for i, line in enumerate(handle):
                frame = json.loads(line)
                periods[i] = frame.get("period", 0) or 0
                elapsed[i] = frame.get("periodElapsedTime", 0.0) or 0.0

                for p in frame.get("homePlayers", []):
                    parsed = _extract_lenient(p)
                    if parsed is None:
                        continue
                    pid, x, y = parsed
                    home_xy[i, home_idx[pid]] = (x + pitch_length / 2, y + pitch_width / 2)

                for p in frame.get("awayPlayers", []):
                    parsed = _extract_lenient(p)
                    if parsed is None:
                        continue
                    pid, x, y = parsed
                    away_xy[i, away_idx[pid]] = (x + pitch_length / 2, y + pitch_width / 2)

                balls = frame.get("balls", [])
                if balls:
                    b = balls[0]
                    bx, by = b.get("x"), b.get("y")
                    if bx is not None and by is not None:
                        ball_xy[i] = (bx + pitch_length / 2, by + pitch_width / 2)
    except EOFError:
        logger.warning(
            "%s ended during the second read; retaining decoded tracking frames.",
            tracking_path,
        )

    goalkeepers = metadata.get("goalkeepers", {})
    formations_df = pd.read_csv(formations_path) if include_formations and formations_path.exists() else None

    return Match(
        metadata=metadata,
        tracking={
            "pitch_length": pitch_length,
            "pitch_width": pitch_width,
            "periods": periods,
            "elapsed": elapsed,
            "home_ids": home_ids,
            "away_ids": away_ids,
            "home_xy": home_xy,
            "away_xy": away_xy,
            "ball_xy": ball_xy,
            "goalkeepers": goalkeepers,
            "formations_df": formations_df,
            "match_dir": str(folder),
        },
    )


def _get_first(d: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in d:
            return d[key]
    return default


def _extract_lenient(player_dict: dict[str, Any]):
    x = _get_first(player_dict, ["x", "X"])
    y = _get_first(player_dict, ["y", "Y"])
    pid = _get_first(player_dict, _LOADER_PLAYER_ID_KEYS)
    if x is None or y is None or pid is None:
        return None
    return pid, float(x), float(y)
