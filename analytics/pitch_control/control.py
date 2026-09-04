"""Pitch control computation for tracking data (Voronoi fallback)."""

import json
import bz2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from ...core.logger import get_logger
from ...io.paths import match_dir
from ...io.field_keys import PLAYER_ID_KEYS, PLAYER_X_KEYS, PLAYER_Y_KEYS

logger = get_logger(__name__)

# We'll use scipy for Voronoi if available
try:
    from scipy.spatial import Voronoi, voronoi_plot_2d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("scipy not available – Voronoi fallback disabled.")


def _get_first(d, keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default

def _extract_player_xy(player_dict):
    x = _get_first(player_dict, PLAYER_X_KEYS)
    y = _get_first(player_dict, PLAYER_Y_KEYS)
    pid = _get_first(player_dict, PLAYER_ID_KEYS)
    if x is None or y is None or pid is None:
        return None
    return pid, float(x), float(y)

def compute_pitch_control_frame(home_positions, away_positions, ball_xy,
                                pitch_length, pitch_width,
                                home_vel=None, away_vel=None,
                                grid_resolution=50):
    """
    Compute home/away control using Voronoi tessellation over a grid.
    Ignores velocities – a simple spatial control.
    """
    if not HAS_SCIPY:
        raise RuntimeError("scipy is required for Voronoi fallback. Install scipy.")
    
    # Combine all players and label them
    all_points = []
    labels = []
    for (x, y) in home_positions:
        if 0 <= x <= pitch_length and 0 <= y <= pitch_width:
            all_points.append([x, y])
            labels.append('home')
    for (x, y) in away_positions:
        if 0 <= x <= pitch_length and 0 <= y <= pitch_width:
            all_points.append([x, y])
            labels.append('away')
    
    # If not enough points, return equal control (or skip)
    if len(all_points) < 4:
        return {"home_control": 0.5, "away_control": 0.5, "grid": None}
    
    # Create grid of points covering the pitch
    xs = np.linspace(0, pitch_length, grid_resolution)
    ys = np.linspace(0, pitch_width, grid_resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    
    try:
        vor = Voronoi(all_points)
    except Exception as e:
        logger.warning("Voronoi failed: %s – falling back to equal control", e)
        return {"home_control": 0.5, "away_control": 0.5, "grid": None}
    
    # For each grid point, find the nearest player (by brute force – fine for 50x50)
    try:
        from scipy.spatial.distance import cdist
        dists = cdist(grid_points, all_points)
        nearest_idx = np.argmin(dists, axis=1)
    except ImportError:
        nearest_idx = np.zeros(len(grid_points), dtype=int)
        for i, pt in enumerate(grid_points):
            min_dist = np.inf
            for j, player in enumerate(all_points):
                d = np.hypot(pt[0]-player[0], pt[1]-player[1])
                if d < min_dist:
                    min_dist = d
                    nearest_idx[i] = j
    
    # Count cells per team
    home_count = 0
    away_count = 0
    for idx in nearest_idx:
        if labels[idx] == 'home':
            home_count += 1
        else:
            away_count += 1
    total = home_count + away_count
    if total == 0:
        home_frac = 0.5
        away_frac = 0.5
    else:
        home_frac = home_count / total
        away_frac = away_count / total
    
    return {"home_control": home_frac, "away_control": away_frac, "grid": None}

def compute_pitch_control_for_match(match_id, processed_dir, downsample=1, force_recompute=False):
    """Compute pitch control for every frame (or downsampled) using Voronoi fallback.

    Reuses a cached CSV if it already exists unless a fresh recomputation is forced.
    """
    match_dir_path = match_dir(match_id, processed_dir)
    cache_path = match_dir_path / "pitch_control_frames.csv"
    if cache_path.is_file() and not force_recompute:
        logger.info("Loading cached pitch control data from %s", cache_path)
        return pd.read_csv(cache_path)

    metadata_path = match_dir_path / "metadata.json"
    tracking_path = match_dir_path / "tracking.jsonl.bz2"
    
    if not metadata_path.is_file() or not tracking_path.is_file():
        raise FileNotFoundError(f"Processed files missing for {match_id} in {match_dir_path}")
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    pitch_length = metadata["pitch"]["length"]
    pitch_width = metadata["pitch"]["width"]
    x_shift = pitch_length / 2.0
    y_shift = pitch_width / 2.0
    
    records = []
    frame_count = 0
    
    try:
        with bz2.open(tracking_path, "rt") as f:
            for line in f:
                frame = json.loads(line)
                frame_count += 1
                if frame_count % downsample != 0:
                    continue
            
                period = frame.get("period")
                elapsed = frame.get("periodElapsedTime")
                if period is None or elapsed is None:
                    continue

                home_positions = []
                for p in frame.get("homePlayers", []):
                    parsed = _extract_player_xy(p)
                    if parsed is not None:
                        _, x, y = parsed
                        home_positions.append((x + x_shift, y + y_shift))
                away_positions = []
                for p in frame.get("awayPlayers", []):
                    parsed = _extract_player_xy(p)
                    if parsed is not None:
                        _, x, y = parsed
                        away_positions.append((x + x_shift, y + y_shift))

                balls = frame.get("balls", [])
                if not balls:
                    continue
                ball = balls[0]
                bx = ball.get("x")
                by = ball.get("y")
                if bx is None or by is None:
                    continue
                bx += x_shift
                by += y_shift

                result = compute_pitch_control_frame(
                    home_positions, away_positions, (bx, by),
                    pitch_length, pitch_width,
                )
                records.append({
                    "period": period,
                    "elapsed": elapsed,
                    "home_control": result["home_control"],
                    "away_control": result["away_control"],
                })
    except EOFError:
        logger.warning(
            "%s is truncated; retaining Pitch Control for %d decoded frames.",
            tracking_path,
            frame_count,
        )
    
    df = pd.DataFrame(records)
    if not df.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        logger.info("Saved pitch control cache to %s", cache_path)
    logger.info("Computed pitch control for %d frames", len(df))
    return df
    


def aggregate_pitch_control_by_window(control_df, formations_df):
    """Merge control frames with formation windows."""
    results = []
    for _, row in formations_df.iterrows():
        period = row["period"]
        start = row["windowStartSec"]
        end = row["windowEndSec"]
        mask = (control_df["period"] == period) & (control_df["elapsed"] >= start) & (control_df["elapsed"] < end)
        sub = control_df[mask]
        if len(sub) == 0:
            home_mean = away_mean = np.nan
        else:
            home_mean = sub["home_control"].mean()
            away_mean = sub["away_control"].mean()
        out = row.to_dict()
        out["mean_home_control"] = home_mean
        out["mean_away_control"] = away_mean
        results.append(out)
    return pd.DataFrame(results)