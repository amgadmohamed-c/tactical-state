"""Build one research feature vector per detected possession sequence.

Pitch Control and OBSO are deliberately treated as outcome columns.  Tactical
input columns are calculated from every valid tracking frame in a possession
and are never derived from either outcome.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from ..formations.detector import process_match as detect_formations
from ..obso import compute_obso_for_match
from ..pitch_control import compute_pitch_control_for_match
from ..possession import (
    attack_direction,
    detect_possession_sequences,
    forward_fill_owner,
    get_base_directions,
    infer_fps,
    smooth_owner,
    stream_ball_and_owner,
)
from ...io.loader import load_processed_match
from ...io.paths import EPV_GRID_PATH, match_dir


def _team_id(metadata: dict[str, Any], team: str) -> str:
    value = metadata.get(f"{team}Team", {}).get("id")
    return str(value) if value is not None else team


def _safe_stats(values: list[float], prefix: str) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {f"{prefix}_{s}": float("nan") for s in ("start", "end", "mean", "min", "max", "std", "delta")}
    return {
        f"{prefix}_start": float(a[0]), f"{prefix}_end": float(a[-1]),
        f"{prefix}_mean": float(np.mean(a)), f"{prefix}_min": float(np.min(a)),
        f"{prefix}_max": float(np.max(a)), f"{prefix}_std": float(np.std(a)),
        f"{prefix}_delta": float(a[-1] - a[0]),
    }


def _hull_area(points: np.ndarray) -> float:
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 3:
        return 0.0
    try:
        return float(ConvexHull(points).volume)
    except (ValueError, QhullError):
        return 0.0


try:
    from scipy.spatial import QhullError
except ImportError:  # pragma: no cover - scipy is a declared dependency
    QhullError = ValueError


def _frame_features(home: np.ndarray, away: np.ndarray, ball: np.ndarray,
                    team: str, direction: int, length: float, width: float) -> dict[str, float]:
    own = home if team == "home" else away
    opp = away if team == "home" else home
    own = own[np.all(np.isfinite(own), axis=1)]
    opp = opp[np.all(np.isfinite(opp), axis=1)]
    if not len(own) or not np.all(np.isfinite(ball)):
        return {}
    own = own.copy()
    opp = opp.copy()
    b = ball.copy()
    if direction < 0:
        own[:, 0], opp[:, 0], b[0] = length - own[:, 0], length - opp[:, 0], length - b[0]
    centroid = own.mean(axis=0)
    distances = np.linalg.norm(own - b, axis=1)
    x = own[:, 0]
    ahead = x > b[0]
    # Quantiles are stable with missing players and avoid assuming shirt order.
    q = np.quantile(x, [1 / 3, 2 / 3])
    defensive = own[x <= q[0]]
    midfield = own[(x > q[0]) & (x <= q[1])]
    attacking = own[x > q[1]]
    def cx(group: np.ndarray) -> float:
        return float(group[:, 0].mean()) if len(group) else float("nan")
    hull = _hull_area(own)
    nearest = np.linalg.norm(opp - b, axis=1) if len(opp) else np.array([np.nan])
    return {
        "team_centroid_x": float(centroid[0]), "team_centroid_y": float(centroid[1]),
        "offensive_centroid_x": float(own[own[:, 0] >= centroid[0], 0].mean()) if np.any(own[:, 0] >= centroid[0]) else float("nan"),
        "offensive_centroid_y": float(own[own[:, 0] >= centroid[0], 1].mean()) if np.any(own[:, 0] >= centroid[0]) else float("nan"),
        "defensive_centroid_x": float(own[own[:, 0] < centroid[0], 0].mean()) if np.any(own[:, 0] < centroid[0]) else float("nan"),
        "defensive_centroid_y": float(own[own[:, 0] < centroid[0], 1].mean()) if np.any(own[:, 0] < centroid[0]) else float("nan"),
        "ball_x": float(b[0]), "ball_y": float(b[1]),
        "ball_distance_to_goal": float(length - b[0]),
        "defensive_line_x": cx(defensive), "midfield_line_x": cx(midfield), "attacking_line_x": cx(attacking),
        "defensive_midfield_gap": cx(midfield) - cx(defensive),
        "midfield_attacking_gap": cx(attacking) - cx(midfield),
        "team_width": float(np.ptp(own[:, 1])), "team_depth": float(np.ptp(own[:, 0])),
        "compactness": float(np.mean(np.linalg.norm(own - centroid, axis=1))),
        "convex_hull_area": hull, "occupied_area": hull, "free_space": length * width - hull,
        "space_ahead_of_ball": float(max(length - b[0], 0) * width),
        "mean_player_ball_distance": float(np.mean(distances)),
        "median_player_ball_distance": float(np.median(distances)),
        "min_player_ball_distance": float(np.min(distances)),
        "players_within_5m": float(np.sum(distances <= 5)), "players_within_10m": float(np.sum(distances <= 10)),
        "players_within_20m": float(np.sum(distances <= 20)), "players_ahead_ball": float(np.sum(ahead)),
        "players_behind_ball": float(np.sum(~ahead)), "attackers_near_ball": float(np.sum((distances <= 15) & ahead)),
        "defenders_near_ball": float(np.sum((distances <= 15) & ~ahead)),
        "numerical_superiority": float(np.sum((distances <= 15) & ahead) - np.sum(nearest <= 15)),
        "space_between_lines": float(max(cx(attacking) - cx(defensive), 0)) if len(defensive) and len(attacking) else float("nan"),
        "open_passing_space": float(max(length - b[0], 0) * width - hull),
    }


def _formation_for(formations: pd.DataFrame | None, period: int, second: float, team: str) -> str:
    if formations is None or formations.empty:
        return "unknown"
    m = formations[(formations["period"] == period) & (formations["team"] == team) &
                   (formations["windowStartSec"] <= second) & (formations["windowEndSec"] > second)]
    return str(m.iloc[0]["formation"]) if len(m) else "unknown"


def build_match_dataset(match_id: str | int, processed_dir: str | Path,
                        epv_grid_path: str | Path | None = None,
                        force_recompute: bool = False) -> tuple[pd.DataFrame, dict[str, int]]:
    folder = match_dir(match_id, processed_dir)
    loaded = load_processed_match(match_id, processed_dir)
    metadata = loaded.metadata
    t = loaded.tracking
    periods, elapsed, bx, by, owner = stream_ball_and_owner(
        folder / "tracking.jsonl.bz2", t["pitch_length"], t["pitch_width"]
    )
    if not len(owner):
        return pd.DataFrame(), {"failed": 0, "discarded": 0, "generated": 0}
    fps = float(metadata.get("fps") or infer_fps(elapsed, periods))
    owner = forward_fill_owner(smooth_owner(owner, periods, fps), periods, elapsed)
    sequences = detect_possession_sequences(owner, periods, elapsed, fps)
    home_dir, away_dir = get_base_directions(metadata)
    formations = t.get("formations_df")
    if formations is None and not (folder / "formations.csv").exists():
        detect_formations(str(match_id), processed_dir=str(processed_dir))
        if (folder / "formations.csv").exists():
            formations = pd.read_csv(folder / "formations.csv")
    pc = obso = pd.DataFrame()
    try:
        pc = compute_pitch_control_for_match(match_id, processed_dir, force_recompute=force_recompute)
    except (FileNotFoundError, RuntimeError, OSError):
        pass
    if epv_grid_path is not None or Path(EPV_GRID_PATH).exists():
        try:
            obso = compute_obso_for_match(match_id, processed_dir, epv_grid_path, force_recompute=force_recompute)
        except (FileNotFoundError, RuntimeError, OSError):
            pass
    rows: list[dict[str, Any]] = []
    discarded = 0
    for seq_no, seq in enumerate(sequences, 1):
        if seq["team"] is None:
            discarded += 1
            continue
        start, end = int(seq["start_idx"]), int(seq["end_idx"])
        team = str(seq["team"])
        if start >= len(t["periods"]) or end < start:
            discarded += 1
            continue
        direction = attack_direction(team, int(seq["period"]), home_dir, away_dir)
        frames = []
        frame_times = []
        for i in range(start, min(end + 1, len(t["periods"]))):
            f = _frame_features(t["home_xy"][i], t["away_xy"][i], t["ball_xy"][i],
                                team, direction, t["pitch_length"], t["pitch_width"])
            if f:
                frames.append(f)
                frame_times.append(float(t["elapsed"][i]))
        if not frames:
            discarded += 1
            continue
        row: dict[str, Any] = {
            "match_id": str(match_id), "team_id": _team_id(metadata, team),
            "team": team, "possession_id": f"{match_id}_{seq['period']}_{seq_no}",
            "period": int(seq["period"]), "start_frame": start, "end_frame": end,
            "start_time": float(seq["start_sec"]), "end_time": float(seq["end_sec"]),
            "duration": float(seq["duration"]), "n_frames": len(frames),
            "number_of_passes": 0, "formation": _formation_for(formations, int(seq["period"]), float(seq["start_sec"]), team),
        }
        keys = frames[0].keys()
        for key in keys:
            vals = [f[key] for f in frames]
            if key in {"team_width", "team_depth", "compactness"}:
                row.update(_safe_stats(vals, key))
            else:
                row[key] = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else float("nan")
        ball_path = np.array([[f["ball_x"], f["ball_y"]] for f in frames])
        row["ball_distance_travelled"] = float(np.linalg.norm(np.diff(ball_path, axis=0), axis=1).sum()) if len(ball_path) > 1 else 0.0
        times = np.asarray(frame_times, dtype=float)
        dt = np.diff(times)
        dt[dt <= 0] = np.nan
        centroid_path = np.array([[f["team_centroid_x"], f["team_centroid_y"]] for f in frames])
        row["mean_team_velocity"] = float(np.nanmean(np.linalg.norm(np.diff(centroid_path, axis=0), axis=1) / dt)) if len(dt) else 0.0
        row["mean_ball_velocity"] = float(np.nanmean(np.linalg.norm(np.diff(ball_path, axis=0), axis=1) / dt)) if len(dt) else 0.0
        row["centroid_movement"] = float(np.linalg.norm(centroid_path[-1] - centroid_path[0])) if len(centroid_path) else 0.0
        event_path = folder / "events.json"
        if event_path.exists():
            events = json.loads(event_path.read_text(encoding="utf-8"))
            row["number_of_passes"] = int(sum(
                (str(e.get("teamId")) == row["team_id"] or (e.get("teamId") is None and str(e.get("team", "")).lower() == team))
                and (str(e.get("gameEventType", "")).upper() == "PASS"
                     or str(e.get("possessionEventType", "")).upper() == "PASS")
                and int(e.get("period", -1)) == int(seq["period"])
                and float(e.get("periodElapsedTimeEstimate", -1)) >= float(seq["start_sec"])
                and float(e.get("periodElapsedTimeEstimate", -1)) <= float(seq["end_sec"])
                for e in events
            ))
        mask_pc = (pc.get("period", pd.Series(dtype=int)) == seq["period"]) & (pc.get("elapsed", pd.Series(dtype=float)) >= seq["start_sec"]) & (pc.get("elapsed", pd.Series(dtype=float)) <= seq["end_sec"])
        if len(pc) and mask_pc.any():
            vals = pc.loc[mask_pc, "home_control" if team == "home" else "away_control"].tolist()
            row.update(_safe_stats(vals, "pitch_control"))
        else:
            row.update(_safe_stats([], "pitch_control"))
        mask_ob = (obso.get("period", pd.Series(dtype=int)) == seq["period"]) & (obso.get("elapsed", pd.Series(dtype=float)) >= seq["start_sec"]) & (obso.get("elapsed", pd.Series(dtype=float)) <= seq["end_sec"]) & (obso.get("team", pd.Series(dtype=str)) == team)
        row.update(_safe_stats(obso.loc[mask_ob, "obso"].tolist() if len(obso) else [], "obso"))
        rows.append(row)
    return pd.DataFrame(rows), {"failed": 0, "discarded": discarded, "generated": len(rows)}


def build_dataset(processed_dir: str | Path, output_path: str | Path,
                  match_id: str | int | None = None, epv_grid_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(processed_dir)
    if not root.is_dir():
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        empty = pd.DataFrame()
        try:
            empty.to_parquet(output, index=False)
            actual = output
        except (ImportError, ModuleNotFoundError, ValueError):
            actual = output.with_suffix(".csv")
            empty.to_csv(actual, index=False)
        return {"matches_processed": 0, "possessions_generated": 0, "failed_matches": [],
                "discarded_possessions": 0, "features": 0, "output": str(actual)}
    matches = [str(match_id)] if match_id is not None else sorted(p.name for p in root.iterdir() if p.is_dir())
    all_rows, failures, discarded = [], [], 0
    for mid in matches:
        try:
            df, stats = build_match_dataset(mid, root, epv_grid_path)
            all_rows.append(df)
            discarded += stats["discarded"]
        except (FileNotFoundError, ValueError, OSError) as exc:
            failures.append({"match_id": mid, "error": str(exc)})
    result = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    actual = output
    try:
        result.to_parquet(output, index=False)
    except (ImportError, ModuleNotFoundError, ValueError):
        actual = output.with_suffix(".csv")
        result.to_csv(actual, index=False)
    return {"matches_processed": len(matches) - len(failures), "possessions_generated": len(result),
            "failed_matches": failures, "discarded_possessions": discarded,
            "features": len(result.columns), "output": str(actual)}
