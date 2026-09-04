# Tactical State dataset and training lock

## What was implemented

The first research pipeline is available through:

```bash
python run_tactical_pipeline.py \
  --processed-dir /path/to/processed_tracking
```

It processes every match folder, or one match when a match ID is supplied. The
pipeline reuses the existing tracking loader, nearest-player possession
inference, possession smoothing/forward filling, attack-direction helpers,
formation detector, Pitch Control implementation, and OBSO implementation.
It creates one row per valid contiguous possession sequence and writes the
combined result under `outputs/`.

The detailed feature definitions are in
[tactical-state-feature-manifest.md](./tactical-state-feature-manifest.md).

## Row contract

Every row represents exactly one possession:

- `match_id`, `team_id`, `team`, `period`
- `possession_id`
- `start_frame`, `end_frame`
- `start_time`, `end_time`, `duration`, `n_frames`
- `number_of_passes`, `ball_distance_travelled`
- `formation`
- spatial, structure, shape, ball-relationship, numerical, space, and
  dynamics features
- separate `pitch_control_*` and `obso_*` outcome columns

`possession_id` is deterministic for a run: match ID, period, and sequence
number. It is unique within the generated dataset.

## Training lock: freeze before modeling

The generated file must be treated as a **versioned dataset snapshot**, not as
a live table. Before training:

1. Run preprocessing and the Tactical State pipeline.
2. Confirm the final summary reports the expected match and possession counts.
3. Inspect the output for duplicate `possession_id` values, missing required
   identifiers, invalid durations, and unexpected feature types.
4. Record the exact input directory, code commit, command, Python/dependency
   versions, output path, row count, column count, and failed/discarded
   matches.
5. Copy the completed output to an immutable, versioned location, for example:

   ```text
   datasets/tactical_state/v1/tactical_state_features.parquet
   datasets/tactical_state/v1/manifest.json
   ```

6. Compute and record a SHA-256 checksum:

   ```bash
   shasum -a 256 datasets/tactical_state/v1/tactical_state_features.parquet
   ```

Do not overwrite a locked snapshot. Any change to source data, preprocessing,
possession thresholds, feature code, aggregation, or dependencies creates a
new dataset version (`v2`, `v3`, etc.) and requires rerunning validation.

## Model-input lock and leakage rules

The autoencoder/representation model should use only tactical input columns.
Exclude:

- identifiers: `match_id`, `team_id`, `team`, `period`, `possession_id`
- timing/index columns: `start_frame`, `end_frame`, `start_time`, `end_time`
- outcome columns: every `pitch_control_*` and `obso_*` column
- administrative fields such as failure flags or source paths

Pitch Control and OBSO are retained for later evaluation of discovered states.
They must not be used to fit normalization, the autoencoder, or clustering if
the research question is whether tactical states predict those outcomes.

Fit normalization parameters on the training split only, then apply the frozen
parameters to validation/test data. Split by match (or by match and team), not
randomly by row, so possessions from the same match cannot appear in both
training and evaluation.

## Missing values and corrupted inputs

Missing player coordinates can produce NaN feature values; degenerate convex
hulls produce zero area. These cases must be counted and handled by the
training code using a documented imputation or row-filtering policy.

Truncated tracking streams are logged and decoded frames are retained where
possible. Such matches must be marked in the dataset manifest and should
normally be excluded from the primary benchmark or analyzed in a sensitivity
run. Never silently treat a partial match as complete.

## Required pre-training checks

At minimum, verify:

```python
assert df["possession_id"].is_unique
assert (df["duration"] >= 0).all()
assert (df["n_frames"] > 0).all()
assert df[["match_id", "team_id", "possession_id"]].notna().all().all()
```

Also report the number of matches, possessions, features, missing values per
feature, and possessions per match/team. Keep the locked dataset and its
manifest together so a trained model can always be traced back to its exact
research input.

## Current limitations

- Pass counts depend on the available normalized event labels; tracking-only
  matches may have zero counted passes.
- The existing Pitch Control implementation is a Voronoi spatial fallback,
  not a velocity-aware probabilistic model.
- OBSO depends on the configured EPV grid and radius.
- Formation is assigned from the matching formation window containing the
  possession start time.
- A missing Parquet engine causes the CLI to write CSV instead; install a
  Parquet engine and create a new locked snapshot when scale requires Parquet.
