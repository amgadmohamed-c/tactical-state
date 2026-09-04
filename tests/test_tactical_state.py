from __future__ import annotations

import unittest

import numpy as np

from Transitions.analytics.tactical_state.pipeline import _frame_features, _safe_stats


class TacticalStateFeatureTests(unittest.TestCase):
    def test_sequence_statistics_have_explicit_delta(self) -> None:
        result = _safe_stats([2.0, 4.0, 6.0], "width")
        self.assertEqual(result["width_start"], 2.0)
        self.assertEqual(result["width_end"], 6.0)
        self.assertEqual(result["width_mean"], 4.0)
        self.assertEqual(result["width_delta"], 4.0)

    def test_orientation_mirrors_progress_to_positive_x(self) -> None:
        home = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
        away = np.array([[70.0, 10.0], [80.0, 20.0], [90.0, 30.0]])
        ball = np.array([20.0, 20.0])
        normal = _frame_features(home, away, ball, "home", 1, 100.0, 60.0)
        flipped = _frame_features(home, away, ball, "home", -1, 100.0, 60.0)
        self.assertAlmostEqual(normal["ball_x"], 20.0)
        self.assertAlmostEqual(flipped["ball_x"], 80.0)
        self.assertAlmostEqual(normal["team_width"], flipped["team_width"])


if __name__ == "__main__":
    unittest.main()
