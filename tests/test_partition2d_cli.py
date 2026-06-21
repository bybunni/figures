import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from figures import partition2d


class Partition2dRobustnessTests(unittest.TestCase):
    def test_f_is_negative_at_ellipsoid_center(self):
        cx, cy, _, _ = partition2d.ELLIPSOIDS[0]

        self.assertLess(partition2d.f(cx, cy), 0)

    def test_f_is_zero_on_ellipsoid_boundary(self):
        cx, cy, sx, _ = partition2d.ELLIPSOIDS[0]

        self.assertAlmostEqual(partition2d.f(cx + sx, cy), 0.0)

    def test_f_is_positive_outside_ellipsoids(self):
        self.assertGreater(partition2d.f(5.0, 5.0), 0)

    def test_evaluate_returns_classification_and_distance_to_limit(self):
        cx, cy, sx, _ = partition2d.ELLIPSOIDS[0]

        classification, robustness, distance_to_limit = partition2d.evaluate(
            np.array([cx, cx + sx, 5.0]),
            np.array([cy, cy, 5.0]),
        )

        self.assertEqual(classification.shape, (3,))
        self.assertEqual(robustness.shape, (3,))
        self.assertEqual(distance_to_limit.shape, (3,))
        self.assertEqual(classification[0], "negative")
        self.assertEqual(classification[1], "positive")
        self.assertEqual(classification[2], "positive")
        np.testing.assert_allclose(distance_to_limit, np.abs(robustness))

    def test_robustness_sample_weights_favor_zero_limit(self):
        weights = partition2d._robustness_sample_weights(np.array([0.01, 1.0, 10.0]))

        self.assertAlmostEqual(weights.sum(), 1.0)
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])

    def test_region_sample_appends_points_and_robustness_values(self):
        old_rng = partition2d.rng
        partition2d.rng = np.random.default_rng(123)
        try:
            region = partition2d.Region(*partition2d.BOUNDS)
            region.sample(5)
            region.sample(3)
        finally:
            partition2d.rng = old_rng

        self.assertEqual(region.pts.shape, (8, 2))
        self.assertEqual(region.vals.shape, (8,))
        np.testing.assert_allclose(
            region.vals,
            partition2d.f(region.pts[:, 0], region.pts[:, 1]),
        )


class Partition2dCliTests(unittest.TestCase):
    def test_default_output_path_uses_repo_output_directory(self):
        repo_root = Path(__file__).resolve().parents[1]

        self.assertEqual(
            partition2d.default_output_path(),
            repo_root / "output" / "partition_iterations.png",
        )

    def test_default_snapshot_at_stays_within_iterations(self):
        self.assertEqual(partition2d._default_snapshot_at(9), (1, 3, 6, 9))
        self.assertEqual(partition2d._default_snapshot_at(3), (1, 3))
        self.assertEqual(partition2d._default_snapshot_at(1), (1,))

    def test_parse_snapshot_at_deduplicates_and_sorts(self):
        self.assertEqual(partition2d._parse_snapshot_at("3,1,3", 9), (1, 3))

    def test_parse_snapshot_at_rejects_invalid_values(self):
        invalid_values = ("", "1,,3", "0", "10", "nope")

        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    partition2d._parse_snapshot_at(value, 9)

    def test_main_writes_output_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "partition.png"

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = partition2d.main(
                    ["--iterations", "1", "--output", str(output_path)]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.is_file())


if __name__ == "__main__":
    unittest.main()
