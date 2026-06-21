import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
        self.assertEqual(
            partition2d.default_gif_output_path(),
            repo_root / "output" / "partition_iterations.gif",
        )
        self.assertEqual(
            partition2d.default_mpg_output_path(),
            repo_root / "output" / "partition_iterations.mpg",
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

    def test_parser_accepts_animation_options(self):
        parser = partition2d._build_parser()
        args = parser.parse_args(
            [
                "--anim",
                "--gif-output",
                "/tmp/partition.gif",
                "--mpg-output",
                "/tmp/partition.mpg",
            ]
        )

        self.assertTrue(args.anim)
        self.assertEqual(args.gif_output, Path("/tmp/partition.gif"))
        self.assertEqual(args.mpg_output, Path("/tmp/partition.mpg"))

    def test_run_trace_records_animation_events(self):
        old_rng = partition2d.rng
        partition2d.rng = np.random.default_rng(7)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = partition2d.run(n_iters=1, snapshot_at=(1,), trace=True)
        finally:
            partition2d.rng = old_rng

        self.assertIsInstance(result, partition2d.RunTrace)
        self.assertIn(1, result.snapshots)
        events = [frame.event for frame in result.frames]
        self.assertIn("start", events)
        self.assertIn("sample", events)
        self.assertIn("classification", events)
        self.assertIn("partition", events)

    def test_run_without_trace_returns_snapshots_only(self):
        old_rng = partition2d.rng
        partition2d.rng = np.random.default_rng(7)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                snapshots = partition2d.run(n_iters=1, snapshot_at=(1,))
        finally:
            partition2d.rng = old_rng

        self.assertIsInstance(snapshots, dict)
        self.assertNotIsInstance(snapshots, partition2d.RunTrace)

    def test_main_with_animation_writes_png_and_calls_animate(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "partition.png"
            gif_path = Path(tmp_dir) / "partition.gif"
            mpg_path = Path(tmp_dir) / "partition.mpg"

            with (
                mock.patch.object(partition2d, "animate") as animate_mock,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                exit_code = partition2d.main(
                    [
                        "--iterations",
                        "1",
                        "--anim",
                        "--output",
                        str(output_path),
                        "--gif-output",
                        str(gif_path),
                        "--mpg-output",
                        str(mpg_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.is_file())
            animate_mock.assert_called_once()
            args, kwargs = animate_mock.call_args
            self.assertGreater(len(args[0]), 0)
            self.assertEqual(kwargs["gif_path"], gif_path)
            self.assertEqual(kwargs["mpg_path"], mpg_path)

    def test_animate_uses_gif_and_mpg_writers(self):
        frame = partition2d.AnimationFrame(
            iteration=0,
            event="start",
            regions=[partition2d.Region(*partition2d.BOUNDS)],
            label="initial partition",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            gif_path = Path(tmp_dir) / "partition.gif"
            mpg_path = Path(tmp_dir) / "partition.mpg"
            anim_mock = mock.Mock()

            with (
                mock.patch.object(
                    partition2d.animation.writers,
                    "is_available",
                    return_value=True,
                ),
                mock.patch.object(
                    partition2d.animation,
                    "FuncAnimation",
                    return_value=anim_mock,
                ),
            ):
                partition2d.animate([frame], gif_path=gif_path, mpg_path=mpg_path)

        self.assertEqual(anim_mock.save.call_count, 2)
        self.assertEqual(anim_mock.save.call_args_list[0].args[0], str(gif_path))
        self.assertEqual(anim_mock.save.call_args_list[1].args[0], str(mpg_path))


if __name__ == "__main__":
    unittest.main()
