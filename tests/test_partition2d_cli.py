import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from figures import partition2d


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
