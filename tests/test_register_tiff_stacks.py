from __future__ import annotations

import csv
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import tifffile

import register_tiff_stacks as registration


def zero_shift(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    output = np.zeros_like(image)
    height, width = image.shape
    sy0, sy1 = max(0, -dy), min(height, height - dy)
    sx0, sx1 = max(0, -dx), min(width, width - dx)
    if sy0 < sy1 and sx0 < sx1:
        dy0, dx0 = max(0, dy), max(0, dx)
        output[dy0 : dy0 + sy1 - sy0, dx0 : dx0 + sx1 - sx0] = image[sy0:sy1, sx0:sx1]
    return output


class RegistrationTests(unittest.TestCase):
    def test_natural_key(self) -> None:
        names = ["slice10", "slice2", "slice1"]
        self.assertEqual(sorted(names, key=registration.natural_key), ["slice1", "slice2", "slice10"])

    def test_plugin_end_to_end_writes_true_zyx_stacks(self) -> None:
        project = Path(__file__).resolve().parents[1]
        root = project / f".test_runtime_{uuid.uuid4().hex}"
        specimen_dir = root / "1606" / "glu0" / "Untitled002"
        output_dir = root / "output"
        try:
            specimen_dir.mkdir(parents=True)
            (output_dir / registration.CSV_DIR_NAME).mkdir(parents=True)
            rng = np.random.default_rng(42)
            base_b = rng.integers(0, 4096, (64, 64), dtype=np.uint16)
            base_a = np.zeros((64, 64), dtype=np.uint16)
            base_a[20:24, 30:35] = 3000
            drifts = [(0, 0), (2, -3), (5, -4), (4, 1)]
            for index, (dy, dx) in enumerate(drifts, start=1):
                suffix = f"_0001_0001_{index:04d}_0001.tif"
                tifffile.imwrite(specimen_dir / f"ChanA{suffix}", zero_shift(base_a, dy, dx), resolution=(100, 100), resolutionunit="CENTIMETER")
                tifffile.imwrite(specimen_dir / f"ChanB{suffix}", zero_shift(base_b, dy, dx), resolution=(100, 100), resolutionunit="CENTIMETER")
            (specimen_dir / "Experiment.xml").write_text('<ThorImageExperiment><ZStage steps="4" stepSizeUM="-0.3" /></ThorImageExperiment>', encoding="utf-8")

            specimens, issues = registration.discover_specimens(root)
            self.assertEqual(issues, [])
            self.assertEqual(len(specimens), 1)
            config = registration.resolve_imagej_config(project / "ImageJ", "1g")
            result = registration.process_specimen(specimens[0], output_dir, False, config)
            self.assertEqual(result.status, "completed", result.detail)

            for channel in ("ChanA", "ChanB"):
                path = output_dir / f"1606_glu0_Untitled002_{channel}_registered.tif"
                with tifffile.TiffFile(path) as tif:
                    self.assertTrue(tif.is_bigtiff)
                    self.assertEqual(tif.series[0].axes, "ZYX")
                    self.assertEqual(tif.series[0].shape, (4, 64, 64))
                    self.assertEqual(len(tif.pages), 4)
                    self.assertIsNotNone(tif.ome_metadata)
                    self.assertTrue(all(page.compression.name == "NONE" for page in tif.pages))
                self.assertEqual(tifffile.imread(path).dtype, np.uint16)

            stack_a = tifffile.imread(output_dir / "1606_glu0_Untitled002_ChanA_registered.tif")
            peak_positions = [tuple(np.unravel_index(np.argmax(plane), plane.shape)) for plane in stack_a]
            self.assertTrue(all(position == peak_positions[0] for position in peak_positions), peak_positions)

            csv_path = output_dir / registration.CSV_DIR_NAME / "1606_glu0_Untitled002_shifts.csv"
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["engine"].startswith("ImageJ MultiStackReg") for row in rows))
            expected = [(0, 0), (3, -2), (4, -5), (-1, -4)]
            actual = [(float(row["cumulative_dx"]), float(row["cumulative_dy"])) for row in rows]
            np.testing.assert_allclose(actual, expected, atol=0.05)

            skipped = registration.process_specimen(specimens[0], output_dir, False, config)
            self.assertEqual(skipped.status, "skipped")
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
