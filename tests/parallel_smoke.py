"""Manual Windows smoke test for the two-process CLI path."""

from __future__ import annotations

import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import tifffile


def zero_shift(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    output = np.zeros_like(image)
    height, width = image.shape
    sy0, sy1 = max(0, -dy), min(height, height - dy)
    sx0, sx1 = max(0, -dx), min(width, width - dx)
    if sy0 < sy1 and sx0 < sx1:
        dy0, dx0 = max(0, dy), max(0, dx)
        output[dy0 : dy0 + sy1 - sy0, dx0 : dx0 + sx1 - sx0] = image[
            sy0:sy1, sx0:sx1
        ]
    return output


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    run_root = project / f".parallel_smoke_{uuid.uuid4().hex}"
    input_root = run_root / "input"
    output_root = run_root / "output"
    try:
        rng = np.random.default_rng(17)
        base_b = rng.integers(0, 4096, (48, 48), dtype=np.uint16)
        base_a = np.zeros((48, 48), dtype=np.uint16)
        base_a[15:18, 23:27] = 2500
        for specimen_name in ("Specimen001", "Specimen002"):
            folder = input_root / "1606" / "glu0" / specimen_name
            folder.mkdir(parents=True)
            (folder / "Experiment.xml").write_text(
                '<ThorImageExperiment><ZStage steps="3" '
                'stepSizeUM="-0.3" /></ThorImageExperiment>',
                encoding="utf-8",
            )
            for index, (dy, dx) in enumerate(((0, 0), (1, -2), (3, -1)), 1):
                suffix = f"_0001_0001_{index:04d}_0001.tif"
                tifffile.imwrite(folder / f"ChanA{suffix}", zero_shift(base_a, dy, dx))
                tifffile.imwrite(folder / f"ChanB{suffix}", zero_shift(base_b, dy, dx))

        completed = subprocess.run(
            [
                sys.executable,
                str(project / "register_tiff_stacks.py"),
                str(input_root),
                str(output_root),
                "--workers",
                "2",
                "--imagej-memory",
                "1g",
            ],
            cwd=project,
            text=True,
            capture_output=True,
            check=False,
        )
        print(completed.stdout, end="")
        print(completed.stderr, end="", file=sys.stderr)
        if completed.returncode != 0:
            return completed.returncode

        tiffs = sorted(output_root.glob("*_registered.tif"))
        csvs = sorted((output_root / "shifts").glob("*_shifts.csv"))
        if len(tiffs) != 4 or len(csvs) != 2:
            print(
                f"unexpected outputs: {len(tiffs)} TIFFs, {len(csvs)} CSVs",
                file=sys.stderr,
            )
            return 1
        for path in tiffs:
            with tifffile.TiffFile(path) as tif:
                if (
                    not tif.is_bigtiff
                    or tif.series[0].axes != "ZYX"
                    or tif.series[0].shape != (3, 48, 48)
                    or len(tif.pages) != 3
                ):
                    print(f"invalid output: {path}", file=sys.stderr)
                    return 1
        print("Parallel smoke test passed.")
        return 0
    finally:
        shutil.rmtree(run_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
