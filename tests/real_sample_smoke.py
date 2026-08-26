"""Run ImageJ registration on the first three real pairs, then clean outputs."""

from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import register_tiff_stacks as registration


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    specimens, issues = registration.discover_specimens(project / "exp")
    if issues or not specimens:
        print(f"Discovery failed: specimens={len(specimens)}, issues={issues}", file=sys.stderr)
        return 1
    source = specimens[0]
    sample = registration.Specimen(source.folder, source.output_prefix, source.pairs[:3])
    output = project / f".real_stackreg_smoke_{uuid.uuid4().hex}"
    try:
        (output / registration.CSV_DIR_NAME).mkdir(parents=True)
        config = registration.resolve_imagej_config(project / "ImageJ", "2g")
        result = registration.process_specimen(sample, output, True, config)
        if result.status != "completed":
            print(result.detail, file=sys.stderr)
            return 1
        for channel in ("ChanA", "ChanB"):
            path = output / f"{sample.output_prefix}_{channel}_registered.tif"
            stack = tifffile.imread(path)
            nonzero = [int(np.count_nonzero(plane)) for plane in stack]
            if stack.shape != (3, 2048, 2048) or any(count == 0 for count in nonzero):
                print(f"Invalid {channel}: shape={stack.shape}, nonzero={nonzero}", file=sys.stderr)
                return 1
            print(f"{channel}: shape={stack.shape}, nonzero pixels per slice={nonzero}")
        print("Real-data ImageJ sample smoke test passed.")
        return 0
    finally:
        shutil.rmtree(output, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
