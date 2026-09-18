#!/usr/bin/env python3
"""Register paired TIFF planes or paired, preassembled TIFF stacks.

By default, ChanB (or r in preassembled mode) is aligned sequentially with
translation-only StackReg/TurboReg. The exact same transformations are applied
to the other channel by MultiStackReg. Output can be OME-BigTIFF or classic
ImageJ TIFF, with one uncompressed ZYX series.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import traceback
import uuid
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import numpy as np
    import tifffile
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"Missing dependency: {exc.name}. Install with "
        "'python -m pip install -r requirements.txt'."
    ) from exc


TIFF_SUFFIXES = {".tif", ".tiff"}
CHANNEL_RE = re.compile(r"^(ChanA|ChanB)(?P<key>.*)$", re.IGNORECASE)
STACK_CHANNEL_RE = re.compile(r"^(?P<prefix>.+)(?P<channel>[gr])$", re.IGNORECASE)
NATURAL_PART_RE = re.compile(r"(\d+)")
MEMORY_RE = re.compile(r"^[1-9]\d*[kKmMgG]$")
LOG_NAME = "registration.log"
CSV_DIR_NAME = "shifts"
WORK_DIR_NAME = ".stackreg_work"


@dataclass(frozen=True)
class ImagePair:
    key: str
    chan_a: Path
    chan_b: Path


@dataclass(frozen=True)
class Specimen:
    folder: Path
    output_prefix: str
    pairs: tuple[ImagePair, ...]
    prestacked: bool = False
    relative_output_dir: Path = Path()
    channel_labels: tuple[str, str] = ("ChanA", "ChanB")


@dataclass(frozen=True)
class SourceMetadata:
    shape: tuple[int, int]
    dtype: np.dtype[Any]
    resolution: tuple[float, float] | None
    resolution_unit: str | None
    pixel_size_um: tuple[float, float] | None
    z_spacing_um: float | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ImageJConfig:
    java: Path
    ij_jar: Path
    plugins_dir: Path
    macro: Path
    memory: str


@dataclass(frozen=True)
class ShiftRecord:
    slice_number: int
    chan_a_name: str
    chan_b_name: str
    incremental_dx: float
    incremental_dy: float
    cumulative_dx: float
    cumulative_dy: float
    source_x: float
    source_y: float
    target_x: float
    target_y: float


@dataclass(frozen=True)
class ProcessResult:
    prefix: str
    status: str
    detail: str


def natural_key(value: str) -> tuple[tuple[int, int | str], ...]:
    parts: list[tuple[int, int | str]] = []
    for part in NATURAL_PART_RE.split(value.casefold()):
        if part:
            parts.append((0, int(part)) if part.isdigit() else (1, part))
    return tuple(parts)


def _channel_and_key(path: Path) -> tuple[str, str] | None:
    if path.suffix.casefold() not in TIFF_SUFFIXES:
        return None
    match = CHANNEL_RE.match(path.stem)
    if match is None:
        return None
    return match.group(1).casefold(), match.group("key").casefold()


def pair_images(folder: Path) -> tuple[ImagePair, ...]:
    by_channel: dict[str, dict[str, Path]] = {"chana": {}, "chanb": {}}
    for path in folder.iterdir():
        if not path.is_file() or (parsed := _channel_and_key(path)) is None:
            continue
        channel, key = parsed
        if key in by_channel[channel]:
            raise ValueError(f"duplicate {channel} slice key {key!r}")
        by_channel[channel][key] = path
    a_keys, b_keys = set(by_channel["chana"]), set(by_channel["chanb"])
    if not a_keys and not b_keys:
        return ()
    if a_keys != b_keys:
        details = []
        if b_keys - a_keys:
            details.append(f"missing ChanA keys: {sorted(b_keys-a_keys, key=natural_key)[:10]}")
        if a_keys - b_keys:
            details.append(f"missing ChanB keys: {sorted(a_keys-b_keys, key=natural_key)[:10]}")
        raise ValueError("unpaired channel files; " + "; ".join(details))
    return tuple(
        ImagePair(key, by_channel["chana"][key], by_channel["chanb"][key])
        for key in sorted(a_keys, key=natural_key)
    )


def discover_specimens(input_path: Path) -> tuple[list[Specimen], list[str]]:
    root = input_path.resolve()
    if not root.is_dir():
        raise ValueError(f"input path is not a directory: {root}")
    candidates: list[Path] = []
    for current, dirs, files in os.walk(root):
        dirs.sort(key=natural_key)
        if any(_channel_and_key(Path(name)) is not None for name in files):
            candidates.append(Path(current))
    specimens: list[Specimen] = []
    issues: list[str] = []
    for folder in sorted(candidates, key=lambda p: natural_key(str(p))):
        if len(folder.parts) < 3:
            issues.append(f"{folder}: cannot derive day/group/specimen name")
            continue
        try:
            pairs = pair_images(folder)
        except Exception as exc:
            issues.append(f"{folder}: {exc}")
            continue
        if pairs:
            specimens.append(Specimen(folder, "_".join(folder.parts[-3:]), pairs))
    seen: dict[str, list[Path]] = {}
    for item in specimens:
        seen.setdefault(item.output_prefix.casefold(), []).append(item.folder)
    rejected = {p for paths in seen.values() if len(paths) > 1 for p in paths}
    for paths in seen.values():
        if len(paths) > 1:
            issues.append("output-name collision for folders: " + ", ".join(map(str, paths)))
    return [s for s in specimens if s.folder not in rejected], issues


def _stack_layout(path: Path) -> tuple[int, tuple[int, int], np.dtype[Any]]:
    with tifffile.TiffFile(path) as tif:
        if len(tif.series) != 1:
            raise ValueError(f"expected one TIFF series, found {len(tif.series)}")
        series = tif.series[0]
        if series.axes != "ZYX" or len(series.shape) != 3:
            raise ValueError(f"expected a grayscale ZYX stack, got axes={series.axes} shape={series.shape}")
        if len(series.pages) != series.shape[0] or series.shape[0] < 2:
            raise ValueError(f"expected at least two physical Z pages, got shape={series.shape} pages={len(series.pages)}")
        if np.dtype(series.dtype).kind not in "buif":
            raise ValueError(f"unsupported pixel dtype {series.dtype}")
        return int(series.shape[0]), (int(series.shape[1]), int(series.shape[2])), np.dtype(series.dtype)


def discover_prestacked(input_path: Path) -> tuple[list[Specimen], list[str]]:
    """Find same-folder TIFF pairs whose final stem character is g/r."""
    root = input_path.resolve()
    if not root.is_dir():
        raise ValueError(f"input path is not a directory: {root}")
    specimens: list[Specimen] = []
    issues: list[str] = []
    for current, dirs, files in os.walk(root):
        dirs.sort(key=natural_key)
        folder = Path(current)
        grouped: dict[tuple[str, str], dict[str, Path]] = {}
        display_prefixes: dict[tuple[str, str], str] = {}
        for name in sorted(files, key=natural_key):
            path = folder / name
            if path.suffix.casefold() not in TIFF_SUFFIXES:
                continue
            match = STACK_CHANNEL_RE.fullmatch(path.stem)
            if match is None:
                continue
            try:
                z_count, _, _ = _stack_layout(path)
            except Exception as exc:
                # Single-plane TIFFs are intentionally outside this mode. Other
                # malformed candidates are reported when they look like stacks.
                try:
                    with tifffile.TiffFile(path) as tif:
                        if len(tif.pages) <= 1:
                            continue
                except Exception:
                    pass
                issues.append(f"{path}: {exc}")
                continue
            if z_count < 2:  # defensive; _stack_layout already enforces this
                continue
            prefix = match.group("prefix")
            key = (prefix.casefold(), path.suffix.casefold())
            channel = match.group("channel").casefold()
            if channel in grouped.setdefault(key, {}):
                issues.append(f"{folder}: duplicate {channel} stack for {prefix!r}{path.suffix}")
                continue
            grouped[key][channel] = path
            display_prefixes.setdefault(key, prefix)

        folder_specimens: list[Specimen] = []
        for key in sorted(grouped, key=lambda item: natural_key(item[0] + item[1])):
            channels = grouped[key]
            prefix = display_prefixes[key]
            if set(channels) != {"g", "r"}:
                missing = "r" if "g" in channels else "g"
                issues.append(f"{folder}: incomplete stack pair {prefix!r}; missing {missing} channel")
                continue
            g_path, r_path = channels["g"], channels["r"]
            try:
                g_layout, r_layout = _stack_layout(g_path), _stack_layout(r_path)
                if g_layout != r_layout:
                    raise ValueError(
                        f"g/r stacks differ: {g_path.name} has {g_layout[0]}x{g_layout[1][0]}x{g_layout[1][1]}/{g_layout[2]}, "
                        f"{r_path.name} has {r_layout[0]}x{r_layout[1][0]}x{r_layout[1][1]}/{r_layout[2]}"
                    )
            except Exception as exc:
                issues.append(f"{folder}: invalid stack pair {prefix!r}: {exc}")
                continue
            pairs = tuple(ImagePair(str(index), g_path, r_path) for index in range(1, g_layout[0] + 1))
            folder_specimens.append(
                Specimen(folder, prefix, pairs, True, folder.relative_to(root), ("g", "r"))
            )

        by_output = {}
        for specimen in folder_specimens:
            by_output.setdefault(specimen.output_prefix.casefold(), []).append(specimen)
        for same_name in by_output.values():
            if len(same_name) > 1:
                issues.append(
                    f"{folder}: output-name collision for stack pairs: "
                    + ", ".join(item.output_prefix for item in same_name)
                )
            else:
                specimens.extend(same_name)
    return specimens, issues


def read_plane(path: Path) -> np.ndarray[Any, Any]:
    # Read physical page 0 explicitly: microscope OME metadata can refer to all
    # acquisition files and make tifffile.imread(path) expand one source image.
    with tifffile.TiffFile(path) as tif:
        if len(tif.pages) != 1:
            raise ValueError(f"source must contain one TIFF page: {path} has {len(tif.pages)}")
        image = np.asarray(tif.pages[0].asarray())
    if image.ndim != 2:
        raise ValueError(f"expected one 2D plane in {path}, got {image.shape}")
    if image.dtype.kind not in "buif":
        raise ValueError(f"unsupported pixel dtype {image.dtype} in {path}")
    return image


def _rational_float(value: Any) -> float:
    return float(value[0]) / float(value[1]) if isinstance(value, tuple) else float(value)


def _xy_resolution(path: Path) -> tuple[tuple[float, float] | None, str | None, tuple[float, float] | None]:
    with tifffile.TiffFile(path) as tif:
        tags = tif.pages[0].tags
        if "XResolution" not in tags or "YResolution" not in tags:
            return None, None, None
        x, y = _rational_float(tags["XResolution"].value), _rational_float(tags["YResolution"].value)
        unit = {1: "NONE", 2: "INCH", 3: "CENTIMETER"}.get(int(tags.get("ResolutionUnit").value if tags.get("ResolutionUnit") else 1))
        physical = None
        if x > 0 and y > 0 and unit == "INCH":
            physical = 25400.0 / x, 25400.0 / y
        elif x > 0 and y > 0 and unit == "CENTIMETER":
            physical = 10000.0 / x, 10000.0 / y
        return ((x, y) if x > 0 and y > 0 else None), unit, physical


def _z_spacing(folder: Path) -> tuple[float | None, str | None]:
    candidates = [p for p in folder.iterdir() if p.is_file() and p.name.casefold() == "experiment.xml"]
    if not candidates:
        return None, "Experiment.xml is missing; Z calibration was omitted"
    try:
        node = ET.parse(candidates[0]).getroot().find(".//ZStage")
        spacing = abs(float(node.attrib["stepSizeUM"])) if node is not None else math.nan
        if not math.isfinite(spacing) or spacing <= 0:
            raise ValueError("invalid ZStage stepSizeUM")
        return spacing, None
    except (ET.ParseError, OSError, KeyError, ValueError) as exc:
        return None, f"could not read Z calibration from Experiment.xml: {exc}"


def _distance_um(value: Any, unit: Any) -> float | None:
    try:
        number = abs(float(value))
    except (TypeError, ValueError):
        return None
    normalized = str(unit or "um").strip().casefold().replace("μ", "µ")
    factors = {
        "nm": 0.001,
        "um": 1.0,
        "µm": 1.0,
        "micron": 1.0,
        "microns": 1.0,
        "mm": 1000.0,
        "cm": 10000.0,
        "m": 1_000_000.0,
        "inch": 25400.0,
        "in": 25400.0,
    }
    factor = factors.get(normalized)
    if factor is None or not math.isfinite(number) or number <= 0:
        return None
    return number * factor


def _embedded_z_spacing(path: Path) -> tuple[float | None, str | None]:
    with tifffile.TiffFile(path) as tif:
        if tif.ome_metadata:
            try:
                root = ET.fromstring(tif.ome_metadata)
                pixels = next(node for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "Pixels")
                spacing = _distance_um(
                    pixels.attrib.get("PhysicalSizeZ"), pixels.attrib.get("PhysicalSizeZUnit", "um")
                )
                if spacing is not None:
                    return spacing, None
            except (ET.ParseError, StopIteration):
                pass
        imagej = tif.imagej_metadata or {}
        spacing = _distance_um(imagej.get("spacing"), imagej.get("unit", "um"))
        if spacing is not None:
            return spacing, None
    return None, f"{path.name} has no usable embedded Z calibration"


def inspect_source(specimen: Specimen) -> SourceMetadata:
    if specimen.prestacked:
        g_path, r_path = specimen.pairs[0].chan_a, specimen.pairs[0].chan_b
        g_layout, r_layout = _stack_layout(g_path), _stack_layout(r_path)
        if g_layout != r_layout:
            raise ValueError(f"g/r stacks differ: {g_layout} vs {r_layout}")
        _, shape, dtype = r_layout
        resolution, unit, physical = _xy_resolution(r_path)
        spacing, embedded_warning = _embedded_z_spacing(r_path)
        warnings: list[str] = []
        if spacing is None:
            spacing, xml_warning = _z_spacing(specimen.folder)
            warnings.append(embedded_warning or "embedded Z calibration is missing")
            if xml_warning is not None:
                warnings.append(xml_warning)
        return SourceMetadata(shape, dtype, resolution, unit, physical, spacing, tuple(warnings))
    a, b = read_plane(specimen.pairs[0].chan_a), read_plane(specimen.pairs[0].chan_b)
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError(f"first ChanA/ChanB planes differ: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}")
    resolution, unit, physical = _xy_resolution(specimen.pairs[0].chan_b)
    spacing, warning = _z_spacing(specimen.folder)
    return SourceMetadata(b.shape, b.dtype, resolution, unit, physical, spacing, () if warning is None else (warning,))


def resolve_imagej_config(imagej_dir: Path, memory: str) -> ImageJConfig:
    root = imagej_dir.resolve()
    java = root / "jre" / "bin" / ("java.exe" if os.name == "nt" else "java")
    ij_jar = root / "ij.jar"
    macro = Path(__file__).resolve().with_name("imagej_stackreg_batch.ijm")
    required = {
        "bundled Java": java,
        "ij.jar": ij_jar,
        "batch macro": macro,
        "StackReg plugin": root / "plugins" / "StackReg_.jar",
        "TurboReg plugin": root / "plugins" / "TurboReg_.jar",
        "MultiStackReg plugin": root / "plugins" / "MultiStackRegistration_.jar",
    }
    missing = [f"{label} ({path})" for label, path in required.items() if not path.is_file()]
    if missing:
        raise ValueError("ImageJ installation is incomplete; missing: " + ", ".join(missing))
    if not MEMORY_RE.fullmatch(memory):
        raise ValueError("--imagej-memory must look like 8g, 4096m, or 512k")
    return ImageJConfig(java, ij_jar, root, macro, memory.lower())


def emit(queue: Any, level: int, message: str) -> None:
    if queue is None:
        print(message, flush=True)
    else:
        queue.put((level, message))


def _check_plane(plane: np.ndarray[Any, Any], path: Path, metadata: SourceMetadata) -> np.ndarray[Any, Any]:
    if plane.shape != metadata.shape or plane.dtype != metadata.dtype:
        raise ValueError(f"inconsistent plane {path}: {plane.shape}/{plane.dtype}; expected {metadata.shape}/{metadata.dtype}")
    return plane


def _source_planes(specimen: Specimen, metadata: SourceMetadata, channel: str) -> Iterable[np.ndarray[Any, Any]]:
    if specimen.prestacked:
        path = specimen.pairs[0].chan_a if channel == "ChanA" else specimen.pairs[0].chan_b
        with tifffile.TiffFile(path) as tif:
            for page in tif.series[0].pages:
                yield _check_plane(np.asarray(page.asarray()), path, metadata)
        return
    for pair in specimen.pairs:
        path = pair.chan_a if channel == "ChanA" else pair.chan_b
        yield _check_plane(read_plane(path), path, metadata)


def _imagej_metadata(metadata: SourceMetadata) -> dict[str, Any]:
    result: dict[str, Any] = {"axes": "ZYX"}
    if metadata.z_spacing_um is not None:
        result.update({"spacing": metadata.z_spacing_um, "unit": "um"})
    return result


def write_imagej_input(path: Path, specimen: Specimen, metadata: SourceMetadata, channel: str) -> None:
    kwargs: dict[str, Any] = {}
    if metadata.resolution is not None:
        kwargs["resolution"] = metadata.resolution
    if metadata.resolution_unit is not None:
        kwargs["resolutionunit"] = metadata.resolution_unit
    with tifffile.TiffWriter(path, imagej=True) as writer:
        writer.write(
            _source_planes(specimen, metadata, channel),
            shape=(len(specimen.pairs), *metadata.shape),
            dtype=metadata.dtype,
            photometric="minisblack",
            compression=None,
            contiguous=True,
            metadata=_imagej_metadata(metadata),
            **kwargs,
        )


def run_stackreg(config: ImageJConfig, chan_a: Path, chan_b: Path, out_a: Path, out_b: Path, transforms: Path) -> str:
    paths = [chan_a, chan_b, out_a, out_b, transforms]
    if any("|" in str(path) for path in paths):
        raise ValueError("ImageJ paths cannot contain the '|' character")
    argument = "|".join(path.resolve().as_posix() for path in paths)
    command = [
        str(config.java), f"-Xmx{config.memory}", f"-Dplugins.dir={config.plugins_dir}",
        "-cp", str(config.ij_jar), "ij.ImageJ", "-batch", str(config.macro), argument,
    ]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    output = completed.stdout.strip()
    if completed.returncode != 0:
        raise RuntimeError(f"ImageJ exited with code {completed.returncode}: {output[-4000:]}")
    missing = [str(path) for path in (out_a, out_b, transforms) if not path.is_file()]
    if missing:
        raise RuntimeError(f"ImageJ did not create expected file(s): {', '.join(missing)}; output: {output[-4000:]}")
    return output


def parse_transforms(path: Path, specimen: Specimen) -> list[ShiftRecord]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines()]
    if not lines or lines[0] != "MultiStackReg Transformation File":
        raise ValueError(f"unrecognized MultiStackReg transformation file: {path}")
    blocks = [i for i, line in enumerate(lines) if line == "TRANSLATION"]
    expected = max(0, len(specimen.pairs) - 1)
    if len(blocks) != expected:
        raise ValueError(f"transformation file has {len(blocks)} translations; expected {expected}")
    def source_name(source: Path, page: int) -> str:
        return f"{source.name}#page={page}" if specimen.prestacked else source.name

    records = [ShiftRecord(1, source_name(specimen.pairs[0].chan_a, 1), source_name(specimen.pairs[0].chan_b, 1), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
    cumulative_x = cumulative_y = 0.0
    for slice_number, start in enumerate(blocks, start=2):
        try:
            source_x, source_y = map(float, lines[start + 2].split())
            target_x, target_y = map(float, lines[start + 6].split())
        except (IndexError, ValueError) as exc:
            raise ValueError(f"malformed translation block for slice {slice_number}") from exc
        dx, dy = target_x - source_x, target_y - source_y
        cumulative_x += dx
        cumulative_y += dy
        pair = specimen.pairs[slice_number - 1]
        records.append(ShiftRecord(slice_number, source_name(pair.chan_a, slice_number), source_name(pair.chan_b, slice_number), dx, dy, cumulative_x, cumulative_y, source_x, source_y, target_x, target_y))
    return records


def _ome_metadata(metadata: SourceMetadata) -> dict[str, Any]:
    result: dict[str, Any] = {"axes": "ZYX"}
    if metadata.pixel_size_um is not None:
        result.update({"PhysicalSizeX": metadata.pixel_size_um[0], "PhysicalSizeXUnit": "µm", "PhysicalSizeY": metadata.pixel_size_um[1], "PhysicalSizeYUnit": "µm"})
    if metadata.z_spacing_um is not None:
        result.update({"PhysicalSizeZ": metadata.z_spacing_um, "PhysicalSizeZUnit": "µm"})
    return result


def _plugin_planes(path: Path, expected: int) -> Iterable[np.ndarray[Any, Any]]:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        if len(series.pages) != expected:
            raise ValueError(f"ImageJ output {path} has {len(series.pages)} pages; expected {expected}")
        for page in series.pages:
            yield np.asarray(page.asarray())


def write_final_stack(
    path: Path, plugin_output: Path, metadata: SourceMetadata, z_count: int, imagej_tiff: bool = False
) -> None:
    kwargs: dict[str, Any] = {}
    if metadata.resolution is not None:
        kwargs["resolution"] = metadata.resolution
    if metadata.resolution_unit is not None:
        kwargs["resolutionunit"] = metadata.resolution_unit
    writer_options = {"imagej": True} if imagej_tiff else {"bigtiff": True, "ome": True}
    output_metadata = _imagej_metadata(metadata) if imagej_tiff else _ome_metadata(metadata)
    with tifffile.TiffWriter(path, **writer_options) as writer:
        writer.write(
            _plugin_planes(plugin_output, z_count),
            shape=(z_count, *metadata.shape), dtype=metadata.dtype,
            photometric="minisblack", compression=None, contiguous=True,
            metadata=output_metadata, **kwargs,
        )


def validate_stack(
    path: Path, expected_shape: tuple[int, int, int], expected_dtype: np.dtype[Any], imagej_tiff: bool = False
) -> None:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        format_checks = (
            [(not tif.is_bigtiff, "is BigTIFF"), (tif.is_imagej, "is not an ImageJ TIFF")]
            if imagej_tiff
            else [(tif.is_bigtiff, "is not BigTIFF"), (tif.ome_metadata is not None, "has no OME metadata")]
        )
        checks = format_checks + [
            (len(tif.series) == 1, f"has {len(tif.series)} image series"),
            (tuple(series.shape) == expected_shape, f"has shape {series.shape}, expected {expected_shape}"),
            (series.axes == "ZYX", f"has axes {series.axes}, expected ZYX"),
            (np.dtype(series.dtype) == expected_dtype, f"has dtype {series.dtype}, expected {expected_dtype}"),
            (len(tif.pages) == expected_shape[0], f"has {len(tif.pages)} pages, expected {expected_shape[0]}"),
            (all(page.compression == 1 for page in tif.pages), "is compressed"),
        ]
        for okay, detail in checks:
            if not okay:
                raise RuntimeError(f"validation failed: {path} {detail}")


def write_shift_csv(
    path: Path, records: Sequence[ShiftRecord], channel_labels: tuple[str, str] = ("ChanA", "ChanB")
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["slice_number", f"source_{channel_labels[0]}", f"source_{channel_labels[1]}", "incremental_dx", "incremental_dy", "cumulative_dx", "cumulative_dy", "stackreg_source_x", "stackreg_source_y", "stackreg_target_x", "stackreg_target_y", "engine"])
        for r in records:
            writer.writerow([r.slice_number, r.chan_a_name, r.chan_b_name, r.incremental_dx, r.incremental_dy, r.cumulative_dx, r.cumulative_dy, r.source_x, r.source_y, r.target_x, r.target_y, "ImageJ MultiStackReg / StackReg / TurboReg"])


def _output_paths(specimen: Specimen, output_dir: Path) -> tuple[Path, Path, Path]:
    destination = output_dir / specimen.relative_output_dir
    if specimen.prestacked:
        final_a = destination / f"{specimen.output_prefix}{specimen.channel_labels[0]}_registered.tif"
        final_b = destination / f"{specimen.output_prefix}{specimen.channel_labels[1]}_registered.tif"
    else:
        final_a = destination / f"{specimen.output_prefix}_{specimen.channel_labels[0]}_registered.tif"
        final_b = destination / f"{specimen.output_prefix}_{specimen.channel_labels[1]}_registered.tif"
    return final_a, final_b, destination / CSV_DIR_NAME / f"{specimen.output_prefix}_shifts.csv"


def process_specimen(
    specimen: Specimen,
    output_dir: Path,
    overwrite: bool,
    imagej: ImageJConfig,
    log_queue: Any = None,
    align_chan_a: bool = False,
    imagej_tiff: bool = False,
) -> ProcessResult:
    final_a, final_b, final_csv = _output_paths(specimen, output_dir)
    completed = (final_a, final_b, final_csv)
    if not overwrite and all(path.is_file() for path in completed):
        emit(log_queue, logging.INFO, f"[{specimen.output_prefix}] skipped: completed outputs exist")
        return ProcessResult(specimen.output_prefix, "skipped", "all completed outputs already exist")
    partial = tuple(path.with_name(path.name + ".partial") for path in completed)
    work = output_dir / WORK_DIR_NAME / f"{specimen.output_prefix}.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        final_a.parent.mkdir(parents=True, exist_ok=True)
        final_csv.parent.mkdir(parents=True, exist_ok=True)
        for path in partial:
            path.unlink(missing_ok=True)
        work.mkdir(parents=True)
        metadata = inspect_source(specimen)
        for warning in metadata.warnings:
            emit(log_queue, logging.WARNING, f"[{specimen.output_prefix}] {warning}")
        input_a, input_b = work / "ChanA.tif", work / "ChanB.tif"
        plugin_a, plugin_b = work / "ChanA_registered.tif", work / "ChanB_registered.tif"
        transforms = work / "stackreg_transforms.txt"
        if specimen.prestacked:
            input_a, input_b = specimen.pairs[0].chan_a, specimen.pairs[0].chan_b
            emit(log_queue, logging.INFO, f"[{specimen.output_prefix}] using {len(specimen.pairs)}-slice preassembled g/r stacks")
        else:
            emit(log_queue, logging.INFO, f"[{specimen.output_prefix}] assembling {len(specimen.pairs)} slices for ImageJ")
            write_imagej_input(input_a, specimen, metadata, "ChanA")
            write_imagej_input(input_b, specimen, metadata, "ChanB")
        aligned_label = specimen.channel_labels[0] if align_chan_a else specimen.channel_labels[1]
        emit(log_queue, logging.INFO, f"[{specimen.output_prefix}] running ImageJ translation-only MultiStackReg; alignment channel={aligned_label}")
        if align_chan_a:
            imagej_output = run_stackreg(imagej, input_b, input_a, plugin_b, plugin_a, transforms)
        else:
            imagej_output = run_stackreg(imagej, input_a, input_b, plugin_a, plugin_b, transforms)
        if imagej_output:
            emit(log_queue, logging.INFO, f"[{specimen.output_prefix}] ImageJ: {imagej_output[-1000:]}")
        records = parse_transforms(transforms, specimen)
        write_final_stack(partial[0], plugin_a, metadata, len(specimen.pairs), imagej_tiff)
        write_final_stack(partial[1], plugin_b, metadata, len(specimen.pairs), imagej_tiff)
        expected = (len(specimen.pairs), *metadata.shape)
        validate_stack(partial[0], expected, metadata.dtype, imagej_tiff)
        validate_stack(partial[1], expected, metadata.dtype, imagej_tiff)
        write_shift_csv(partial[2], records, specimen.channel_labels)
        os.replace(partial[0], final_a)
        os.replace(partial[1], final_b)
        os.replace(partial[2], final_csv)  # completion marker, promoted last
        detail = f"completed {len(specimen.pairs)} slices; shape={expected}, dtype={metadata.dtype}"
        emit(log_queue, logging.INFO, f"[{specimen.output_prefix}] {detail}")
        return ProcessResult(specimen.output_prefix, "completed", detail)
    except Exception as exc:
        for path in partial:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        detail = f"{type(exc).__name__}: {exc}"
        emit(log_queue, logging.ERROR, f"[{specimen.output_prefix}] failed: {detail}\n{traceback.format_exc()}")
        return ProcessResult(specimen.output_prefix, "failed", detail)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def configure_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("tiff-registration")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def consume_logs(queue: Any, logger: logging.Logger) -> None:
    while (event := queue.get()) is not None:
        logger.log(*event)


def build_parser() -> argparse.ArgumentParser:
    default_imagej = Path(__file__).resolve().with_name("ImageJ")
    parser = argparse.ArgumentParser(description="Build paired stacks and register both channels with translation-only ImageJ MultiStackReg, using ChanB as the alignment channel.")
    parser.add_argument("input_path", type=Path, help="experiment root or specimen folder")
    parser.add_argument("output_dir", type=Path, help="separate output directory")
    parser.add_argument("--workers", type=int, default=2, help="specimen folders processed simultaneously (default: 2)")
    parser.add_argument("--imagej-dir", type=Path, default=default_imagej, help=f"ImageJ directory (default: {default_imagej})")
    parser.add_argument("--imagej-memory", default="8g", help="maximum Java heap per worker (default: 8g)")
    parser.add_argument("--overwrite", action="store_true", help="replace completed outputs")
    parser.add_argument("--dry-run", action="store_true", help="discover and pair inputs only")
    parser.add_argument(
        "--prestacked", action="store_true",
        help="register preassembled stacks paired by a final g/r filename character",
    )
    parser.add_argument(
        "--align-g", action="store_true",
        help="with --prestacked, align g and apply the same transforms to r (default: align r)",
    )
    parser.add_argument(
        "--imagej-tiff", action="store_true",
        help="write classic ImageJ TIFF outputs instead of OME-BigTIFF",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        print("ERROR: --workers must be at least 1", file=sys.stderr)
        return 2
    if args.align_g and not args.prestacked:
        print("ERROR: --align-g requires --prestacked", file=sys.stderr)
        return 2
    try:
        discovery = discover_prestacked if args.prestacked else discover_specimens
        specimens, issues = discovery(args.input_path)
        imagej = None if args.dry_run else resolve_imagej_config(args.imagej_dir, args.imagej_memory)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    discovered_kind = "stack pair(s)" if args.prestacked else "specimen folder(s)"
    print(f"Discovered {len(specimens)} valid {discovered_kind}.")
    for specimen in specimens:
        destination = specimen.relative_output_dir / specimen.output_prefix
        print(f"  {specimen.folder} -> {destination} ({len(specimen.pairs)} paired slices)")
    for issue in issues:
        print(f"WARNING: skipped invalid input: {issue}", file=sys.stderr)
    if args.dry_run:
        return 1 if issues else 0
    if not specimens:
        print("ERROR: no valid specimen folders found", file=sys.stderr)
        return 2
    output = args.output_dir.resolve()
    if output == args.input_path.resolve():
        print("ERROR: output directory must differ from input", file=sys.stderr)
        return 2
    output.mkdir(parents=True, exist_ok=True)
    (output / CSV_DIR_NAME).mkdir(parents=True, exist_ok=True)
    (output / WORK_DIR_NAME).mkdir(parents=True, exist_ok=True)
    logger = configure_logger(output / LOG_NAME)
    logger.info(
        "Run started: input=%s output=%s specimens=%d workers=%d ImageJ=%s memory=%s prestacked=%s alignment_channel=%s output_format=%s",
        args.input_path.resolve(), output, len(specimens), args.workers, args.imagej_dir.resolve(),
        args.imagej_memory, args.prestacked, "g" if args.align_g else ("r" if args.prestacked else "ChanB"),
        "ImageJ TIFF" if args.imagej_tiff else "OME-BigTIFF",
    )
    for issue in issues:
        logger.warning("Skipped invalid input: %s", issue)
    import multiprocessing
    results: list[ProcessResult] = []
    with multiprocessing.Manager() as manager:
        queue = manager.Queue()
        thread = threading.Thread(target=consume_logs, args=(queue, logger), daemon=True)
        thread.start()
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(
                        process_specimen, specimen, output, args.overwrite, imagej, queue,
                        args.align_g, args.imagej_tiff,
                    ): specimen
                    for specimen in specimens
                }
                for future in as_completed(futures):
                    specimen = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        detail = f"worker process crashed: {type(exc).__name__}: {exc}"
                        logger.exception("[%s] %s", specimen.output_prefix, detail)
                        results.append(ProcessResult(specimen.output_prefix, "failed", detail))
        finally:
            queue.put(None)
            thread.join()
    counts = {status: sum(r.status == status for r in results) for status in ("completed", "skipped", "failed")}
    logger.info("Run finished: completed=%d skipped=%d failed=%d discovery_issues=%d", counts["completed"], counts["skipped"], counts["failed"], len(issues))
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    return 1 if counts["failed"] or issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
