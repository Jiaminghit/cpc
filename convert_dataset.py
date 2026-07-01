#!/usr/bin/env python3
"""Simple dataset-level interface for MCAP conversion.

This wrapper keeps the actual conversion logic in the existing modules:

* util/mcap_to_pcd.py
* util/mcap_to_jpg.py
* util/mcap_to_struct.py
* util/convert_calibration_to_newstruct.py

Typical usage:
    conda run -n mcap_convert python -m data_convert.convert_dataset \
        /home/c64508/桌面/dataset/2069758074335653889 \
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

try:
    from .util import convert_calibration_to_newstruct as newstruct
    from .util import mcap_to_jpg, mcap_to_pcd, mcap_to_struct
except ImportError:  # Allow: python convert_dataset.py
    from util import convert_calibration_to_newstruct as newstruct
    from util import mcap_to_jpg, mcap_to_pcd, mcap_to_struct


logger = logging.getLogger("convert_dataset")

DEFAULT_STEPS = ("pcd", "jpg", "struct", "newstruct")
DEFAULT_CAMERA_TOPICS = (
    "/sensor/camera_front_wide_image",
    "/sensor/camera_rear_image",
    "/sensor/camera_side_left_front_image",
    "/sensor/camera_side_left_rear_image",
    "/sensor/camera_side_right_front_image",
    "/sensor/camera_side_right_rear_image",
)


def _resolve_path(path: str | Path | None, default: Path) -> Path:
    return Path(path).expanduser().resolve() if path is not None else default.resolve()


def _optional_input_dir(path: Path, *, skip_missing: bool, step: str) -> Path | None:
    if path.is_dir():
        return path
    if skip_missing:
        logger.warning("Skip %s: input directory does not exist: %s", step, path)
        return None
    raise NotADirectoryError(f"{step} input directory does not exist: {path}")


def _input_dir_or_skip(
    path: Path,
    input_files: list[str],
    *,
    skip_missing: bool,
    step: str,
) -> tuple[Path | None, dict[str, Any] | None]:
    if path.is_dir():
        return path, None
    if input_files:
        return None, None
    skipped = _optional_input_dir(path, skip_missing=skip_missing, step=step)
    if skipped is None:
        return None, {
            "step": step,
            "skipped": True,
            "reason": "missing input directory",
        }
    return skipped, None


def _positive_int(value: int | None, parser: argparse.ArgumentParser, name: str) -> None:
    if value is not None and value <= 0:
        parser.error(f"{name} must be greater than zero")


def _camera_topic_specs(
    args: argparse.Namespace,
    mcap_paths: list[Path],
) -> list[mcap_to_jpg.TopicSpec]:
    if args.camera_topic:
        return mcap_to_jpg.parse_topic_specs(args.camera_topic)
    if args.all_camera_topics:
        return mcap_to_jpg.discover_image_topic_specs(mcap_paths)
    return mcap_to_jpg.parse_topic_specs(DEFAULT_CAMERA_TOPICS)


def _write_pcd(args: argparse.Namespace, paths: argparse.Namespace) -> dict[str, Any]:
    input_dir, skipped = _input_dir_or_skip(
        paths.lidar_input_dir,
        args.lidar_input_file,
        skip_missing=args.skip_missing,
        step="pcd",
    )
    if skipped is not None:
        return skipped

    mcap_paths = mcap_to_pcd.discover_mcap_files(input_dir, args.lidar_input_file)
    frames = mcap_to_pcd.convert_mcaps(
        mcap_paths,
        paths.pcd_output_dir,
        topic=args.lidar_topic,
        sensor_name=args.lidar_sensor_name,
        max_frames=args.lidar_max_frames,
        overwrite=args.overwrite,
    )
    if not frames:
        raise RuntimeError(f"no point-cloud frames converted from {args.lidar_topic!r}")

    meta_path = None
    if not args.no_meta:
        meta = mcap_to_pcd.build_meta(
            dataset_name=args.dataset_name or paths.dataset_root.name,
            sensor_name=args.lidar_sensor_name,
            topic=args.lidar_topic,
            frames=frames,
        )
        meta_path = paths.pcd_output_dir / "meta.json"
        mcap_to_pcd._atomic_write_json(meta_path, meta, overwrite=args.overwrite)

    return {
        "step": "pcd",
        "input_files": [str(path) for path in mcap_paths],
        "output_dir": str(paths.pcd_output_dir),
        "frame_count": len(frames),
        "meta": str(meta_path) if meta_path else None,
    }


def _write_jpg(args: argparse.Namespace, paths: argparse.Namespace) -> dict[str, Any]:
    input_dir, skipped = _input_dir_or_skip(
        paths.camera_input_dir,
        args.camera_input_file,
        skip_missing=args.skip_missing,
        step="jpg",
    )
    if skipped is not None:
        return skipped

    mcap_paths = mcap_to_jpg.discover_mcap_files(input_dir, args.camera_input_file)
    topic_specs = _camera_topic_specs(args, mcap_paths)
    frames, stats = mcap_to_jpg.convert_mcaps(
        mcap_paths,
        paths.jpg_output_dir,
        topic_specs=topic_specs,
        max_frames=args.camera_max_frames,
        overwrite=args.overwrite,
        quality=args.jpg_quality,
        show_decoder_log=args.show_decoder_log,
    )
    if not frames:
        raise RuntimeError("no JPG frames converted")

    meta_path = None
    if not args.no_meta:
        meta = mcap_to_jpg.build_meta(
            dataset_name=args.dataset_name or paths.dataset_root.name,
            topic_specs=topic_specs,
            frames=frames,
            stats=stats,
        )
        meta_path = paths.jpg_output_dir / "meta.json"
        mcap_to_jpg._atomic_write_json(meta_path, meta, overwrite=args.overwrite)

    return {
        "step": "jpg",
        "input_files": [str(path) for path in mcap_paths],
        "output_dir": str(paths.jpg_output_dir),
        "image_count": len(frames),
        "sensor_count": len(topic_specs),
        "meta": str(meta_path) if meta_path else None,
    }


def _write_struct(args: argparse.Namespace, paths: argparse.Namespace) -> dict[str, Any]:
    input_dir, skipped = _input_dir_or_skip(
        paths.struct_input_dir,
        args.struct_input_file,
        skip_missing=args.skip_missing,
        step="struct",
    )
    if skipped is not None:
        return skipped

    mcap_paths = mcap_to_struct.discover_mcap_files(input_dir, args.struct_input_file)
    frames, latest_message = mcap_to_struct.convert_mcaps(
        mcap_paths,
        paths.struct_output_dir,
        topic=args.struct_topic,
        max_frames=args.struct_max_frames,
        overwrite=args.overwrite,
    )
    if not frames:
        raise RuntimeError(f"no calibration messages converted from {args.struct_topic!r}")

    meta_path = None
    if not args.no_meta:
        meta = mcap_to_struct.build_meta(
            dataset_name=args.dataset_name or paths.dataset_root.name,
            topic=args.struct_topic,
            frames=frames,
            latest_message=latest_message,
        )
        meta_path = paths.struct_output_dir / "meta.json"
        mcap_to_struct._atomic_write_json(meta_path, meta, overwrite=args.overwrite)

    return {
        "step": "struct",
        "input_files": [str(path) for path in mcap_paths],
        "output_dir": str(paths.struct_output_dir),
        "frame_count": len(frames),
        "latest": str(paths.struct_output_dir / "calibration_latest.json"),
        "meta": str(meta_path) if meta_path else None,
    }


def _write_newstruct(args: argparse.Namespace, paths: argparse.Namespace) -> dict[str, Any]:
    if not paths.struct_output_dir.is_dir():
        if args.skip_missing:
            logger.warning(
                "Skip newstruct: struct JSON directory does not exist: %s",
                paths.struct_output_dir,
            )
            return {
                "step": "newstruct",
                "skipped": True,
                "reason": "missing struct JSON directory",
            }
        raise NotADirectoryError(
            f"newstruct input directory does not exist: {paths.struct_output_dir}"
        )

    files = newstruct.calibration_files(paths.struct_output_dir)
    count, errors = newstruct.rewrite_files(files, dry_run=args.newstruct_dry_run)
    if errors:
        raise RuntimeError(
            "newstruct validation failed:\n" + "\n".join(errors[:20])
        )
    return {
        "step": "newstruct",
        "struct_json_dir": str(paths.struct_output_dir),
        "file_count": count,
        "dry_run": args.newstruct_dry_run,
    }


def _build_paths(args: argparse.Namespace) -> argparse.Namespace:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    return argparse.Namespace(
        dataset_root=dataset_root,
        lidar_input_dir=_resolve_path(args.lidar_input_dir, dataset_root / "lidar"),
        camera_input_dir=_resolve_path(args.camera_input_dir, dataset_root / "camera"),
        struct_input_dir=_resolve_path(args.struct_input_dir, dataset_root / "struct"),
        pcd_output_dir=_resolve_path(args.pcd_output_dir, dataset_root / "pcd"),
        jpg_output_dir=_resolve_path(args.jpg_output_dir, dataset_root / "jpg"),
        struct_output_dir=_resolve_path(
            args.struct_output_dir,
            dataset_root / "struct_json",
        ),
    )


def _selected_steps(raw_steps: list[str] | None) -> list[str]:
    steps = raw_steps or list(DEFAULT_STEPS)
    ordered = []
    seen = set()
    for step in steps:
        if step in seen:
            continue
        seen.add(step)
        ordered.append(step)
    return ordered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one dataset root from MCAP files to pcd, jpg and "
            "calibration JSON outputs."
        )
    )
    parser.add_argument(
        "dataset_root",
        help="Dataset root containing lidar/, camera/ and struct/ directories.",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=DEFAULT_STEPS,
        help=(
            "Run only the selected step. May be specified multiple times. "
            "Default runs pcd, jpg, struct and newstruct in order."
        ),
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip a selected step when its default input directory is missing.",
    )
    parser.add_argument("--dataset-name", help="Dataset name written to meta.json.")
    parser.add_argument("--overwrite", action="store_true", help="Replace outputs.")
    parser.add_argument("--no-meta", action="store_true", help="Do not write meta.json.")

    parser.add_argument("--lidar-input-dir", help="Defaults to <dataset_root>/lidar.")
    parser.add_argument(
        "--lidar-input-file",
        action="append",
        default=[],
        help="Additional lidar .mcap file; may be specified multiple times.",
    )
    parser.add_argument("--pcd-output-dir", help="Defaults to <dataset_root>/pcd.")
    parser.add_argument("--lidar-topic", default="/lidar/pandar")
    parser.add_argument("--lidar-sensor-name", default="p128_0")
    parser.add_argument("--lidar-max-frames", type=int)

    parser.add_argument("--camera-input-dir", help="Defaults to <dataset_root>/camera.")
    parser.add_argument(
        "--camera-input-file",
        action="append",
        default=[],
        help="Additional camera .mcap file; may be specified multiple times.",
    )
    parser.add_argument("--jpg-output-dir", help="Defaults to <dataset_root>/jpg.")
    parser.add_argument(
        "--camera-topic",
        action="append",
        default=[],
        help=(
            "Camera topic to convert. Use /topic or sensor_name=/topic. "
            "May be specified multiple times. If omitted, the top-level "
            "converter uses its six default camera topics."
        ),
    )
    parser.add_argument(
        "--all-camera-topics",
        action="store_true",
        help="Discover and convert all ImageV2 camera topics from the MCAP summary.",
    )
    parser.add_argument("--camera-max-frames", type=int)
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument(
        "--show-decoder-log",
        action="store_true",
        help="Show FFmpeg/OpenCV decoder warnings for H265 streams.",
    )

    parser.add_argument("--struct-input-dir", help="Defaults to <dataset_root>/struct.")
    parser.add_argument(
        "--struct-input-file",
        action="append",
        default=[],
        help="Additional struct .mcap file; may be specified multiple times.",
    )
    parser.add_argument(
        "--struct-output-dir",
        help="Defaults to <dataset_root>/struct_json.",
    )
    parser.add_argument("--struct-topic", default="/calib/calib_param")
    parser.add_argument("--struct-max-frames", type=int)
    parser.add_argument(
        "--newstruct-dry-run",
        action="store_true",
        help="Validate newstruct conversion without rewriting calibration JSON.",
    )

    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _positive_int(args.lidar_max_frames, parser, "--lidar-max-frames")
    _positive_int(args.camera_max_frames, parser, "--camera-max-frames")
    _positive_int(args.struct_max_frames, parser, "--struct-max-frames")
    if not 1 <= args.jpg_quality <= 100:
        parser.error("--jpg-quality must be between 1 and 100")
    if args.camera_topic and args.all_camera_topics:
        parser.error("--camera-topic and --all-camera-topics cannot be used together")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    paths = _build_paths(args)
    steps = _selected_steps(args.only)
    runners = {
        "pcd": _write_pcd,
        "jpg": _write_jpg,
        "struct": _write_struct,
        "newstruct": _write_newstruct,
    }

    logger.info("Dataset root: %s", paths.dataset_root)
    logger.info("Selected steps: %s", ", ".join(steps))

    results = []
    for step in steps:
        logger.info("Start %s", step)
        result = runners[step](args, paths)
        results.append(result)
        logger.info("Done %s", step)

    print(
        json.dumps(
            {"dataset_root": str(paths.dataset_root), "results": results},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
