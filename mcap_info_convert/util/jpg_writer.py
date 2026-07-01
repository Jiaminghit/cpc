"""Write camera ImageV2 frames as JPG images."""

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

try:
    from .mcap_reader import ImageFrame
except ImportError:  # Allow running files directly from this directory.
    from mcap_reader import ImageFrame


H265_ENCODINGS = {"h265", "hevc", "h.265", "h_265"}
JPEG_ENCODINGS = {"jpeg", "jpg", "mjpeg"}
PNG_ENCODINGS = {"png"}


@dataclass(frozen=True)
class WrittenJpg:
    """One JPG image written by this module."""

    timestamp_ns: int
    path: Path
    width: int
    height: int
    source_encoding: str
    source_message_index: int | None = None


def _normalized_encoding(encoding: str) -> str:
    return encoding.strip().lower().replace("-", "").replace("_", "")


def _is_h265(encoding: str) -> bool:
    normalized = encoding.strip().lower()
    return normalized in H265_ENCODINGS or _normalized_encoding(encoding) == "h265"


def _is_jpeg(encoding: str) -> bool:
    return _normalized_encoding(encoding) in JPEG_ENCODINGS


def _is_png(encoding: str) -> bool:
    return _normalized_encoding(encoding) in PNG_ENCODINGS


def _destination(
    output_dir: Path,
    timestamp_ns: int,
    used_timestamps: set[int] | None,
    overwrite: bool,
) -> Path:
    if used_timestamps is not None:
        if timestamp_ns in used_timestamps:
            raise ValueError(f"duplicate image timestamp in input: {timestamp_ns}")
        used_timestamps.add(timestamp_ns)

    destination = output_dir / f"{timestamp_ns}.jpg"
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {destination}; "
            "use --overwrite to replace existing JPG files"
        )
    return destination


def _write_cv_jpg(
    destination: Path,
    image: np.ndarray,
    *,
    quality: int,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp.jpg",
            dir=destination.parent,
            delete=False,
        ) as output:
            temp_path = Path(output.name)

        ok = cv2.imwrite(
            str(temp_path),
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
        )
        if not ok:
            raise RuntimeError(f"failed to encode JPG: {destination}")
        os.chmod(temp_path, 0o644)
        os.replace(temp_path, destination)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    return destination


@contextmanager
def _suppress_stderr():
    """Temporarily silence native decoder logs written directly to stderr."""

    sys.stderr.flush()
    saved_stderr = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        os.close(devnull)


def _decode_encoded_image(frame: ImageFrame) -> np.ndarray:
    array = np.frombuffer(frame.data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            f"{frame.source_file}: failed to decode {frame.encoding} image "
            f"on {frame.topic} at {frame.timestamp_ns}"
        )
    return image


def _reshape_raw(frame: ImageFrame, channels: int) -> np.ndarray:
    row_bytes = frame.width * channels
    if frame.step > 0 and frame.step < row_bytes:
        raise ValueError(
            f"{frame.source_file}: invalid raw image step on {frame.topic}: "
            f"step={frame.step}, expected at least {row_bytes}"
        )

    step = frame.step or row_bytes
    expected_size = step * frame.height
    if len(frame.data) < expected_size:
        raise ValueError(
            f"{frame.source_file}: truncated raw image on {frame.topic}: "
            f"got {len(frame.data)} bytes, expected at least {expected_size}"
        )

    rows = np.frombuffer(frame.data[:expected_size], dtype=np.uint8).reshape(
        frame.height,
        step,
    )
    packed = rows[:, :row_bytes]
    if channels == 1:
        return packed.reshape(frame.height, frame.width)
    return packed.reshape(frame.height, frame.width, channels)


def _decode_raw_image(frame: ImageFrame) -> np.ndarray:
    encoding = _normalized_encoding(frame.encoding)
    if encoding in {"bgr8", "8uc3"}:
        return _reshape_raw(frame, 3)
    if encoding == "rgb8":
        return cv2.cvtColor(_reshape_raw(frame, 3), cv2.COLOR_RGB2BGR)
    if encoding in {"mono8", "8uc1"}:
        return _reshape_raw(frame, 1)
    if encoding == "bgra8":
        return cv2.cvtColor(_reshape_raw(frame, 4), cv2.COLOR_BGRA2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(_reshape_raw(frame, 4), cv2.COLOR_RGBA2BGR)
    raise ValueError(
        f"{frame.source_file}: unsupported image encoding {frame.encoding!r} "
        f"on {frame.topic}; supported encodings are h265, jpeg, png, "
        "bgr8, rgb8, mono8, bgra8 and rgba8"
    )


def write_jpg(
    output_path: str | Path,
    frame: ImageFrame,
    *,
    overwrite: bool = False,
    quality: int = 95,
) -> Path:
    """Write one non-H265 ImageV2 frame as a JPG file."""

    if _is_h265(frame.encoding):
        raise ValueError("H265 frames must be written with write_jpg_sequence()")

    destination = Path(output_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"JPG file already exists: {destination}")

    if _is_jpeg(frame.encoding) or _is_png(frame.encoding):
        image = _decode_encoded_image(frame)
    else:
        image = _decode_raw_image(frame)
    return _write_cv_jpg(destination, image, quality=quality)


def _write_h265_sequence(
    output_dir: Path,
    frames: list[ImageFrame],
    *,
    overwrite: bool,
    quality: int,
    used_timestamps: set[int] | None,
    show_decoder_log: bool,
) -> list[WrittenJpg]:
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".image_stream.",
            suffix=".h265",
            dir=output_dir,
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            for frame in frames:
                stream.write(frame.data)
            stream.flush()
            os.fsync(stream.fileno())

        decoded_images: list[np.ndarray] = []
        context = nullcontext() if show_decoder_log else _suppress_stderr()
        with context:
            capture = cv2.VideoCapture(str(temp_path))
            if not capture.isOpened():
                raise RuntimeError(
                    f"failed to open H265 stream with OpenCV: {temp_path}"
                )

            while True:
                ok, image = capture.read()
                if not ok:
                    break
                decoded_images.append(image)
            capture.release()

        if not decoded_images:
            raise RuntimeError(f"no frames decoded from H265 stream: {temp_path}")
        if len(decoded_images) > len(frames):
            raise RuntimeError(
                f"decoded more frames than input messages: "
                f"decoded={len(decoded_images)}, messages={len(frames)}"
            )

        start_index = len(frames) - len(decoded_images)
        written: list[WrittenJpg] = []
        for offset, image in enumerate(decoded_images):
            source_index = start_index + offset
            source_frame = frames[source_index]
            destination = _destination(
                output_dir,
                source_frame.timestamp_ns,
                used_timestamps,
                overwrite,
            )
            _write_cv_jpg(destination, image, quality=quality)
            written.append(
                WrittenJpg(
                    timestamp_ns=source_frame.timestamp_ns,
                    path=destination,
                    width=int(image.shape[1]),
                    height=int(image.shape[0]),
                    source_encoding=source_frame.encoding,
                    source_message_index=source_index,
                )
            )
        return written
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def write_jpg_sequence(
    output_dir: str | Path,
    frames: Iterable[ImageFrame],
    *,
    overwrite: bool = False,
    quality: int = 95,
    used_timestamps: set[int] | None = None,
    show_decoder_log: bool = False,
) -> list[WrittenJpg]:
    """Write an ImageV2 frame sequence to JPG files.

    H265/HEVC messages are decoded as one continuous stream.  If decoding starts
    mid-GOP, OpenCV may drop the first few packets until reference frames are
    available; decoded frames are then paired with the trailing input timestamps.
    """

    frame_list = list(frames)
    if not frame_list:
        return []

    output_path = Path(output_dir)
    encodings = {_normalized_encoding(frame.encoding) for frame in frame_list}
    if len(encodings) != 1:
        raise ValueError(f"mixed image encodings in one sequence: {sorted(encodings)}")

    first = frame_list[0]
    if _is_h265(first.encoding):
        return _write_h265_sequence(
            output_path,
            frame_list,
            overwrite=overwrite,
            quality=quality,
            used_timestamps=used_timestamps,
            show_decoder_log=show_decoder_log,
        )

    written: list[WrittenJpg] = []
    for index, frame in enumerate(frame_list):
        destination = _destination(
            output_path,
            frame.timestamp_ns,
            used_timestamps,
            overwrite,
        )
        write_jpg(destination, frame, overwrite=overwrite, quality=quality)
        written.append(
            WrittenJpg(
                timestamp_ns=frame.timestamp_ns,
                path=destination,
                width=frame.width,
                height=frame.height,
                source_encoding=frame.encoding,
                source_message_index=index,
            )
        )
    return written
