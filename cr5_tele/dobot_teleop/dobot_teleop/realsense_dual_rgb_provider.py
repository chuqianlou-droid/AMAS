"""Small dual-RealSense RGB provider used by the CR5A PI0 recorder.

``pyrealsense2`` is imported only when hardware capture starts, so schema and
dataset tools remain usable on development machines without RealSense drivers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DualRealSenseRGBProvider:
    # 224x224 is the PI0 input size, not a RealSense hardware profile. Both
    # D415 and D435 reliably expose this common RGB profile; frames are then
    # resized in memory to the requested recorder dimensions.
    CAPTURE_WIDTH = 640
    CAPTURE_HEIGHT = 480
    CAPTURE_FPS = 30

    d415_serial: str
    d435_serial: str
    width: int = 224
    height: int = 224
    fps: int = 15
    timeout_ms: int = 3000

    def __post_init__(self) -> None:
        self._rs = None
        self._d415_pipeline = None
        self._d435_pipeline = None

    def start(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("pyrealsense2 is required for RealSense capture. Install librealsense Python bindings.") from exc
        self._rs = rs
        try:
            self._d415_pipeline = self._start_camera(self.d415_serial)
            self._d435_pipeline = self._start_camera(self.d435_serial)
        except Exception:
            self.close()
            raise

    def _start_camera(self, serial: str):
        assert self._rs is not None
        pipeline = self._rs.pipeline()
        config = self._rs.config()
        config.enable_device(serial)
        config.enable_stream(
            self._rs.stream.color,
            self.CAPTURE_WIDTH,
            self.CAPTURE_HEIGHT,
            self._rs.format.rgb8,
            self.CAPTURE_FPS,
        )
        pipeline.start(config)
        return pipeline

    # Retry constants for transient USB frame drops
    _READ_RETRIES = 5
    _READ_RETRY_TIMEOUT_MS = 1000
    _READ_RETRY_DELAY_S = 0.1

    def get_rgb_images(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(d435_rgb, d415_rgb)`` as HWC RGB uint8 arrays."""
        if self._d415_pipeline is None or self._d435_pipeline is None:
            raise RuntimeError("DualRealSenseRGBProvider.start() must be called before capture")
        d435 = self._read_color(self._d435_pipeline, self.d435_serial)
        d415 = self._read_color(self._d415_pipeline, self.d415_serial)
        return d435, d415

    def _read_color(self, pipeline, serial: str) -> np.ndarray:
        last_error: Exception | None = None
        for attempt in range(self._READ_RETRIES):
            try:
                # Use a shorter timeout per attempt so total wait is bounded
                timeout = self.timeout_ms if attempt == 0 else self._READ_RETRY_TIMEOUT_MS
                frames = pipeline.wait_for_frames(timeout)
            except RuntimeError as exc:
                last_error = exc
                if attempt < self._READ_RETRIES - 1:
                    import time
                    time.sleep(self._READ_RETRY_DELAY_S)
                continue

            color_frame = frames.get_color_frame()
            if not color_frame:
                last_error = RuntimeError(f"RealSense {serial} returned no RGB frame")
                if attempt < self._READ_RETRIES - 1:
                    import time
                    time.sleep(self._READ_RETRY_DELAY_S)
                continue

            image = np.asanyarray(color_frame.get_data())
            if image.shape != (self.CAPTURE_HEIGHT, self.CAPTURE_WIDTH, 3):
                last_error = RuntimeError(f"RealSense {serial} returned unexpected RGB shape {image.shape}")
                if attempt < self._READ_RETRIES - 1:
                    import time
                    time.sleep(self._READ_RETRY_DELAY_S)
                continue

            image = image.astype(np.uint8, copy=False)
            if (self.width, self.height) == (self.CAPTURE_WIDTH, self.CAPTURE_HEIGHT):
                return image
            return self._resize_rgb_nearest(image, self.width, self.height)

        # All retries exhausted
        raise RuntimeError(
            f"RealSense {serial}: failed to read frame after {self._READ_RETRIES} attempts"
        ) from last_error

    @staticmethod
    def _resize_rgb_nearest(image: np.ndarray, width: int, height: int) -> np.ndarray:
        """Dependency-free resize for recorder output; model preprocessing can refine it later."""
        if width <= 0 or height <= 0:
            raise ValueError("Requested RealSense output width and height must be positive")
        y_indices = np.linspace(0, image.shape[0] - 1, height).astype(np.intp)
        x_indices = np.linspace(0, image.shape[1] - 1, width).astype(np.intp)
        return image[y_indices][:, x_indices]

    def close(self) -> None:
        for pipeline_name in ("_d415_pipeline", "_d435_pipeline"):
            pipeline = getattr(self, pipeline_name, None)
            if pipeline is not None:
                try:
                    pipeline.stop()
                finally:
                    setattr(self, pipeline_name, None)

    def __enter__(self) -> "DualRealSenseRGBProvider":
        self.start()
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()
