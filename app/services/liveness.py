from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from app.config import settings


class LivenessUnavailableError(RuntimeError):
    """Raised when passive liveness cannot produce a trustworthy result."""


@dataclass(frozen=True)
class LivenessResult:
    live: bool
    score: float
    threshold: float


class LivenessService:
    _instance = None
    _crop_scale = 1.2

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: Optional[float] = None,
        input_size: Optional[Tuple[int, int]] = None,
        live_class_index: Optional[int] = None,
        session_factory: Optional[Callable] = None,
    ):
        if self._initialized:
            return

        self.model_path = model_path if model_path is not None else settings.LIVENESS_MODEL_PATH
        self.threshold = threshold if threshold is not None else settings.LIVENESS_THRESHOLD
        self.input_size = input_size if input_size is not None else settings.LIVENESS_INPUT_SIZE
        self.live_class_index = (
            live_class_index
            if live_class_index is not None
            else settings.LIVENESS_LIVE_CLASS_INDEX
        )

        if not self.model_path or not Path(self.model_path).is_file():
            raise LivenessUnavailableError("Liveness model is unavailable")
        if len(self.input_size) != 2 or min(self.input_size) <= 0:
            raise LivenessUnavailableError("Liveness input size is invalid")
        if not 0.0 <= self.threshold <= 1.0 or self.live_class_index < 0:
            raise LivenessUnavailableError("Liveness configuration is invalid")

        factory = session_factory or ort.InferenceSession
        try:
            self.session = factory(self.model_path, providers=["CPUExecutionProvider"])
            inputs = self.session.get_inputs()
            if not inputs or not inputs[0].name:
                raise ValueError("model has no named input")
            self.input_name = inputs[0].name
        except Exception as exc:
            raise LivenessUnavailableError("Liveness model is unavailable") from exc

        self._initialized = True

    def analyze(self, image: np.ndarray, bbox: np.ndarray) -> LivenessResult:
        tensor = self._preprocess(image, bbox)
        try:
            outputs = self.session.run(None, {self.input_name: tensor})
            logits = self._validate_output(outputs)
            shifted = logits - np.max(logits)
            exponents = np.exp(shifted)
            score = float((exponents / np.sum(exponents))[self.live_class_index])
        except LivenessUnavailableError:
            raise
        except Exception as exc:
            raise LivenessUnavailableError("Liveness inference failed") from exc

        return LivenessResult(score >= self.threshold, score, self.threshold)

    def _preprocess(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise LivenessUnavailableError("Invalid image for liveness analysis")

        coordinates = np.asarray(bbox, dtype=np.float32).reshape(-1)
        if coordinates.size != 4 or not np.all(np.isfinite(coordinates)):
            raise LivenessUnavailableError("Invalid face bounds")
        x1, y1, x2, y2 = coordinates
        if x2 <= x1 or y2 <= y1:
            raise LivenessUnavailableError("Invalid face bounds")

        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        half_width = (x2 - x1) * self._crop_scale / 2.0
        half_height = (y2 - y1) * self._crop_scale / 2.0
        image_height, image_width = image.shape[:2]
        left = max(0, int(np.floor(center_x - half_width)))
        top = max(0, int(np.floor(center_y - half_height)))
        right = min(image_width, int(np.ceil(center_x + half_width)))
        bottom = min(image_height, int(np.ceil(center_y + half_height)))
        if right <= left or bottom <= top:
            raise LivenessUnavailableError("Face bounds fall outside the image")

        crop = image[top:bottom, left:right]
        resized = cv2.resize(crop, self.input_size, interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
        return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None, ...])

    def _validate_output(self, outputs) -> np.ndarray:
        if not outputs:
            raise LivenessUnavailableError("Liveness model returned no output")
        output = np.asarray(outputs[0])
        if output.ndim == 2 and output.shape[0] == 1:
            output = output[0]
        if (
            output.ndim != 1
            or output.size < 2
            or self.live_class_index >= output.size
            or not np.all(np.isfinite(output))
        ):
            raise LivenessUnavailableError("Liveness model returned malformed output")
        return output.astype(np.float64, copy=False)
