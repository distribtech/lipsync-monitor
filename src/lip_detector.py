"""
Lip openness extraction via InsightFace (GPU-accelerated).

Why InsightFace instead of MediaPipe
------------------------------------
MediaPipe's Python package is CPU-only on Windows and its BlazeFace detector
misses small / non-frontal faces — exactly the case in HD broadcast frames
(a presenter ~5 % of a 1080p frame, often looking down). InsightFace runs on
the GPU via onnxruntime-CUDA and uses:

  * SCRFD (det_10g) — a far more robust face detector, run at high input
    resolution so small faces are found.
  * 1k3d68 — 68 3D facial landmarks (dlib ordering) for precise lip points.

Landmark indices (standard 68-point / dlib ordering)
----------------------------------------------------
 62 — inner upper lip centre   (top of mouth opening)
 66 — inner lower lip centre   (bottom of mouth opening)
 36 — right eye outer corner   ┐
 45 — left eye outer corner    ┘  reference for scale normalisation
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .gpu_setup import register_cuda_dlls

register_cuda_dlls()                       # must run before onnxruntime sessions

from insightface.app import FaceAnalysis   # noqa: E402  (after DLL registration)

log = logging.getLogger(__name__)


class LipDetector:
    """
    Returns a per-frame lip-openness scalar (float ≥ 0) normalised by the
    inter-eye distance so the signal is scale-invariant.

    Returns None when no face is found in the frame.
    """

    _UPPER_LIP = 62
    _LOWER_LIP = 66
    _LEFT_EYE  = 36
    _RIGHT_EYE = 45

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        det_size: int = 1024,
        model_name: str = "buffalo_l",
    ) -> None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection", "landmark_3d_68"],
            providers=providers,
        )
        self._app.prepare(
            ctx_id=0,
            det_thresh=min_detection_confidence,
            det_size=(det_size, det_size),
        )

        # Report whether we actually got the GPU.
        active = "CPU"
        try:
            p = self._app.models["detection"].session.get_providers()[0]
            active = "GPU (CUDA)" if "CUDA" in p else "CPU"
        except Exception:
            pass
        log.info(f"LipDetector: InsightFace {model_name} on {active}, "
                 f"det_size={det_size}")

    def detect(self, frame_bgr: np.ndarray) -> Optional[float]:
        """Process one BGR frame; returns lip openness or None."""
        faces = self._app.get(frame_bgr)
        if not faces:
            return None

        # Largest / most confident face on screen.
        face = max(faces, key=lambda f: f.det_score)
        lm = face.landmark_3d_68
        if lm is None:
            return None
        lm = lm[:, :2]                       # drop Z

        eye_dist = float(np.linalg.norm(lm[self._LEFT_EYE] - lm[self._RIGHT_EYE]))
        if eye_dist < 1.0:
            return None

        mouth = float(np.linalg.norm(lm[self._UPPER_LIP] - lm[self._LOWER_LIP]))
        return mouth / eye_dist

    def close(self) -> None:
        # InsightFace holds onnxruntime sessions; nothing explicit to free.
        pass
