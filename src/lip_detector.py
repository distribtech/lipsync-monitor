"""
Lip openness extraction via MediaPipe FaceMesh.

Landmark indices used
---------------------
 13  — centre of inner upper lip   (top of mouth opening)
 14  — centre of inner lower lip   (bottom of mouth opening)
 33  — left eye outer corner       ┐
263  — right eye outer corner      ┘  reference for scale normalisation
"""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional


class LipDetector:
    """
    Returns a per-frame lip-openness scalar (float ≥ 0) normalised by the
    inter-eye distance so the signal is scale-invariant.

    Returns None when no face is found in the frame.
    """

    _UPPER_LIP  = 13
    _LOWER_LIP  = 14
    _LEFT_EYE   = 33
    _RIGHT_EYE  = 263

    def __init__(self, min_detection_confidence: float = 0.5) -> None:
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )

    def detect(self, frame_bgr: np.ndarray) -> Optional[float]:
        """Process one BGR frame; returns lip openness or None."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._mesh.process(rgb)

        if not result.multi_face_landmarks:
            return None

        lm = result.multi_face_landmarks[0].landmark

        def pt(idx: int) -> np.ndarray:
            return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float64)

        upper = pt(self._UPPER_LIP)
        lower = pt(self._LOWER_LIP)
        left  = pt(self._LEFT_EYE)
        right = pt(self._RIGHT_EYE)

        eye_dist = np.linalg.norm(right - left)
        if eye_dist < 1.0:          # face too small / partially visible
            return None

        return float(np.linalg.norm(upper - lower) / eye_dist)

    def close(self) -> None:
        self._mesh.close()
