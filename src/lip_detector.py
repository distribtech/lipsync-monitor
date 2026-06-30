"""
Lip openness extraction via a two-stage CNN pipeline.

Stage 1 — Face localisation
    MediaPipe Face Detection, full-range model (a BlazeFace CNN). Unlike
    FaceMesh, this reliably finds small / distant faces in high-resolution
    broadcast frames (e.g. a presenter occupying ~5 % of a 1080p frame).

Stage 2 — Lip landmarks
    The detected face box is expanded to a square ROI, cropped and upscaled,
    then MediaPipe FaceMesh runs on that zoomed crop. Feeding FaceMesh a
    tight, square, high-resolution face crop gives far more precise lip
    landmarks than running it on the whole wide frame — and avoids the
    "NORM_RECT without IMAGE_DIMENSIONS" non-square ROI degradation.

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

    # ROI expansion around the detected face box, and the size the square
    # crop is upscaled to before FaceMesh sees it.
    _ROI_MARGIN = 2.0
    _CROP_SIZE  = 256

    def __init__(self, min_detection_confidence: float = 0.5) -> None:
        # Stage 1: full-range face detector (model_selection=1) — finds the
        # small, distant faces that FaceMesh alone misses on wide frames.
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=min_detection_confidence,
        )
        # Stage 2: landmark mesh on the upscaled crop. static_image_mode=True
        # because successive crops are not a continuous video of one face
        # (the ROI jumps around), so per-frame detection is more reliable.
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )

    def detect(self, frame_bgr: np.ndarray) -> Optional[float]:
        """Process one BGR frame; returns lip openness or None."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # --- Stage 1: locate the face -------------------------------------
        det = self._detector.process(rgb)
        if not det.detections:
            return None
        box = det.detections[0].location_data.relative_bounding_box

        # Square ROI (in pixels) centred on the face, expanded by a margin.
        fw, fh = box.width * w, box.height * h
        cx = (box.xmin + box.width  / 2) * w
        cy = (box.ymin + box.height / 2) * h
        side = max(fw, fh) * self._ROI_MARGIN
        if side < 2.0:
            return None

        x0 = max(0, int(round(cx - side / 2)))
        y0 = max(0, int(round(cy - side / 2)))
        x1 = min(w, int(round(cx + side / 2)))
        y1 = min(h, int(round(cy + side / 2)))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None

        crop = rgb[y0:y1, x0:x1]

        # Pad to a square (preserve aspect ratio) then upscale, so FaceMesh
        # receives an undistorted, high-resolution face.
        ch, cw = crop.shape[:2]
        s = max(ch, cw)
        square = np.zeros((s, s, 3), dtype=crop.dtype)
        square[:ch, :cw] = crop
        square = cv2.resize(square, (self._CROP_SIZE, self._CROP_SIZE))

        # --- Stage 2: lip landmarks on the crop ---------------------------
        result = self._mesh.process(square)
        if not result.multi_face_landmarks:
            return None

        lm = result.multi_face_landmarks[0].landmark

        def pt(idx: int) -> np.ndarray:
            return np.array(
                [lm[idx].x * self._CROP_SIZE, lm[idx].y * self._CROP_SIZE],
                dtype=np.float64,
            )

        upper = pt(self._UPPER_LIP)
        lower = pt(self._LOWER_LIP)
        left  = pt(self._LEFT_EYE)
        right = pt(self._RIGHT_EYE)

        eye_dist = np.linalg.norm(right - left)
        if eye_dist < 1.0:          # landmarks collapsed / unreliable
            return None

        return float(np.linalg.norm(upper - lower) / eye_dist)

    def close(self) -> None:
        self._detector.close()
        self._mesh.close()
