"""
Optional live preview window (OpenCV) with detection overlays.

  * RED   box  — detected face
  * GREEN box + points — lips

Also overlays the latest offset / status line. Press 'q' or ESC to quit.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .lip_detector import LipResult

_RED   = (0, 0, 255)      # BGR
_GREEN = (0, 255, 0)
_CYAN  = (0, 255, 255)


class Viewer:
    def __init__(self, title: str = "lipsync-monitor", max_width: int = 1280) -> None:
        self.title = title
        self.max_width = max_width
        self._sized = False
        # KEEPRATIO letterboxes instead of stretching if the user resizes.
        cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    def show(
        self,
        frame_bgr: np.ndarray,
        result: Optional[LipResult],
        status: str = "",
    ) -> bool:
        """Draw overlays and display. Returns False if the user asked to quit."""
        img = frame_bgr.copy()

        if result is not None:
            x1, y1, x2, y2 = result.face_bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), _RED, 2)
            cv2.putText(img, f"face {result.det_score:.2f}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, _RED, 2)

            pts = result.mouth_pts
            for x, y in pts:
                cv2.circle(img, (int(x), int(y)), 2, _GREEN, -1)
            mx1, my1 = pts[:, 0].min(), pts[:, 1].min()
            mx2, my2 = pts[:, 0].max(), pts[:, 1].max()
            cv2.rectangle(img, (int(mx1), int(my1)), (int(mx2), int(my2)), _GREEN, 2)

        if status:
            cv2.putText(img, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, _CYAN, 2, cv2.LINE_AA)

        h, w = img.shape[:2]
        if w > self.max_width:
            scale = self.max_width / w
            img = cv2.resize(img, (self.max_width, int(h * scale)))

        # Size the window to the frame's aspect ratio on the first draw so it
        # opens as 16:9 instead of the default square-ish WINDOW_NORMAL size.
        if not self._sized:
            dh, dw = img.shape[:2]
            cv2.resizeWindow(self.title, dw, dh)
            self._sized = True

        cv2.imshow(self.title, img)
        key = cv2.waitKey(1) & 0xFF
        return key not in (ord("q"), 27)     # 27 = ESC

    def close(self) -> None:
        cv2.destroyAllWindows()
