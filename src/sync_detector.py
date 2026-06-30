"""
A/V offset estimation via cross-correlation of lip-openness and audio-RMS signals.

Sign convention (same as SyncNet / Oxford VGG paper):
    offset_ms > 0  →  audio leads video  (sound arrives before lips move on screen)
    offset_ms < 0  →  audio lags  video  (sound arrives after  lips move on screen)

ITU-R BT.1359-1 acceptable range: −125 ms … +45 ms.
"""

from __future__ import annotations

import numpy as np
import scipy.signal
from collections import deque
from typing import Deque, Optional, Tuple


# (pts_seconds, value) samples
_Sample = Tuple[float, float]


class SyncDetector:
    def __init__(
        self,
        fps: float,
        window_sec: float = 3.0,
        min_lip_variance: float = 1e-4,
        max_offset_ms: float = 500.0,
    ) -> None:
        self.fps              = fps
        self.window_sec       = window_sec
        self.min_lip_variance = min_lip_variance
        self._max_lag_frames  = int(max_offset_ms / 1000.0 * fps)

        self._lip:   Deque[_Sample] = deque()
        self._audio: Deque[_Sample] = deque()

    # ------------------------------------------------------------------
    # Feed data
    # ------------------------------------------------------------------

    def add_lip(self, pts: float, value: Optional[float]) -> None:
        self._lip.append((pts, value if value is not None else 0.0))
        self._trim(self._lip, pts)

    def add_audio(self, pts: float, rms: float) -> None:
        self._audio.append((pts, rms))
        self._trim(self._audio, pts)

    def _trim(self, buf: Deque[_Sample], latest_pts: float) -> None:
        cutoff = latest_pts - self.window_sec
        while buf and buf[0][0] < cutoff:
            buf.popleft()

    @property
    def lip_buf_len(self) -> int:
        return len(self._lip)

    @property
    def audio_buf_len(self) -> int:
        return len(self._audio)

    # ------------------------------------------------------------------
    # Compute offset
    # ------------------------------------------------------------------

    def compute(self) -> Tuple[Optional[float], float, str]:
        """
        Returns (offset_ms, confidence, status).

        status values
        -------------
        'ok'                 — valid measurement
        'insufficient_data'  — not enough samples yet
        'no_face'            — lip signal is all zeros (face never detected)
        'low_speech'         — lips or audio not moving enough to correlate
        """
        if len(self._lip) < int(self.fps * 1.5) or len(self._audio) < 10:
            return None, 0.0, 'insufficient_data'

        # Build a common uniform time grid at video FPS
        t0 = max(self._lip[0][0],  self._audio[0][0])
        t1 = min(self._lip[-1][0], self._audio[-1][0])
        if t1 - t0 < 1.0:
            return None, 0.0, 'insufficient_data'

        t_grid = np.arange(t0, t1, 1.0 / self.fps)
        if len(t_grid) < 15:
            return None, 0.0, 'insufficient_data'

        lip_t,   lip_v   = zip(*self._lip)
        audio_t, audio_v = zip(*self._audio)

        lip_grid   = np.interp(t_grid, lip_t,   lip_v)
        audio_grid = np.interp(t_grid, audio_t, audio_v)

        # Speech activity checks
        lip_var   = float(np.var(lip_grid))
        audio_var = float(np.var(audio_grid))

        if np.mean(lip_grid) < 1e-9 and lip_var < 1e-12:
            return None, 0.0, 'no_face'
        if lip_var < self.min_lip_variance:
            return None, 0.0, 'low_speech'
        if audio_var < 1e-9:
            return None, 0.0, 'low_speech'

        # Z-score normalise
        lip_z   = (lip_grid   - lip_grid.mean())   / (lip_grid.std()   + 1e-9)
        audio_z = (audio_grid - audio_grid.mean()) / (audio_grid.std() + 1e-9)

        # Full cross-correlation
        # scipy convention: positive lag → in1 (lip) lags in2 (audio) → audio leads
        corr = scipy.signal.correlate(lip_z, audio_z, mode='full')
        lags = scipy.signal.correlation_lags(len(lip_z), len(audio_z), mode='full')

        # Restrict search to ±max_offset window
        mask         = np.abs(lags) <= self._max_lag_frames
        corr_masked  = np.where(mask, corr, -np.inf)
        peak_idx     = int(np.argmax(corr_masked))
        lag_frames   = int(lags[peak_idx])

        # offset_ms: positive → audio leads
        offset_ms  = (lag_frames / self.fps) * 1000.0

        # Confidence: normalised peak (0–1)
        confidence = float(np.clip(corr[peak_idx] / max(len(lip_z), 1), 0.0, 1.0))

        return offset_ms, confidence, 'ok'
