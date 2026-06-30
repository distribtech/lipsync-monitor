"""
Audio energy extraction — stateless, single-call.
"""

from __future__ import annotations

import numpy as np


class AudioAnalyzer:
    """Computes RMS energy of a mono audio chunk."""

    @staticmethod
    def rms(samples: np.ndarray) -> float:
        """Root-mean-square of float32 samples in [-1, 1]."""
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
