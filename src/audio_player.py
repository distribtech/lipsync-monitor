"""
Optional speaker playback of the decoded audio (mono), for use with --show.

Audio frames from the capture pipeline are pushed via feed(); a sounddevice
callback pulls them for playback. The internal buffer is capped so audio
latency stays bounded even if the video side (GPU detection) runs slow — old
samples are dropped rather than letting A/V drift without limit.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

log = logging.getLogger(__name__)


class AudioPlayer:
    def __init__(self, samplerate: int, max_latency_s: float = 0.4) -> None:
        import sounddevice as sd            # imported lazily; only when --audio
        self._buf = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._max = int(samplerate * max_latency_s)
        self._stream = sd.OutputStream(
            samplerate=samplerate,
            channels=1,
            dtype="float32",
            blocksize=1024,
            callback=self._callback,
        )

    def _callback(self, outdata, frames, time, status) -> None:  # noqa: ANN001
        with self._lock:
            n = min(len(self._buf), frames)
            outdata[:n, 0] = self._buf[:n]
            outdata[n:, 0] = 0.0            # underrun -> silence
            self._buf = self._buf[n:]

    def start(self) -> None:
        self._stream.start()

    def feed(self, mono: np.ndarray) -> None:
        with self._lock:
            self._buf = np.concatenate((self._buf, mono.astype(np.float32, copy=False)))
            if len(self._buf) > self._max:  # bound latency: drop oldest
                self._buf = self._buf[-self._max:]

    def close(self) -> None:
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass
