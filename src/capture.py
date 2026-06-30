"""
Stream/file capture using PyAV (FFmpeg bindings).
Supports video files and UDP multicast (udp://@GROUP:PORT).
"""

from __future__ import annotations

import av
import numpy as np
from typing import Iterator, Optional, Tuple


class StreamCapture:
    """Decodes video frames and audio samples from any source PyAV/FFmpeg can open."""

    def __init__(
        self,
        source: str,
        buffer_size: int = 2_097_152,
        timeout: int = 5_000_000,
    ) -> None:
        self.source = source
        self.is_multicast = source.lower().startswith(('udp://', 'rtp://', 'rtsp://'))

        options: dict[str, str] = {}
        if self.is_multicast:
            options = {
                'buffer_size':      str(buffer_size),
                'reuse':            '1',
                'timeout':          str(timeout),
                'overrun_nonfatal': '1',
            }

        self._container = av.open(source, options=options or None)

        self.video_stream: Optional[av.VideoStream] = None
        self.audio_stream: Optional[av.AudioStream] = None

        for s in self._container.streams:
            if s.type == 'video' and self.video_stream is None:
                self.video_stream = s
                s.thread_type = 'AUTO'
            elif s.type == 'audio' and self.audio_stream is None:
                self.audio_stream = s

        if self.video_stream is None:
            raise ValueError(f'No video stream found in: {source}')
        if self.audio_stream is None:
            raise ValueError(f'No audio stream found in: {source}')

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        r = self.video_stream.average_rate or self.video_stream.base_rate
        return float(r) if r else 25.0

    @property
    def audio_sample_rate(self) -> int:
        return self.audio_stream.sample_rate

    @property
    def duration(self) -> float:
        """File duration in seconds; 0 for live streams."""
        d = self._container.duration
        return d / av.time_base if d else 0.0

    # ------------------------------------------------------------------
    # Packet iterator
    # ------------------------------------------------------------------

    def packets(self) -> Iterator[Tuple[str, np.ndarray, float]]:
        """
        Yields (kind, data, pts_seconds) for each decoded frame.

        kind == 'video': data is np.ndarray BGR uint8, shape (H, W, 3)
        kind == 'audio': data is np.ndarray float32 mono, shape (N,)
        """
        streams = [self.video_stream, self.audio_stream]
        for packet in self._container.demux(*streams):
            try:
                for frame in packet.decode():
                    if frame.pts is None:
                        continue
                    pts = float(frame.pts * frame.time_base)

                    if isinstance(frame, av.VideoFrame):
                        yield 'video', frame.to_ndarray(format='bgr24'), pts

                    elif isinstance(frame, av.AudioFrame):
                        # shape: (channels, samples) → average → mono float32
                        pcm = frame.to_ndarray(format='fltp')
                        mono = np.mean(pcm, axis=0).astype(np.float32)
                        yield 'audio', mono, pts

            except av.AVError:
                continue

    def close(self) -> None:
        self._container.close()
