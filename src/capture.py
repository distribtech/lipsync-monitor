"""
Stream/file capture using PyAV (FFmpeg bindings).
Supports video files and UDP multicast (udp://@GROUP:PORT).
"""

from __future__ import annotations

import collections
import threading
import av
import numpy as np
from typing import Iterator, Optional, Tuple

# PyAV renamed the base error class: <=12 exposed ``av.AVError``,
# newer releases (>=14) only expose ``av.FFmpegError``.
_AVError = getattr(av, 'AVError', None) or getattr(av, 'FFmpegError', Exception)


class _FrameBuffer:
    """
    Thread-safe hand-off buffer between the decode thread and the consumer.

    Never blocks the producer: when more than ``max_video`` video frames are
    queued (consumer is behind), the oldest video frames are dropped so the
    reader keeps draining the socket. Audio frames are always kept.
    """

    def __init__(self, max_video: int = 8) -> None:
        self._items: "collections.deque" = collections.deque()
        self._cond = threading.Condition()
        self._closed = False
        self._max_video = max_video
        self._n_video = 0

    def put(self, item: Tuple[str, np.ndarray, float]) -> None:
        with self._cond:
            self._items.append(item)
            if item[0] == 'video':
                self._n_video += 1
                while self._n_video > self._max_video:
                    for i, it in enumerate(self._items):
                        if it[0] == 'video':
                            del self._items[i]
                            self._n_video -= 1
                            break
                    else:
                        break
            self._cond.notify()

    def get(self) -> Optional[Tuple[str, np.ndarray, float]]:
        with self._cond:
            while not self._items and not self._closed:
                self._cond.wait()
            if not self._items:
                return None
            item = self._items.popleft()
            if item[0] == 'video':
                self._n_video -= 1
            return item

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()


class StreamCapture:
    """Decodes video frames and audio samples from any source PyAV/FFmpeg can open."""

    def __init__(
        self,
        source: str,
        buffer_size: int = 16_777_216,
        timeout: int = 5_000_000,
    ) -> None:
        self.source = source
        self.is_multicast = source.lower().startswith(('udp://', 'rtp://', 'rtsp://'))

        options: dict[str, str] = {}
        if self.is_multicast:
            options = {
                'buffer_size':      str(buffer_size),   # OS socket recv buffer
                'fifo_size':        '1000000',          # FFmpeg UDP packet FIFO
                'reuse':            '1',
                'timeout':          str(timeout),
                'overrun_nonfatal': '1',
                # Drop corrupt packets/frames instead of decoding garbage; a
                # dropped UDP packet otherwise smears macroblocks across the
                # frame (and pollutes the lip signal).
                'fflags':           'discardcorrupt',
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

        # Resampler to coerce any input audio into mono float32 planar.
        # Newer PyAV dropped the ``to_ndarray(format=...)`` shortcut, so we
        # convert explicitly here instead.
        self._audio_resampler = av.AudioResampler(format='fltp', layout='mono')

        # Threaded-reader state (used for live streams only).
        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self._buf: Optional["_FrameBuffer"] = None
        self._error: Optional[Exception] = None

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
    # Decode
    # ------------------------------------------------------------------

    def _decode(self) -> Iterator[Tuple[str, np.ndarray, float]]:
        """
        Yields (kind, data, pts_seconds) for each decoded frame.

        kind == 'video': data is np.ndarray BGR uint8, shape (H, W, 3)
        kind == 'audio': data is np.ndarray float32 mono, shape (N,)
        """
        streams = [self.video_stream, self.audio_stream]
        for packet in self._container.demux(*streams):
            if self._stop:
                return
            try:
                for frame in packet.decode():
                    if frame.pts is None:
                        continue
                    pts = float(frame.pts * frame.time_base)

                    if isinstance(frame, av.VideoFrame):
                        bgr = frame.reformat(format='bgr24').to_ndarray()
                        yield 'video', bgr, pts

                    elif isinstance(frame, av.AudioFrame):
                        # Resample to mono float32 planar; one input frame may
                        # yield zero or more output frames.
                        for out in self._audio_resampler.resample(frame):
                            pcm = out.to_ndarray()          # shape (1, samples)
                            mono = pcm.reshape(-1).astype(np.float32)
                            yield 'audio', mono, pts

            except _AVError:
                continue

    # ------------------------------------------------------------------
    # Packet iterator
    # ------------------------------------------------------------------

    def packets(self) -> Iterator[Tuple[str, np.ndarray, float]]:
        """
        Yields decoded (kind, data, pts) frames.

        For a file the decoder is driven directly (every frame matters). For a
        live stream a background thread continuously drains + decodes the
        socket so packets are never lost to a slow consumer; if the consumer
        (detection + display) falls behind, the oldest *video* frames are
        dropped while audio is kept intact. This prevents the UDP receive
        buffer from overflowing, which is what causes macroblock glitches.
        """
        if not self.is_multicast:
            yield from self._decode()
            return

        self._buf = _FrameBuffer(max_video=8)

        def _reader() -> None:
            try:
                for item in self._decode():
                    if self._stop:
                        break
                    self._buf.put(item)
            except Exception as exc:            # noqa: BLE001
                self._error = exc
            finally:
                self._buf.close()

        self._thread = threading.Thread(target=_reader, daemon=True)
        self._thread.start()

        while True:
            item = self._buf.get()
            if item is None:
                break
            yield item

        if self._error is not None:
            raise self._error

    def close(self) -> None:
        self._stop = True
        if self._buf is not None:
            self._buf.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._container.close()
