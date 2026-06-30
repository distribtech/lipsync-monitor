#!/usr/bin/env python3
"""
lipsync-monitor — audio/video desync detector via lip movement analysis.

Reads a video file or a live UDP multicast stream, detects the on-screen
speaker's lip movements, correlates them with audio energy, and reports
the A/V offset in milliseconds together with a timestamp.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys

from src.capture       import StreamCapture
from src.lip_detector  import LipDetector
from src.audio_analyzer import AudioAnalyzer
from src.sync_detector  import SyncDetector
from src.reporter       import Reporter


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='lipsync_monitor.py',
        description='Detect audio/video desync via lip movement cross-correlation.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
examples:
  UDP multicast:   %(prog)s -i udp://@239.0.0.1:5500
  Video file:      %(prog)s -i news.mp4
  Alert + log:     %(prog)s -i udp://@239.0.0.1:5500 -t 80 -o sync.log
  Debug verbose:   %(prog)s -i news.mp4 -v -w 5 -n 0.5
        ''',
    )

    p.add_argument(
        '-i', '--input', required=True,
        help='Input source: file path  OR  udp://@GROUP:PORT for multicast',
    )
    p.add_argument(
        '-w', '--window', type=float, default=3.0, metavar='SEC',
        help='Analysis window length in seconds (default: 3.0)',
    )
    p.add_argument(
        '-n', '--interval', type=float, default=1.0, metavar='SEC',
        help='How often to print a result, in seconds (default: 1.0)',
    )
    p.add_argument(
        '-t', '--threshold', type=float, default=80.0, metavar='MS',
        help='Alert threshold in ms — CRIT when |offset| ≥ threshold (default: 80)',
    )
    p.add_argument(
        '--face-confidence', type=float, default=0.5,
        help='MediaPipe min face-detection confidence, 0–1 (default: 0.5)',
    )
    p.add_argument(
        '--min-lip-variance', type=float, default=1e-4,
        help='Min variance of the lip signal to consider the person is speaking '
             '(default: 1e-4)',
    )
    p.add_argument(
        '--max-offset-search', type=float, default=500.0, metavar='MS',
        help='Maximum A/V offset to search for in ms (default: 500)',
    )
    p.add_argument(
        '--buffer-size', type=int, default=2_097_152,
        help='UDP receive buffer size in bytes (default: 2097152 = 2 MB)',
    )
    p.add_argument(
        '--timeout', type=int, default=5_000_000,
        help='Network read timeout in microseconds (default: 5000000 = 5 s)',
    )
    p.add_argument(
        '-o', '--output', metavar='FILE',
        help='Write all output lines to a log file as well',
    )
    p.add_argument(
        '-v', '--verbose', action='store_true',
        help='Print per-interval debug stats (frame count, face rate, buffer sizes)',
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _build_parser().parse_args()

    # Output contains Unicode (…, —); force UTF-8 so Windows consoles on a
    # legacy code page don't mangle it.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8')
        except (AttributeError, ValueError):
            pass

    logging.basicConfig(
        format='%(message)s',
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    log = logging.getLogger(__name__)

    # --- open source -------------------------------------------------------
    log.info(f'Opening: {args.input}')
    try:
        capture = StreamCapture(args.input, args.buffer_size, args.timeout)
    except Exception as exc:
        log.error(f'Cannot open source: {exc}')
        return 1

    mode = 'LIVE multicast' if capture.is_multicast else f'FILE  (duration {capture.duration:.1f}s)'
    log.info(
        f'Mode: {mode}  |  fps={capture.fps:.2f}  |  '
        f'audio_rate={capture.audio_sample_rate} Hz'
    )

    # --- init components ---------------------------------------------------
    lip_det  = LipDetector(args.face_confidence)
    audio_an = AudioAnalyzer()
    sync_det = SyncDetector(
        fps=capture.fps,
        window_sec=args.window,
        min_lip_variance=args.min_lip_variance,
        max_offset_ms=args.max_offset_search,
    )
    reporter = Reporter(
        threshold_ms=args.threshold,
        output_file=args.output,
        is_live=capture.is_multicast,
    )

    # --- graceful stop -----------------------------------------------------
    running = True

    def _stop(sig, frame):  # noqa: ANN001
        nonlocal running
        running = False

    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    # --- main loop ---------------------------------------------------------
    last_report_pts = -args.interval
    n_frames = n_faces = 0

    try:
        for kind, data, pts in capture.packets():
            if not running:
                break

            if kind == 'video':
                n_frames += 1
                val = lip_det.detect(data)
                if val is not None:
                    n_faces += 1
                sync_det.add_lip(pts, val)

                if pts - last_report_pts >= args.interval:
                    last_report_pts = pts
                    offset, conf, status = sync_det.compute()
                    reporter.report(pts, offset, conf, status)

                    if args.verbose and n_frames:
                        log.debug(
                            f'  frames={n_frames:4d}  '
                            f'face_rate={n_faces / n_frames * 100:4.0f}%  '
                            f'lip_buf={sync_det.lip_buf_len:4d}  '
                            f'audio_buf={sync_det.audio_buf_len:4d}'
                        )
                        n_frames = n_faces = 0

            elif kind == 'audio':
                sync_det.add_audio(pts, audio_an.rms(data))

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        log.error(f'Stream error: {exc}')
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    finally:
        lip_det.close()
        capture.close()
        reporter.close()
        log.info('Stopped.')

    return 0


if __name__ == '__main__':
    sys.exit(main())
