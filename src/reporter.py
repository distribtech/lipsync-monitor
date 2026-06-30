"""
Result formatting and output (stdout + optional log file).
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from typing import Optional

# ITU-R BT.1359-1 acceptable synchronisation limits
_ITU_LEAD_LIMIT =  45.0    # ms — max audio lead
_ITU_LAG_LIMIT  = -125.0   # ms — max audio lag


class Reporter:
    """Prints one result line per report interval."""

    def __init__(
        self,
        threshold_ms: float = 80.0,
        output_file: Optional[str] = None,
        is_live: bool = False,
    ) -> None:
        self.threshold_ms = threshold_ms
        self.is_live      = is_live
        self._fh          = open(output_file, 'w', buffering=1) if output_file else None

    # ------------------------------------------------------------------

    def report(
        self,
        pts: float,
        offset_ms: Optional[float],
        confidence: float,
        status: str,
    ) -> None:
        ts = self._timestamp(pts)

        if offset_ms is None:
            messages = {
                'insufficient_data': 'Buffering — collecting samples…',
                'no_face':           'No face detected in frame',
                'low_speech':        'No speech detected — lips are not moving',
            }
            line = f'[{ts}] {messages.get(status, status)}'
        else:
            direction = 'audio leads' if offset_ms >= 0 else 'audio lags '
            alert     = self._alert(offset_ms)
            itu       = 'PASS' if _ITU_LAG_LIMIT <= offset_ms <= _ITU_LEAD_LIMIT else 'FAIL'
            line = (
                f'[{ts}]  '
                f'Offset: {offset_ms:+7.1f} ms  ({direction})  |  '
                f'{alert:4s}  |  '
                f'conf: {confidence:.2f}  |  '
                f'ITU: {itu}'
            )

        print(line, flush=True)
        if self._fh:
            self._fh.write(line + '\n')

    def close(self) -> None:
        if self._fh:
            self._fh.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _timestamp(self, pts: float) -> str:
        if self.is_live:
            return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        td    = timedelta(seconds=pts)
        total = int(td.total_seconds())
        h, r  = divmod(total, 3600)
        m, s  = divmod(r, 60)
        ms    = int(round((td.total_seconds() - total) * 1000))
        return f'{h:02d}:{m:02d}:{s:02d}.{ms:03d}'

    def _alert(self, offset_ms: float) -> str:
        abs_ms = abs(offset_ms)
        if abs_ms >= self.threshold_ms:
            return 'CRIT'
        if abs_ms >= self.threshold_ms * 0.75:
            return 'WARN'
        return 'OK'
