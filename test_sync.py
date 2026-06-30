"""
Ground-truth test of the delay-estimation math (SyncDetector).

We synthesise a shared "speech envelope" (a few mouth-open bursts), then feed
it as the lip signal at video FPS and as the audio-RMS signal at a higher rate,
with the audio deliberately shifted by a KNOWN offset. The tool must recover
that offset, with the correct sign.

Sign convention (per README / SyncNet):
    offset_ms > 0  -> audio leads video (sound arrives BEFORE lips move)
    offset_ms < 0  -> audio lags  video (sound arrives AFTER  lips move)

So if audio leads by +D ms, the lips move D ms LATER than the audio energy:
    audio_rms(t) = envelope(t)
    lip(t)       = envelope(t - D)
"""
import numpy as np
from src.sync_detector import SyncDetector

FPS = 25.0
AUDIO_HZ = 50.0          # ~ real packet rate after resampling
DURATION = 6.0           # seconds of signal
RNG = np.random.default_rng(42)


def envelope(t: np.ndarray) -> float | np.ndarray:
    """A few smooth speech-like bursts over time."""
    bursts = [(0.8, 0.15), (1.6, 0.2), (2.4, 0.15), (3.3, 0.25),
              (4.1, 0.18), (4.9, 0.2), (5.5, 0.15)]
    v = np.zeros_like(t, dtype=float)
    for c, w in bursts:
        v += np.exp(-((t - c) ** 2) / (2 * w ** 2))
    return v


def run_case(true_offset_ms: float) -> tuple[float, float]:
    """Returns (recovered_offset_ms, confidence) for a given true offset."""
    sd = SyncDetector(fps=FPS, window_sec=DURATION + 1, max_offset_ms=600)
    d = true_offset_ms / 1000.0

    # Video/lip samples at FPS: lips move D later than audio -> envelope(t - d)
    for i in range(int(DURATION * FPS)):
        t = i / FPS
        lip = float(envelope(np.array([t - d]))[0])
        lip += RNG.normal(0, 0.02)          # small sensor noise
        sd.add_lip(t, lip)

    # Audio samples at AUDIO_HZ: envelope(t)
    for i in range(int(DURATION * AUDIO_HZ)):
        t = i / AUDIO_HZ
        rms = float(envelope(np.array([t]))[0])
        rms += RNG.normal(0, 0.02)
        sd.add_audio(t, max(0.0, rms))

    offset, conf, status = sd.compute()
    return offset, conf


print(f"{'true (ms)':>10} | {'recovered':>10} | {'conf':>5} | {'err (ms)':>8}")
print("-" * 46)
ok = True
for true_ms in (0.0, 40.0, 120.0, -80.0, -200.0, 200.0):
    rec, conf = run_case(true_ms)
    if rec is None:
        print(f"{true_ms:10.0f} | {'None':>10} | {conf:5.2f} |   (no result)")
        ok = False
        continue
    err = rec - true_ms
    flag = "" if abs(err) <= 1000.0 / FPS else "  <-- off"
    print(f"{true_ms:10.0f} | {rec:10.1f} | {conf:5.2f} | {err:8.1f}{flag}")
    # within one frame (40 ms) of truth, and high confidence
    if abs(err) > 1000.0 / FPS + 1 or conf < 0.7:
        ok = False

print("-" * 46)
print("PASS" if ok else "FAIL — see rows marked off / low confidence")
