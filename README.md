# lipsync-monitor

*Read this in other languages: [Українська](README.uk.md).*

Real-time **audio/video desynchronisation detector** via lip-movement analysis.
Supports **UDP multicast** (IPTV / DVB-IP) and **local video files**, reports the
A/V offset in milliseconds, and can show a live preview window with the
detected face and lips plus sound.

Face detection and facial landmarks run on the **GPU** (NVIDIA, via
InsightFace + onnxruntime-CUDA), so it keeps up with Full-HD broadcast in real
time and detects small / non-frontal faces that lighter CPU detectors miss.

---

## How it works — overview

```
Input (file / UDP multicast)
        │
        ▼
  ┌─────────────┐     ┌──────────────────────────┐
  │ Video frame │────▶│  LipDetector (GPU)        │──▶ lip_openness[t] ─┐
  └─────────────┘     │  InsightFace SCRFD + 68pt │                     │
                      └──────────────────────────┘             ┌────────▼────────┐
  ┌─────────────┐     ┌──────────────────────────┐             │  SyncDetector   │
  │ Audio frame │────▶│  AudioAnalyzer (RMS)      │──▶ audio_rms[t] ─▶│  cross-corr  │─▶ offset_ms
  └─────────────┘     └──────────────────────────┘             └─────────────────┘
```

The capture side runs in a **background thread** that continuously drains the
UDP socket; if detection can't keep up, the oldest *video* frames are dropped
(audio kept), so the socket never overflows — this is what prevents macroblock
glitches on a live stream.

---

## Algorithm in detail

### Step 1 — Lip-openness signal (GPU)

**Model:** [InsightFace](https://github.com/deepinsight/insightface) `buffalo_l`
pack — **SCRFD** (`det_10g`) face detector + **1k3d68** 68-point 3D landmarks,
both executed by **onnxruntime** on the CUDA (GPU) provider, with automatic CPU
fallback.

For each frame:

1. SCRFD detects faces (default input resolution `--det-size 1024`). The most
   confident face is used.
2. The 68 facial landmarks are predicted for that face (dlib ordering).
3. The **lip openness** is the inner-lip gap normalised by the inter-eye
   distance, so it is scale-invariant:

```
lip_openness[t] = ‖p62 − p66‖ / ‖p36 − p45‖
```

| Landmark | Meaning |
|---|---|
| **62** | inner upper lip centre |
| **66** | inner lower lip centre |
| **36** | right-eye outer corner ┐ scale reference |
| **45** | left-eye outer corner  ┘ (inter-eye distance) |

When no face is detected the sample is `0.0`; if the whole window is `0.0` the
status is `no_face`.

*Why InsightFace and not MediaPipe?* MediaPipe's Python build is CPU-only on
Windows and its lightweight detector misses small / turned faces common in
broadcast (a presenter ~5 % of a 1080p frame, looking down at a product).
SCRFD on the GPU is far more robust and runs the full stream in real time.

### Step 2 — Audio energy envelope

**Library:** PyAV + NumPy. Each decoded audio packet is resampled to mono
float32 and reduced to a per-packet **RMS** energy:

```
audio_rms[t] = sqrt( mean( samples² ) )
```

This rises during speech and falls during silence, mirroring lip openness.

### Step 3 — Cross-correlation and offset estimate

**Library:** `scipy.signal`. Both signals are collected in a sliding window
(default 3 s) keyed by PTS. Each report interval:

1. **Resample** both signals to a uniform grid at video FPS (`np.interp`).
2. **Speech check** — if the lip variance is below `--min-lip-variance`, skip
   (`low_speech`).
3. **Z-score normalise** each signal.
4. **Cross-correlate** (`scipy.signal.correlate`), restricted to
   `±--max-offset-search`.
5. The **lag at the correlation peak** becomes the offset:

```
offset_ms = lag_frames / fps × 1000
```

Confidence is the normalised correlation peak (0–1).

### Sign convention

| `offset_ms` | Meaning |
|---|---|
| **positive** | **audio leads video** — sound arrives *before* the lips move |
| **negative** | **audio lags video** — sound arrives *after* the lips move |

### Acceptable limits — ITU-R BT.1359-1

| Direction | Limit |
|---|---|
| Audio lead | **+45 ms** |
| Audio lag  | **−125 ms** |

The `ITU:` column shows `PASS` / `FAIL` against these limits.

---

## Requirements

**Hardware (for GPU acceleration):** an NVIDIA GPU (tested on RTX 3090) with
**CUDA 12.x** and **cuDNN 9.x** installed and on `PATH`. Without a GPU it still
runs on CPU (much slower) via onnxruntime's CPU provider.

**Software:** Python 3.11 (Windows 10/11 tested). Python packages are pinned in
`requirements.txt`:

| Component | Library |
|---|---|
| Face detection + landmarks | `insightface` (SCRFD + 1k3d68) |
| Inference runtime | `onnxruntime-gpu==1.22.0` (CUDA 12 build) |
| Stream/file decode | `av` (PyAV / FFmpeg) |
| Cross-correlation | `scipy` |
| Image ops / preview | `opencv-python` |
| Numerics | `numpy<2` |
| Audio playback (optional) | `sounddevice` |

> **onnxruntime / CUDA matching.** `onnxruntime-gpu==1.22.0` targets **CUDA 12**
> + **cuDNN 9**. If you have CUDA 13, use a newer onnxruntime-gpu; to run on CPU
> only, replace it with plain `onnxruntime`. On Windows, Python 3.8+ does not
> search `PATH` for a DLL's dependencies, so `src/gpu_setup.py` registers the
> CUDA/cuDNN directories and preloads the runtime — otherwise it silently falls
> back to CPU.

---

## Installation

```powershell
git clone https://github.com/distribtech/lipsync-monitor
cd lipsync-monitor

python -m venv .venv
.\.venv\Scripts\Activate.ps1          # PowerShell
pip install -r requirements.txt
```

On the **first run** InsightFace downloads the `buffalo_l` model pack
(~280 MB) into `~/.insightface/models` automatically.

On Windows, set UTF-8 output so the console shows `…`, `—` correctly:

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

---

## Usage

```powershell
# Video file
python lipsync_monitor.py -i news_clip.mp4

# UDP multicast (IPTV)
python lipsync_monitor.py -i "udp://@228.0.3.89:1234"

# Live preview window + sound
python lipsync_monitor.py -i "udp://@228.0.3.89:1234" --show --audio

# Alert threshold 80 ms, log to file, verbose stats
python lipsync_monitor.py -i "udp://@228.0.3.89:1234" -t 80 -o sync.log -v
```

In the **preview window** (`--show`): **red box** = detected face,
**green box + points** = lips, cyan text = current offset/status. Press **q**
or **ESC** to quit.

---

## Parameters reference

| Flag | Default | Description |
|---|---|---|
| `-i, --input` | *(required)* | File path **or** `udp://@GROUP:PORT` |
| `-w, --window SEC` | `3.0` | Sliding analysis window length |
| `-n, --interval SEC` | `1.0` | How often to print a result |
| `-t, --threshold MS` | `80.0` | `CRIT` when `|offset| ≥ threshold` |
| `--face-confidence` | `0.3` | Min face-detection confidence (0–1) |
| `--det-size PX` | `1024` | Detector input resolution (larger = smaller faces, slower) |
| `--min-lip-variance` | `1e-4` | Min lip variance to treat as speech |
| `--max-offset-search MS` | `500.0` | Max offset to search (±ms) |
| `--buffer-size` | `16777216` | UDP receive buffer (bytes) |
| `--timeout` | `5000000` | Network read timeout (µs) |
| `-s, --show` | off | Live preview window with overlays |
| `--show-width PX` | `1280` | Preview window width |
| `-a, --audio` | off | Play the stream's audio (mono) |
| `-o, --output FILE` | — | Also write output to a log file |
| `-v, --verbose` | off | Per-interval debug stats |

---

## Output format

```
Mode: LIVE multicast  |  fps=25.00  |  audio_rate=48000 Hz
LipDetector: InsightFace buffalo_l on GPU (CUDA), det_size=1024
[2026-06-30 14:32:04]  Offset:   +12.0 ms  (audio leads)  |  OK    |  conf: 0.83  |  ITU: PASS
[2026-06-30 14:32:05]  Offset:   -87.5 ms  (audio lags )  |  CRIT  |  conf: 0.91  |  ITU: PASS
[2026-06-30 14:32:06] No face detected in frame
```

| Status | Meaning |
|---|---|
| `OK`   | `|offset| < threshold` |
| `WARN` | `|offset| ≥ 75 % of threshold` |
| `CRIT` | `|offset| ≥ threshold` |

**Confidence** (`conf`) is the normalised cross-correlation peak: ≥ 0.7 reliable,
0.4–0.7 moderate, < 0.4 discard. A confident reading needs a speaking face on
screen for at least ~one analysis window.

---

## Limitations

- Needs a **visible speaking face**. Silence, music-only, voice-over, or heavy
  head rotation give `no_face` / `low_speech` / low confidence.
- Bursty content (a face on screen < ~3 s at a time) rarely reaches high
  confidence — a sustained talking head gives the cleanest reading.
- **Dubbed** content shows a large offset by design (lips never match audio).
- Offset resolution is one video frame (40 ms at 25 fps).

---

## File structure

```
lipsync-monitor/
├── lipsync_monitor.py    — CLI entry point, main loop
├── requirements.txt
├── test_sync.py          — offset-math regression test
└── src/
    ├── capture.py        — PyAV reader; threaded UDP drain + drop-on-overload
    ├── gpu_setup.py      — register CUDA/cuDNN DLLs so onnxruntime finds the GPU
    ├── lip_detector.py   — InsightFace face+landmark detection, lip openness
    ├── audio_analyzer.py — per-packet RMS energy
    ├── sync_detector.py  — sliding-window cross-correlation, offset estimate
    ├── viewer.py         — optional OpenCV preview window with overlays
    ├── audio_player.py   — optional sounddevice speaker playback
    └── reporter.py       — formatted stdout + optional log file
```

---

## References

- J. S. Chung & A. Zisserman, *Out of Time: Automated Lip Sync in the Wild*,
  ACCV 2016 — [paper](https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/chung16a.pdf)
- ITU-R BT.1359-1 — *Relative timing of sound and vision for broadcasting*
- InsightFace — [github.com/deepinsight/insightface](https://github.com/deepinsight/insightface)
- SCRFD — *Sample and Computation Redistribution for Efficient Face Detection*
