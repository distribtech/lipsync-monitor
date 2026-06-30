# lipsync-monitor

Real-time **audio/video desynchronisation detector** via lip-movement analysis.  
Supports **UDP multicast** (IPTV / DVB-IP) and **local video files**.  
Reports the A/V offset in milliseconds with a wall-clock or file-position timestamp.

---

## How it works — overview

```
Input (file / UDP multicast)
        │
        ▼
  ┌─────────────┐     ┌──────────────────────┐
  │ Video frame │────▶│  LipDetector          │──▶ lip_openness[t]  ─┐
  └─────────────┘     │  (MediaPipe FaceMesh) │                       │
                      └──────────────────────┘               ┌────────▼────────┐
  ┌─────────────┐     ┌──────────────────────┐               │  SyncDetector   │
  │ Audio frame │────▶│  AudioAnalyzer        │──▶ audio_rms[t] ──▶│ cross-corr  │──▶ offset_ms
  └─────────────┘     │  (NumPy RMS)          │               └─────────────────┘
                      └──────────────────────┘
```

---

## Algorithm in detail

### Step 1 — Lip openness signal

**Library:** [MediaPipe FaceMesh](https://google.github.io/mediapipe/solutions/face_mesh)

FaceMesh detects **468 3D facial landmarks** per frame in real time on CPU.  
Two inner-lip landmarks are used:

| Landmark | Location |
|---|---|
| **13** | Centre of the inner upper lip (top of mouth opening) |
| **14** | Centre of the inner lower lip (bottom of mouth opening) |

The **Euclidean distance** `‖lm[13] − lm[14]‖` is divided by the **inter-eye distance**
`‖lm[33] − lm[263]‖` to produce a scale-invariant scalar `lip_openness[t] ∈ [0, ∞)`.

```
lip_openness[t] = ‖pixel(13) − pixel(14)‖ / ‖pixel(33) − pixel(263)‖
```

This gives a signal that is:
- Close to **0** when the mouth is closed
- **Higher** when the mouth is open during speech
- **Independent** of camera distance / face size

When no face is detected in a frame the value is set to `0.0` and the frame
is flagged; if all frames in the window are `0.0` the status is `no_face`.

---

### Step 2 — Audio energy envelope

**Library:** PyAV + NumPy

For each decoded audio packet:
1. Convert to float32 planar (`fltp`) format → shape `(channels, samples)`.
2. Average channels → mono.
3. Compute **RMS** (root mean square):

```
audio_rms[t] = sqrt( mean( samples² ) )
```

This gives a per-packet energy scalar `audio_rms[t] ∈ [0, 1]` that rises during
speech and falls during silence — mirroring the lip openness signal.

---

### Step 3 — Cross-correlation and offset estimation

**Library:** `scipy.signal`

Both signals are collected in a **sliding window** (default 3 s) keyed by PTS
(presentation timestamp in seconds). At each report interval:

1. **Resample** both signals to a uniform time grid at video FPS using `np.interp`.
2. **Speech activity check** — if the lip signal variance is below
   `--min-lip-variance` the person is not speaking; skip the window.
3. **Z-score normalise** each signal (zero mean, unit variance).
4. **Cross-correlate** using `scipy.signal.correlate(lip_z, audio_z, mode='full')`.
5. Restrict the search to `±max_offset_ms` (default ±500 ms) to reject noise peaks.
6. The **lag at the peak** of the correlation is converted to milliseconds:

```
offset_ms = lag_frames / fps × 1000
```

**Why cross-correlation works:**  
When a person speaks, their lips open → high `lip_openness[t]`, and simultaneously
their voice produces energy → high `audio_rms[t]`.  
If audio and video are in sync, these two signals align at lag = 0.  
If the audio is 50 ms ahead of the video the correlation will peak at a lag
corresponding to +50 ms.

This is the same core idea as Oxford VGG's **SyncNet** paper  
("Out of Time: Automated Lip Sync in the Wild", Chung & Zisserman, ACCV 2016)  
but uses hand-crafted features instead of CNN embeddings — making it fast,
dependency-light, and interpretable.

---

### Sign convention

| `offset_ms` | Meaning |
|---|---|
| **positive** | **Audio leads video** — sound arrives *before* lips move on screen |
| **negative** | **Audio lags video** — sound arrives *after* lips move on screen |

---

### Acceptable limits — ITU-R BT.1359-1

| Direction | Limit |
|---|---|
| Audio lead | **+45 ms** |
| Audio lag  | **−125 ms** |

The `ITU:` column in the output shows `PASS` / `FAIL` relative to these limits.

---

## Models and libraries

| Component | Library / Model | Version |
|---|---|---|
| Facial landmark detection | [MediaPipe FaceMesh](https://google.github.io/mediapipe/solutions/face_mesh) | ≥ 0.10 |
| Stream decode (file + UDP) | [PyAV](https://pyav.org/) (FFmpeg bindings) | ≥ 10.0 |
| Cross-correlation | [SciPy](https://scipy.org/) `signal.correlate` | ≥ 1.11 |
| Image processing | [OpenCV](https://opencv.org/) (BGR→RGB conversion) | ≥ 4.8 |
| Numerics | [NumPy](https://numpy.org/) | ≥ 1.24 |

**Why MediaPipe FaceMesh and not SyncNet?**

| | MediaPipe + cross-corr | SyncNet (CNN) |
|---|---|---|
| Pre-trained weights needed | No | Yes (600 MB+) |
| GPU required | No | Recommended |
| Latency per frame | < 5 ms (CPU) | 20–50 ms (CPU) |
| Works on arbitrary faces | Yes | Yes |
| Interpretable output | Yes | Black-box score |
| Suitable for real-time broadcast | **Yes** | Harder |

---

## Installation

```bash
git clone <repo>
cd lipsync-monitor
pip install -r requirements.txt
```

**For multicast** — join the multicast group on your NIC first:

```bash
# Linux example
ip route add 239.0.0.0/8 dev eth0
```

---

## Usage

### Video file

```bash
python lipsync_monitor.py -i news_clip.mp4
```

### UDP multicast

```bash
python lipsync_monitor.py -i udp://@239.0.0.1:5500
```

### Multicast with 80 ms alert and log file

```bash
python lipsync_monitor.py -i udp://@239.0.0.1:5500 -t 80 -o sync.log
```

### Custom window and interval

```bash
# 5-second analysis window, report every 2 seconds
python lipsync_monitor.py -i video.mp4 -w 5 -n 2
```

### Verbose debug mode

```bash
python lipsync_monitor.py -i video.mp4 -v
```

---

## Parameters reference

| Flag | Default | Description |
|---|---|---|
| `-i, --input` | *(required)* | Source: file path **or** `udp://@GROUP:PORT` |
| `-w, --window SEC` | `3.0` | Sliding analysis window length in seconds |
| `-n, --interval SEC` | `1.0` | How often to print a result (seconds) |
| `-t, --threshold MS` | `80.0` | Alert threshold — `CRIT` when `|offset| ≥ threshold` |
| `--face-confidence` | `0.5` | MediaPipe min face-detection confidence (0.0 – 1.0) |
| `--min-lip-variance` | `1e-4` | Minimum lip signal variance to consider speech active |
| `--max-offset-search MS` | `500.0` | Maximum offset range to search (±ms) |
| `--buffer-size` | `2097152` | UDP receive buffer in bytes (2 MB) |
| `--timeout` | `5000000` | Network read timeout in microseconds (5 s) |
| `-o, --output FILE` | — | Also write output to a log file |
| `-v, --verbose` | — | Print per-interval debug stats |

---

## Output format

### File input — timestamp is the PTS position in the file

```
Opening: news.mp4
Mode: FILE  (duration 120.3s)  |  fps=25.00  |  audio_rate=48000 Hz
[00:00:00.000] Buffering — collecting samples…
[00:00:03.000]  Offset:   +12.0 ms  (audio leads)  |  OK    |  conf: 0.81  |  ITU: PASS
[00:00:04.000]  Offset:   +45.0 ms  (audio leads)  |  WARN  |  conf: 0.76  |  ITU: PASS
[00:00:05.000]  Offset:   +91.3 ms  (audio leads)  |  CRIT  |  conf: 0.88  |  ITU: FAIL
[00:00:06.000]  Offset:  -130.0 ms  (audio lags )  |  CRIT  |  conf: 0.79  |  ITU: FAIL
[00:00:07.000] No speech detected — lips are not moving
```

### Multicast (live) — timestamp is the system wall clock

```
Opening: udp://@239.0.0.1:5500
Mode: LIVE multicast  |  fps=25.00  |  audio_rate=48000 Hz
[2026-06-30 14:32:01] Buffering — collecting samples…
[2026-06-30 14:32:04]  Offset:   +12.0 ms  (audio leads)  |  OK    |  conf: 0.83  |  ITU: PASS
[2026-06-30 14:32:05]  Offset:   -87.5 ms  (audio lags )  |  CRIT  |  conf: 0.91  |  ITU: PASS
```

### Status codes

| Code | Meaning |
|---|---|
| `OK`   | `|offset| < threshold` |
| `WARN` | `|offset| ≥ 75 % of threshold` |
| `CRIT` | `|offset| ≥ threshold` |

### Confidence column

`conf` is the normalised peak value of the cross-correlation (0.0 – 1.0).

| Range | Interpretation |
|---|---|
| ≥ 0.7 | Reliable estimate |
| 0.4 – 0.7 | Moderate — person may be speaking quietly or partially off-camera |
| < 0.4 | Unreliable — discard or wait for more speech |

---

## Limitations

- Requires a **visible speaking face** in the frame.  
  Silence, music-only, voice-over (off-camera speaker), or heavy head rotation
  (> ~45°) will produce `No speech detected` or low confidence.

- The window size sets the **minimum detectable granularity**.  
  With a 3 s window the tool cannot detect offsets that last less than ~1 s.
  Use `--window 1.5` for faster response (lower accuracy).

- **Dubbed / re-voiced content** (e.g. foreign-language dub) will always show
  a large offset because lip movements do not match the audio by design.

- Very **noisy UDP networks** (heavy packet loss) may cause PyAV decode errors;
  these are silently skipped and only affect confidence.

---

## File structure

```
lipsync-monitor/
├── lipsync_monitor.py   — CLI entry point, main loop
├── requirements.txt
├── README.md
└── src/
    ├── capture.py       — PyAV stream/file reader
    ├── lip_detector.py  — MediaPipe FaceMesh lip-openness extraction
    ├── audio_analyzer.py — RMS energy per audio packet
    ├── sync_detector.py  — sliding-window cross-correlation, offset estimate
    └── reporter.py       — formatted stdout + optional log file output
```

---

## References

- J. S. Chung & A. Zisserman, *Out of Time: Automated Lip Sync in the Wild*,
  ACCV 2016 Workshop — [paper](https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/chung16a.pdf) |
  [code](https://github.com/joonson/syncnet_python)
- ITU-R BT.1359-1 — *Relative timing of sound and vision for broadcasting*
- MediaPipe FaceMesh — [documentation](https://google.github.io/mediapipe/solutions/face_mesh) |
  [landmark map](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png)
- UbiCast qr-lipsync — [github.com/UbiCastTeam/qr-lipsync](https://github.com/UbiCastTeam/qr-lipsync)
