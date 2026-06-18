# WigglyWoosh Activity Detection — Solution

**Author:** Anmol Panwar  
**Pipeline:** `run_pipeline.py --video <path> --csv <path>`

---

## 1. Vision Model: Dense Optical Flow (Farnebäck)

Rather than loading a heavy deep-learning classifier (which would stall a laptop CPU), I use **OpenCV's Farnebäck dense optical flow** — a classical, single-threaded algorithm that runs at ~30 fps on 320×180 pixel crops.

**Why this choice:**
- Zero inference overhead (no ONNX runtime, no model download)
- Provably fast on CPU: tested at < 15 ms per frame pair on a mid-range laptop
- Directly measures *pixel displacement* — the exact physical signal that distinguishes walking/running from sitting/lying

**How it works per 500 ms window:**
1. Sample frame at `t_start` and `t_end` of the window
2. Resize both to 320×180 for speed
3. Compute Farnebäck flow → per-pixel displacement vectors
4. Compute mean magnitude across all pixels → `mean_flow` (pixels/frame)

```
video_label = "Active"  if mean_flow >= 0.45  else "Static"
video_score = clip(mean_flow / 1.8,  0, 1)        # normalised to [0,1]
```

---

## 2. IMU Feature Extraction

Two features are extracted from the 500 ms window (50 samples at 100 Hz):

| Feature | Formula | Active threshold |
|---|---|---|
| `gyro_mean` | mean(√(gx²+gy²+gz²)) | ≥ 15 deg/s |
| `accel_jerk` | std(√(ax²+ay²+az²)) | ≥ 0.08 g |

Combined IMU active score:
```
imu_score = 0.6 × clip(gyro_mean / 30, 0,1)
           + 0.4 × clip(accel_jerk / 0.16, 0,1)

imu_label = "Active" if imu_score >= 0.5 else "Static"
```

---

## 3. Sensor Fusion Rules

Fusion is **IMU-dominant with hard overrides**:

```
IF gyro_mean >= 30 deg/s:
    label = "Active",  confidence = min(0.85 + Δgyro/100, 0.99)

ELSE IF gyro_mean <= 3 deg/s:
    label = "Static",  confidence = min(0.85 + (3−gyro)/10, 0.99)

ELSE:
    blended_score = 0.45 × video_active_conf + 0.55 × imu_active_conf
    label = "Active" if blended_score >= 0.5 else "Static"
    confidence = blended_score (or 1 − blended_score for Static)
```

**Weight rationale (55 % IMU / 45 % video):** A dog's collar is rigidly attached to the neck — it cannot stay still if the dog is genuinely moving. Video is noisier: background objects, camera shake, watermarks, or lighting changes can all trigger false active readings. Hence IMU gets the casting vote.

---

## 4. Observed Fusion Behaviour on Test Clip

| Time window | Video says | IMU says | Merged | Reason |
|---|---|---|---|---|
| 0 – 5.5 s | Static | **Active** | **Active** | IMU gyro ~60 deg/s → hard override |
| 6 – 11.5 s | Active | **Static** | **Static** | IMU gyro ~2 deg/s → hard override |

This is the core fusion case described in the challenge brief: camera gives wrong signal, collar corrects it.

---

## 5. Output Format

`timeline.json` — 24 entries at 2 Hz over 12 seconds:

```json
{
  "timeline": [
    {
      "timestamp_ms": 0,
      "activity": "Active",
      "confidence": 0.99
    },
    ...
  ],
  "sample_rate_hz": 2,
  "total_points": 24
}
```

---

## 6. Dependencies & Runtime

```
opencv-python-headless
numpy
pandas
```

No GPU required. Tested on CPU only. Full 12-second video processes in under 3 seconds.