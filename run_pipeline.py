

import argparse
import json
import sys
import numpy as np
import pandas as pd
import cv2



# THRESHOLDS  (tuned on the provided 12-second sample)

OPTICAL_FLOW_ACTIVE_THRESH = 0.45  
GYRO_ACTIVE_THRESH         = 15.0   
ACCEL_JERK_ACTIVE_THRESH   = 0.08   

# Sensor-fusion weights (must sum to 1.0)
VIDEO_WEIGHT = 0.45
IMU_WEIGHT   = 0.55    


# ─────────────────────────────────────────────────────────────────────────────
# IMU PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def load_imu(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"timestamp_ms", "accel_x", "accel_y", "accel_z",
                "gyro_x", "gyro_y", "gyro_z"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Missing IMU columns: {missing}")

    df["accel_mag"] = np.sqrt(df["accel_x"]**2 + df["accel_y"]**2 + df["accel_z"]**2)
    df["gyro_mag"]  = np.sqrt(df["gyro_x"]**2  + df["gyro_y"]**2  + df["gyro_z"]**2)
    return df


def imu_score_at(df: pd.DataFrame, t_ms: int, window_ms: int = 500) -> dict:
    """
    Returns a dict with:
      - imu_label  : 'Active' or 'Static'
      - imu_conf   : float in [0, 1]
      - gyro_mean  : raw gyro magnitude
      - accel_jerk : std of accel magnitude
    """
    win = df[(df["timestamp_ms"] >= t_ms) & (df["timestamp_ms"] < t_ms + window_ms)]
    if win.empty:
        return {"imu_label": "Static", "imu_conf": 0.5, "gyro_mean": 0.0, "accel_jerk": 0.0}

    gyro_mean  = win["gyro_mag"].mean()
    accel_jerk = win["accel_mag"].std()

    # Normalised scores (sigmoid-like clamp to [0,1])
    gyro_score  = min(gyro_mean  / (GYRO_ACTIVE_THRESH  * 2), 1.0)
    jerk_score  = min(accel_jerk / (ACCEL_JERK_ACTIVE_THRESH * 2), 1.0)
    imu_active_score = 0.6 * gyro_score + 0.4 * jerk_score   # weighted blend

    imu_label = "Active" if imu_active_score >= 0.5 else "Static"
    imu_conf  = imu_active_score if imu_label == "Active" else 1.0 - imu_active_score

    return {
        "imu_label":   imu_label,
        "imu_conf":    round(float(imu_conf), 4),
        "gyro_mean":   round(float(gyro_mean), 4),
        "accel_jerk":  round(float(accel_jerk), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO PROCESSING – lightweight dense optical flow (Farnebäck, CPU-only)
# ─────────────────────────────────────────────────────────────────────────────
def video_score_at(cap: cv2.VideoCapture, fps: float,
                   t_ms: int, window_ms: int = 500) -> dict:
    """
    Samples two frames (start and end of the window) and computes
    Farnebäck optical flow.  Returns label + confidence.
    """
    t_start_s = t_ms / 1000.0
    t_end_s   = (t_ms + window_ms) / 1000.0

    def read_frame(t_sec):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_sec * fps))
        ret, frame = cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray1 = read_frame(t_start_s)
    gray2 = read_frame(t_end_s)

    if gray1 is None or gray2 is None:
        return {"video_label": "Static", "video_conf": 0.5, "mean_flow": 0.0}

    # Resize to 320×180 for speed on CPU
    small1 = cv2.resize(gray1, (320, 180))
    small2 = cv2.resize(gray2, (320, 180))

    flow = cv2.calcOpticalFlowFarneback(
        small1, small2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mean_flow = float(mag.mean())

    # Normalised to [0, 1]
    video_active_score = min(mean_flow / (OPTICAL_FLOW_ACTIVE_THRESH * 4), 1.0)
    video_label = "Active" if mean_flow >= OPTICAL_FLOW_ACTIVE_THRESH else "Static"
    video_conf  = video_active_score if video_label == "Active" else 1.0 - video_active_score

    return {
        "video_label": video_label,
        "video_conf":  round(float(video_conf), 4),
        "mean_flow":   round(mean_flow, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FUSION LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def fuse(video: dict, imu: dict) -> tuple[str, float]:
    """
    Weighted confidence fusion with IMU-override rule.

    Override rules (in priority order):
    1. IMU gyro very high  → force Active  (collar always wins for motion proof)
    2. IMU gyro very low   → force Static  (collar always wins for rest proof)
    3. Otherwise           → weighted blend of video + imu confidence scores

    Returns: (final_label, final_confidence)
    """
    gyro  = imu["gyro_mean"]
    video_active = 1.0 if video["video_label"] == "Active" else 0.0
    imu_active   = 1.0 if imu["imu_label"]    == "Active" else 0.0

    video_active_conf = video["video_conf"] * video_active + (1 - video["video_conf"]) * (1 - video_active)
    imu_active_conf   = imu["imu_conf"]     * imu_active   + (1 - imu["imu_conf"])     * (1 - imu_active)

    # Hard overrides
    if gyro >= GYRO_ACTIVE_THRESH * 2:    # very strong IMU signal → Active
        label = "Active"
        conf  = min(0.85 + (gyro - GYRO_ACTIVE_THRESH * 2) / 100, 0.99)
        return label, round(conf, 4)

    if gyro <= 3.0:                        # dog is very still → Static
        label = "Static"
        conf  = min(0.85 + (3.0 - gyro) / 10, 0.99)
        return label, round(conf, 4)

    # Weighted blend
    blended_active_score = VIDEO_WEIGHT * video_active_conf + IMU_WEIGHT * imu_active_conf
    label = "Active" if blended_active_score >= 0.5 else "Static"
    conf  = blended_active_score if label == "Active" else 1.0 - blended_active_score

    return label, round(float(conf), 4)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(video_path: str, csv_path: str) -> list[dict]:
    print(f"[INFO] Loading IMU data from: {csv_path}")
    imu_df = load_imu(csv_path)
    max_imu_ms = int(imu_df["timestamp_ms"].max())

    print(f"[INFO] Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_ms  = int((total_frames / fps) * 1000)

    # Use IMU duration as ground truth (12 000 ms)
    end_ms   = max_imu_ms
    step_ms  = 500    # 2 Hz

    print(f"[INFO] Video: {fps:.2f} fps, {duration_ms} ms")
    print(f"[INFO] IMU:   {len(imu_df)} samples, {max_imu_ms} ms")
    print(f"[INFO] Generating timeline at 2 Hz ({end_ms // step_ms} points)…")

    timeline = []
    for t_ms in range(0, end_ms, step_ms):
        vid = video_score_at(cap, fps, t_ms)
        imu = imu_score_at(imu_df, t_ms)
        label, conf = fuse(vid, imu)

        entry = {
            "timestamp_ms":     t_ms,
            "activity":         label,
            "confidence":       conf,
            # Debug fields (remove if submission needs to be minimal)
            "_video_label":     vid["video_label"],
            "_video_conf":      vid["video_conf"],
            "_mean_flow":       vid["mean_flow"],
            "_imu_label":       imu["imu_label"],
            "_imu_conf":        imu["imu_conf"],
            "_gyro_mean":       imu["gyro_mean"],
            "_accel_jerk":      imu["accel_jerk"],
        }
        timeline.append(entry)

        status = f"[{t_ms:>6} ms]  video={vid['video_label']:<7} imu={imu['imu_label']:<7}  => {label}  conf={conf:.2f}"
        print(status)

    cap.release()
    return timeline


def main():
    parser = argparse.ArgumentParser(
        description="WigglyWoosh Dog Activity Detection Pipeline"
    )
    parser.add_argument("--video", required=True, help="Path to dog_video.mp4")
    parser.add_argument("--csv",   required=True, help="Path to collar_imu.csv")
    parser.add_argument("--output", default="timeline.json", help="Output JSON path")
    args = parser.parse_args()

    timeline = run_pipeline(args.video, args.csv)

    # Write output JSON
    out = {"timeline": timeline, "sample_rate_hz": 2, "total_points": len(timeline)}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[DONE] Wrote {len(timeline)} data points → {args.output}")

    # Quick summary
    active_count = sum(1 for e in timeline if e["activity"] == "Active")
    static_count = len(timeline) - active_count
    print(f"[SUMMARY] Active={active_count} ({active_count*500/10:.0f}%)  Static={static_count} ({static_count*500/10:.0f}%)")


if __name__ == "__main__":
    main()
