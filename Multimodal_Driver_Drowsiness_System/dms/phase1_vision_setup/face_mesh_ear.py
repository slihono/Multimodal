"""
Phase 1 - System Setup
=======================
Objective: basic computer-vision pipeline for face detection + Face Mesh,
producing EAR (Eye Aspect Ratio) and PERCLOS (percentage of eye closure
over time) for baseline drowsiness detection.

Deliverable of this file: a Python script capable of detecting a face and
estimating drowsiness in real time from a webcam, an Intel RealSense
stream, or a video file.

Usage:
    python face_mesh_ear.py --source 0                 # webcam
    python face_mesh_ear.py --source video.mp4          # video file
    python face_mesh_ear.py --source 0 --headless        # no on-screen window

Requires: opencv-python, mediapipe, numpy
    pip install opencv-python mediapipe numpy --break-system-packages
"""

import argparse
import collections
import sys
import time

import numpy as np

try:
    import cv2
except ImportError:
    print("[!] opencv-python not installed. Run: pip install opencv-python --break-system-packages")
    sys.exit(1)

try:
    import mediapipe as mp
except ImportError:
    mp = None
    print("[!] mediapipe not installed. Run: pip install mediapipe --break-system-packages")

sys.path.append("..")
from common.schemas import VisionFrame, now  # noqa: E402


# MediaPipe FaceMesh landmark indices for eyes (6-point EAR model, Soukupová & Čech 2016)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 291, 39, 181, 0, 17]  # left, right, top-left, bottom-left, top, bottom (approx MAR)

EAR_CLOSED_THRESHOLD = 0.21     # below this, eye considered closed
PERCLOS_WINDOW_SEC = 60.0       # rolling window for PERCLOS
YAWN_MAR_THRESHOLD = 0.6


def _dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def eye_aspect_ratio(landmarks, idxs):
    p = [landmarks[i] for i in idxs]
    # p = [p1..p6] following the 6-point convention
    vertical1 = _dist(p[1], p[5])
    vertical2 = _dist(p[2], p[4])
    horizontal = _dist(p[0], p[3])
    if horizontal == 0:
        return 0.0
    return (vertical1 + vertical2) / (2.0 * horizontal)


def mouth_aspect_ratio(landmarks, idxs):
    p = [landmarks[i] for i in idxs]
    vertical = _dist(p[2], p[3])
    horizontal = _dist(p[0], p[1])
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


class DrowsinessEstimator:
    """Stateful EAR/PERCLOS estimator. Feed it frames; it maintains the
    rolling eye-closure history needed to compute PERCLOS."""

    def __init__(self, window_sec: float = PERCLOS_WINDOW_SEC):
        self.window_sec = window_sec
        self.history = collections.deque()  # (timestamp, is_closed)
        if mp is not None:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            self.face_mesh = None

    def _perclos(self, ts: float) -> float:
        while self.history and ts - self.history[0][0] > self.window_sec:
            self.history.popleft()
        if not self.history:
            return 0.0
        closed = sum(1 for _, c in self.history if c)
        return closed / len(self.history)

    def process(self, frame_bgr) -> VisionFrame:
        ts = now()
        if self.face_mesh is None:
            return VisionFrame(timestamp=ts, ear=0.3, perclos=0.0, face_found=False)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            self.history.append((ts, False))
            return VisionFrame(timestamp=ts, ear=0.0, perclos=self._perclos(ts), face_found=False)

        h, w = frame_bgr.shape[:2]
        lm = results.multi_face_landmarks[0].landmark
        pts = [(p.x * w, p.y * h) for p in lm]

        left_ear = eye_aspect_ratio(pts, LEFT_EYE)
        right_ear = eye_aspect_ratio(pts, RIGHT_EYE)
        ear = (left_ear + right_ear) / 2.0
        mar = mouth_aspect_ratio(pts, MOUTH)

        is_closed = ear < EAR_CLOSED_THRESHOLD
        self.history.append((ts, is_closed))
        perclos = self._perclos(ts)
        yawn_score = min(1.0, max(0.0, (mar - 0.3) / (YAWN_MAR_THRESHOLD - 0.3)))

        return VisionFrame(
            timestamp=ts,
            ear=ear,
            perclos=perclos,
            yawn_score=yawn_score,
            face_found=True,
        )

    def draw_overlay(self, frame_bgr, vf: VisionFrame):
        color = (0, 0, 255) if vf.perclos > 0.15 else (0, 200, 0)
        cv2.putText(frame_bgr, f"EAR: {vf.ear:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame_bgr, f"PERCLOS: {vf.perclos*100:.1f}%", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame_bgr, f"Yawn: {vf.yawn_score:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if vf.perclos > 0.15:
            cv2.putText(frame_bgr, "DROWSINESS ALERT", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        return frame_bgr


def main():
    ap = argparse.ArgumentParser(description="Phase 1: Face Mesh EAR/PERCLOS drowsiness estimator")
    ap.add_argument("--source", default="0", help="Camera index or video file path")
    ap.add_argument("--headless", action="store_true", help="Don't open a display window")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[!] Could not open video source: {source}")
        sys.exit(1)

    estimator = DrowsinessEstimator()
    print("[*] Phase 1 pipeline running. Press 'q' to quit (if not headless).")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            vf = estimator.process(frame)

            if not args.headless:
                frame = estimator.draw_overlay(frame, vf)
                cv2.imshow("Phase 1 - EAR / PERCLOS", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print(f"t={vf.timestamp:.2f} ear={vf.ear:.3f} perclos={vf.perclos:.3f} "
                      f"yawn={vf.yawn_score:.2f} face_found={vf.face_found}")
                time.sleep(0.03)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
