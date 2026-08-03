"""
Phase 7 - Real-World Validation: Session Logger
==================================================
Runs during actual vehicle test drives (daytime / nighttime / adverse
weather per the Phase 7 tasks). Logs every fusion-model tick alongside
an observer-entered ground-truth label, building the labeled dataset
that metrics.py then scores -- and that Phase 3/4 models get
fine-tuned/retrained on for "Improve overall model accuracy and
robustness."

Usage (during a test drive, observer in passenger seat):
    python validation_pipeline.py --session nighttime_2026-07-20 --condition nighttime

Observer presses:
    [0] normal   [1] drowsy episode   [2] distraction episode   [q] end session
while the pipeline keeps logging the live fusion score in the
background.
"""

import argparse
import csv
import os
import sys
import time
import threading

sys.path.append("..")
from common.schemas import FusionOutput  # noqa: E402

try:
    import msvcrt  # Windows
    _WINDOWS = True
except ImportError:
    import termios
    import tty
    _WINDOWS = False


def _read_key_nonblocking():
    """Cross-platform single-keypress read used for observer labeling."""
    if _WINDOWS:
        if msvcrt.kbhit():
            return msvcrt.getch().decode(errors="ignore")
        return None
    else:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            import select
            r, _, _ = select.select([sys.stdin], [], [], 0)
            if r:
                return sys.stdin.read(1)
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


class ValidationSession:
    def __init__(self, session_name: str, condition: str, out_dir: str = "sessions"):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, f"{session_name}.csv")
        self.condition = condition
        self._label = 0  # 0=normal, 1=drowsy, 2=distraction
        self._running = True
        self._rows = []

    def _label_listener(self):
        print("[*] Observer controls: [0]=normal [1]=drowsy [2]=distraction [q]=end session")
        while self._running:
            key = _read_key_nonblocking()
            if key in ("0", "1", "2"):
                self._label = int(key)
                print(f"[label] -> {self._label}")
            elif key == "q":
                self._running = False
            time.sleep(0.05)

    def log_tick(self, fusion_out: FusionOutput):
        self._rows.append({
            "timestamp": fusion_out.timestamp,
            "condition": self.condition,
            "drowsiness_score": fusion_out.drowsiness_score,
            "health_anomaly_score": fusion_out.health_anomaly_score,
            "distraction_score": fusion_out.distraction_score,
            "time_to_event_s": fusion_out.time_to_event_s,
            "observer_label": self._label,
        })

    def run(self, fusion_output_stream):
        """fusion_output_stream: an iterator/generator yielding FusionOutput
        objects in real time (wire this to your live Phase 4 model output)."""
        listener = threading.Thread(target=self._label_listener, daemon=True)
        listener.start()

        print(f"[*] Recording session '{self.path}' (condition={self.condition})")
        try:
            for fusion_out in fusion_output_stream:
                if not self._running:
                    break
                self.log_tick(fusion_out)
        finally:
            self._running = False
            self._save()

    def _save(self):
        if not self._rows:
            print("[!] No rows recorded.")
            return
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self._rows[0].keys()))
            writer.writeheader()
            writer.writerows(self._rows)
        print(f"[*] Saved {len(self._rows)} rows to {self.path}")


def simulated_fusion_stream(rate_hz: float = 2.0):
    """Stand-in for the live Phase 4 model output, for testing this
    script without the full pipeline running."""
    import random
    while True:
        yield FusionOutput(
            timestamp=time.time(),
            drowsiness_score=random.random(),
            health_anomaly_score=random.random() * 0.3,
            distraction_score=random.random(),
            time_to_event_s=None,
        )
        time.sleep(1.0 / rate_hz)


def main():
    ap = argparse.ArgumentParser(description="Phase 7: vehicle validation session logger")
    ap.add_argument("--session", required=True, help="Session name, e.g. nighttime_2026-07-20")
    ap.add_argument("--condition", required=True,
                     choices=["daytime", "nighttime", "rain", "fog", "other"])
    ap.add_argument("--duration-s", type=float, default=None,
                     help="Optional auto-stop duration for smoke-testing")
    args = ap.parse_args()

    session = ValidationSession(args.session, args.condition)
    stream = simulated_fusion_stream()

    if args.duration_s:
        # Non-interactive smoke test mode (used in CI / this repo's self-test)
        start = time.time()
        while time.time() - start < args.duration_s:
            session.log_tick(next(stream))
        session._save()
    else:
        session.run(stream)


if __name__ == "__main__":
    main()
