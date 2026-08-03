"""
Phase 3 - Advanced Computer Vision: Remote PPG (rPPG)
========================================================
Extracts heart rate remotely from facial video using the classic CHROM
method (de Haan & Jeanne, 2013), which is robust, lightweight (no
training needed) and appropriate for real-time edge inference -- a good
complement to the contact MAX30102 sensor from Phase 2, and a fallback
when the driver's finger isn't on the steering-wheel sensor.

Requires: opencv-python, numpy, scipy
    pip install opencv-python numpy scipy --break-system-packages
"""

import collections
import sys

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

sys.path.append("..")
from common.schemas import now  # noqa: E402

# ROI: forehead + cheeks give the strongest, least motion-affected PPG signal
FOREHEAD_LANDMARKS = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323]


def _bandpass(signal, fs, low=0.7, high=4.0, order=3):
    """0.7-4.0 Hz corresponds to 42-240 BPM, the physiological HR range."""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


class RPPGExtractor:
    """Streaming CHROM-method rPPG extractor. Feed it (roi_bgr, timestamp)
    pairs; call estimate_hr() periodically (e.g. every 1s) once enough
    samples have accumulated (needs >= ~4s of frames for a stable FFT)."""

    def __init__(self, buffer_seconds: float = 8.0, expected_fps: float = 30.0):
        self.buffer_size = int(buffer_seconds * expected_fps)
        self.rgb_means = collections.deque(maxlen=self.buffer_size)
        self.timestamps = collections.deque(maxlen=self.buffer_size)

    def add_frame(self, face_roi_bgr: np.ndarray, timestamp: float = None):
        if face_roi_bgr.size == 0:
            return
        b, g, r = cv2_mean_bgr(face_roi_bgr)
        self.rgb_means.append((r, g, b))
        self.timestamps.append(timestamp if timestamp is not None else now())

    def estimate_hr(self) -> float:
        if len(self.rgb_means) < 30:
            return None

        arr = np.array(self.rgb_means)  # (N, 3) columns = R, G, B
        r, g, b = arr[:, 0], arr[:, 1], arr[:, 2]

        # Normalize each channel
        r_n, g_n, b_n = r / r.mean(), g / g.mean(), b / b.mean()

        # CHROM projection (de Haan & Jeanne 2013)
        x = 3 * r_n - 2 * g_n
        y = 1.5 * r_n + g_n - 1.5 * b_n
        alpha = np.std(x) / (np.std(y) + 1e-8)
        chrom_signal = x - alpha * y

        ts = np.array(self.timestamps)
        duration = ts[-1] - ts[0]
        if duration <= 0:
            return None
        fs = len(ts) / duration

        try:
            filtered = _bandpass(chrom_signal, fs)
        except ValueError:
            return None  # not enough samples for the filter order at this fs

        peaks, _ = find_peaks(filtered, distance=fs * 60.0 / 200.0)  # cap at 200 bpm
        if len(peaks) < 2:
            return None
        intervals_s = np.diff(ts[peaks])
        if len(intervals_s) == 0 or np.mean(intervals_s) == 0:
            return None
        hr_bpm = 60.0 / np.mean(intervals_s)
        return float(np.clip(hr_bpm, 40, 220))


def cv2_mean_bgr(roi_bgr):
    return roi_bgr[:, :, 0].mean(), roi_bgr[:, :, 1].mean(), roi_bgr[:, :, 2].mean()


if __name__ == "__main__":
    # Smoke test with a synthetic pulsatile signal embedded in random noise.
    extractor = RPPGExtractor(buffer_seconds=8.0, expected_fps=30.0)
    fs = 30.0
    true_hr = 75.0
    t0 = now()
    for i in range(int(8 * fs)):
        t = i / fs
        pulse = 3 * np.sin(2 * np.pi * (true_hr / 60.0) * t)
        base = 150 + pulse
        fake_roi = np.clip(np.random.randn(20, 20, 3) * 2 + base, 0, 255).astype(np.uint8)
        extractor.add_frame(fake_roi, timestamp=t0 + t)
    estimated = extractor.estimate_hr()
    print(f"[*] True HR ~{true_hr} bpm, rPPG estimate: {estimated}")
