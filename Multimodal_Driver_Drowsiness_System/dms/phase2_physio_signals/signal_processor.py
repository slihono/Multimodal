"""
Phase 2 - Physiological Signal Acquisition (PC side)
======================================================
Reads the JSON-line stream produced by the ESP32-S3 firmware
(main.ino) over UART (or MQTT), applies Butterworth + moving-average
filtering, and yields synchronized PhysioSample objects with accurate
timestamps -- the Phase 2 deliverable.

Two input modes:
  - UART:  --port /dev/ttyUSB0 --baud 115200
  - MQTT:  --mqtt-host 192.168.1.100 --mqtt-topic driver/physio
  - Simulation (no hardware attached): --simulate

Requires: pyserial, paho-mqtt, scipy, numpy
    pip install pyserial paho-mqtt scipy numpy --break-system-packages
"""

import argparse
import json
import sys
import time
import collections
import random

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

sys.path.append("..")
from common.schemas import PhysioSample, now  # noqa: E402


class ButterworthFilter:
    """Streaming (stateful) low-pass Butterworth filter, used to clean the
    PPG/GSR signal of high-frequency noise before feeding Phase 4."""

    def __init__(self, cutoff_hz: float, fs_hz: float, order: int = 3):
        nyq = 0.5 * fs_hz
        normal_cutoff = cutoff_hz / nyq
        self.b, self.a = butter(order, normal_cutoff, btype="low", analog=False)
        self.zi = lfilter_zi(self.b, self.a)
        self._initialized = False

    def filter(self, sample: float) -> float:
        if not self._initialized:
            self.zi = self.zi * sample
            self._initialized = True
        y, self.zi = lfilter(self.b, self.a, [sample], zi=self.zi)
        return float(y[0])


class MovingAverage:
    def __init__(self, window: int = 8):
        self.buf = collections.deque(maxlen=window)

    def filter(self, sample: float) -> float:
        self.buf.append(sample)
        return sum(self.buf) / len(self.buf)


class PhysioSignalProcessor:
    """Consumes raw sensor dicts (as decoded from the ESP32 JSON) and
    produces filtered, synchronized PhysioSample objects."""

    def __init__(self, sample_rate_hz: float = 25.0):
        self.hr_filter = MovingAverage(window=5)
        self.gsr_filter = ButterworthFilter(cutoff_hz=0.5, fs_hz=sample_rate_hz)
        self.spo2_filter = MovingAverage(window=5)

    def process(self, raw: dict) -> PhysioSample:
        hr = raw.get("hr_bpm")
        gsr = raw.get("gsr_raw")
        spo2 = self._estimate_spo2(raw.get("ir_raw"), raw.get("red_raw"))

        return PhysioSample(
            timestamp=now(),
            hr_bpm=self.hr_filter.filter(hr) if hr else None,
            spo2_pct=self.spo2_filter.filter(spo2) if spo2 else None,
            gsr_raw=self.gsr_filter.filter(gsr) if gsr is not None else None,
            accel_xyz=(raw.get("ax", 0.0), raw.get("ay", 0.0), raw.get("az", 0.0)),
            gyro_xyz=(raw.get("gx", 0.0), raw.get("gy", 0.0), raw.get("gz", 0.0)),
        )

    @staticmethod
    def _estimate_spo2(ir_raw, red_raw) -> float:
        """Simplified ratio-of-ratios SpO2 estimate (empirical calibration
        curve). For production use, replace with the calibrated lookup
        table from the MAX30102 datasheet / SparkFun spo2_algorithm."""
        if not ir_raw or not red_raw or ir_raw == 0:
            return None
        ratio = red_raw / ir_raw
        spo2 = 110.0 - 25.0 * ratio
        return float(np.clip(spo2, 70.0, 100.0))


# ---------------------------------------------------------------- sources --

def stream_from_uart(port: str, baud: int):
    import serial
    ser = serial.Serial(port, baud, timeout=1)
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def stream_from_mqtt(host: str, topic: str, port: int = 1883):
    import paho.mqtt.client as mqtt
    q = collections.deque()

    def on_message(client, userdata, msg):
        try:
            q.append(json.loads(msg.payload.decode()))
        except json.JSONDecodeError:
            pass

    client = mqtt.Client()
    client.on_message = on_message
    client.connect(host, port)
    client.subscribe(topic)
    client.loop_start()
    while True:
        if q:
            yield q.popleft()
        else:
            time.sleep(0.01)


def stream_simulated(rate_hz: float = 25.0):
    """Synthetic sensor stream for development / CI without hardware."""
    t = 0.0
    hr_base = 72.0
    while True:
        hr = hr_base + random.uniform(-2, 2) + 3 * np.sin(t / 20.0)
        yield {
            "t": t * 1000,
            "hr_bpm": max(45, min(150, hr)),
            "ir_raw": 50000 + random.uniform(-500, 500),
            "red_raw": 40000 + random.uniform(-500, 500),
            "gsr_raw": 300 + random.uniform(-10, 10) + 20 * max(0, np.sin(t / 60.0)),
            "ax": random.uniform(-0.05, 0.05),
            "ay": random.uniform(-0.05, 0.05),
            "az": 1.0 + random.uniform(-0.02, 0.02),
            "gx": random.uniform(-1, 1),
            "gy": random.uniform(-1, 1),
            "gz": random.uniform(-1, 1),
        }
        t += 1.0 / rate_hz
        time.sleep(1.0 / rate_hz)


def main():
    ap = argparse.ArgumentParser(description="Phase 2: physiological signal processor")
    ap.add_argument("--port", help="Serial port, e.g. /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--mqtt-host")
    ap.add_argument("--mqtt-topic", default="driver/physio")
    ap.add_argument("--simulate", action="store_true", help="Use synthetic sensor data")
    args = ap.parse_args()

    if args.simulate:
        source = stream_simulated()
    elif args.port:
        source = stream_from_uart(args.port, args.baud)
    elif args.mqtt_host:
        source = stream_from_mqtt(args.mqtt_host, args.mqtt_topic)
    else:
        print("[!] Specify --simulate, --port, or --mqtt-host")
        sys.exit(1)

    processor = PhysioSignalProcessor()
    print("[*] Phase 2 signal processor running. Ctrl+C to stop.")
    try:
        for raw in source:
            sample = processor.process(raw)
            print(f"t={sample.timestamp:.2f} HR={sample.hr_bpm} SpO2={sample.spo2_pct} "
                  f"GSR={sample.gsr_raw} accel={sample.accel_xyz}")
    except KeyboardInterrupt:
        print("\n[*] Stopped.")


if __name__ == "__main__":
    main()
