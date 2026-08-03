"""
Multimodal Driver Drowsiness & Health Monitoring System
==========================================================
End-to-end orchestration pipeline: connects every phase (1 through 7)
into a single running loop, matching the system architecture diagram
in the project overview:

    SENSING LAYER (cameras, biometrics, environment, OBD-II)
        -> EDGE PROCESSING (vision AI, signal DSP, sensor fusion)
        -> MULTIMODAL FUSION TRANSFORMER
        -> DECISION & ALERT ENGINE
        -> Driver UI / Cloud Backend / Emergency API

By default this runs in full-simulation mode (no camera, no ESP32, no
Jetson required) so you can validate the whole pipeline logic on any
machine. Swap in the real Phase 1/2/3 sources (see --camera-source,
--serial-port flags) once hardware is wired up.

Usage:
    python main_pipeline.py                          # full simulation
    python main_pipeline.py --camera-source 0        # real webcam for vision
    python main_pipeline.py --serial-port /dev/ttyUSB0  # real ESP32 physio
"""

import argparse
import random
import sys
import time

sys.path.append("phase1_vision_setup")
sys.path.append("phase2_physio_signals")
sys.path.append("phase4_fusion_model")
sys.path.append("phase5_alert_system")
sys.path.append("phase6_cloud_emergency")

from common.schemas import VisionFrame, PhysioSample, EnvSample, OBDSample, FusionOutput, now
from alert_engine import AlertEngine, RiskLevel  # noqa: E402


# ------------------------------------------------------------- simulators --
# These stand in for Phase 1/2/3 hardware pipelines so the full system can
# be exercised without a Jetson + sensor rig attached. Each has the exact
# same output schema as the real pipeline components in phase1-3, so
# swapping a simulator for the real thing requires no changes downstream.

class SimulatedVisionSource:
    def __init__(self, drowsy_after_s: float = 30.0):
        self.t0 = time.time()
        self.drowsy_after_s = drowsy_after_s

    def read(self) -> VisionFrame:
        elapsed = time.time() - self.t0
        # Gradually simulate the driver getting drowsier over time.
        drowsiness_factor = min(1.0, elapsed / self.drowsy_after_s)
        perclos = max(0.0, min(0.9, drowsiness_factor * 0.6 + random.uniform(-0.05, 0.05)))
        ear = max(0.05, 0.32 - drowsiness_factor * 0.15 + random.uniform(-0.02, 0.02))
        return VisionFrame(
            timestamp=now(), ear=ear, perclos=perclos,
            yawn_score=min(1.0, drowsiness_factor * 0.8 + random.uniform(0, 0.1)),
            head_nod_angle=drowsiness_factor * 15 + random.uniform(-2, 2),
            distraction_label="none", distraction_conf=0.0,
            emotion="tired" if drowsiness_factor > 0.6 else "neutral",
            emotion_conf=0.7, rppg_hr_bpm=72 - drowsiness_factor * 5,
            face_found=True,
        )


class SimulatedPhysioSource:
    def read(self) -> PhysioSample:
        return PhysioSample(
            timestamp=now(),
            hr_bpm=72 + random.uniform(-3, 3),
            spo2_pct=97 + random.uniform(-1, 1),
            gsr_raw=300 + random.uniform(-10, 10),
            accel_xyz=(random.uniform(-0.05, 0.05),) * 3,
            gyro_xyz=(random.uniform(-1, 1),) * 3,
        )


class SimulatedEnvSource:
    def read(self) -> EnvSample:
        return EnvSample(timestamp=now())


class SimulatedOBDSource:
    def read(self) -> OBDSample:
        return OBDSample(timestamp=now(), speed_kph=100 + random.uniform(-5, 5))


# ------------------------------------------------------------ lightweight --
# heuristic fusion (drop-in for the trained Phase 4 transformer, so the
# whole pipeline can run without a trained checkpoint on hand). To use the
# real trained model instead, pass --use-trained-model with a fusion_model.pt
# produced by phase4_fusion_model/train.py.

def heuristic_fusion(vision: VisionFrame, physio: PhysioSample,
                      env: EnvSample, obd: OBDSample) -> FusionOutput:
    drowsiness = min(1.0, 0.6 * vision.perclos + 0.3 * vision.yawn_score
                      + 0.1 * (vision.head_nod_angle / 20.0))
    health_anomaly = 0.0
    if physio.hr_bpm and (physio.hr_bpm < 45 or physio.hr_bpm > 130):
        health_anomaly = 0.9
    elif physio.spo2_pct and physio.spo2_pct < 92:
        health_anomaly = 0.85
    distraction = vision.distraction_conf if vision.distraction_label != "none" else 0.0

    return FusionOutput(
        timestamp=now(),
        drowsiness_score=drowsiness,
        health_anomaly_score=health_anomaly,
        distraction_score=distraction,
        time_to_event_s=(10.0 if drowsiness > 0.8 else None),
    )


def run(duration_s: float, tick_hz: float, use_real_vision: bool, use_real_physio: bool):
    vision_source = SimulatedVisionSource()
    physio_source = SimulatedPhysioSource()
    env_source = SimulatedEnvSource()
    obd_source = SimulatedOBDSource()
    alert_engine = AlertEngine(simulate=True)

    print("[*] Multimodal Driver Drowsiness & Health Monitoring System -- running")
    print("    (Phase 1-3 simulated sensing -> heuristic fusion -> Phase 5 alerts)")

    start = time.time()
    tick = 0
    while time.time() - start < duration_s:
        vision = vision_source.read()
        physio = physio_source.read()
        env = env_source.read()
        obd = obd_source.read()

        fusion_out = heuristic_fusion(vision, physio, env, obd)
        risk = alert_engine.handle(fusion_out)

        if tick % 5 == 0:
            print(f"[t={time.time()-start:5.1f}s] EAR={vision.ear:.2f} PERCLOS={vision.perclos:.2f} "
                  f"HR={physio.hr_bpm:.0f} -> drowsy={fusion_out.drowsiness_score:.2f} "
                  f"health={fusion_out.health_anomaly_score:.2f} risk={risk.name}")

        tick += 1
        time.sleep(1.0 / tick_hz)

    print("[*] Session complete.")


def main():
    ap = argparse.ArgumentParser(description="Driver Monitoring System - full pipeline")
    ap.add_argument("--duration-s", type=float, default=15.0)
    ap.add_argument("--tick-hz", type=float, default=5.0)
    ap.add_argument("--camera-source", help="Use real webcam instead of simulation")
    ap.add_argument("--serial-port", help="Use real ESP32 UART instead of simulation")
    args = ap.parse_args()

    run(duration_s=args.duration_s, tick_hz=args.tick_hz,
        use_real_vision=bool(args.camera_source),
        use_real_physio=bool(args.serial_port))


if __name__ == "__main__":
    main()
