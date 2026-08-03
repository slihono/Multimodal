"""
common/schemas.py
------------------
Shared data structures used across every phase of the Multimodal Driver
Drowsiness & Health Monitoring System. Keeping these in one place means
Phase 1 vision output, Phase 2 physio output, Phase 3 advanced-CV output,
etc. all speak the same language when they reach the Phase 4 fusion model.
"""

from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class VisionFrame:
    """Output of the Phase 1 / Phase 3 vision pipeline for a single frame."""
    timestamp: float
    ear: float                      # Eye Aspect Ratio (both eyes averaged)
    perclos: float                  # % of time eyes closed over rolling window
    yawn_score: float = 0.0         # 0-1, mouth-aspect-ratio derived
    head_nod_angle: float = 0.0     # degrees, pitch deviation from baseline
    distraction_label: str = "none" # "none", "phone", "eating", "smoking", "looking_away"
    distraction_conf: float = 0.0
    emotion: str = "neutral"        # "neutral","angry","anxious","happy","sad","tired"
    emotion_conf: float = 0.0
    rppg_hr_bpm: Optional[float] = None
    face_found: bool = True


@dataclass
class PhysioSample:
    """Output of the Phase 2 ESP32-S3 physiological signal acquisition chain."""
    timestamp: float
    hr_bpm: Optional[float] = None
    spo2_pct: Optional[float] = None
    gsr_raw: Optional[float] = None       # microsiemens (filtered)
    accel_xyz: tuple = (0.0, 0.0, 0.0)    # g
    gyro_xyz: tuple = (0.0, 0.0, 0.0)     # deg/s


@dataclass
class EnvSample:
    """Cabin / environment sensors (subset of BOM section 4.7)."""
    timestamp: float
    temperature_c: float = 22.0
    humidity_pct: float = 45.0
    co2_ppm: float = 600.0
    voc_index: float = 50.0
    alcohol_ppm: float = 0.0
    ambient_lux: float = 300.0


@dataclass
class OBDSample:
    """Vehicle bus data (Section 4.10)."""
    timestamp: float
    speed_kph: float = 0.0
    rpm: float = 0.0
    throttle_pct: float = 0.0
    brake_active: bool = False
    steering_angle_deg: float = 0.0


@dataclass
class FusionOutput:
    """Output of the Phase 4 multimodal transformer."""
    timestamp: float
    drowsiness_score: float        # 0-1
    health_anomaly_score: float    # 0-1
    distraction_score: float       # 0-1
    time_to_event_s: Optional[float]  # estimated seconds until risk threshold, None if safe


def now() -> float:
    return time.time()
