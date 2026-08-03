# Multimodal Driver Drowsiness & Health Monitoring System

Full software implementation, Phase 1 through Phase 7, based on the
project spec (Dr. Abdelrahman Elfikky's lab, supervised by
Muzakkiruddin Ahmed Mohammed).

Every hardware-dependent module (cameras, ESP32-S3 sensors, Jetson,
GPS/4G modem, haptic/LED/OLED peripherals) ships with a **simulated
backend** so the entire pipeline runs end-to-end on a laptop with no
hardware attached. Swap in the real backend (flags/args are documented
in each file's docstring) once your hardware is wired up.

## Quick start

```bash
pip install -r requirements.txt --break-system-packages

# Run the full simulated end-to-end pipeline (Phase 1-6 in one loop):
python main_pipeline.py --duration-s 30

# Or run each phase's script independently, see each phase's README.
```

## Project structure

```
dms/
├── common/
│   └── schemas.py                # Shared data types (VisionFrame, PhysioSample, ...)
├── phase1_vision_setup/          # Face Mesh, EAR, PERCLOS
│   └── face_mesh_ear.py
├── phase2_physio_signals/        # ESP32-S3: HR/SpO2, GSR, IMU
│   ├── esp32_firmware/main.ino
│   └── signal_processor.py
├── phase3_advanced_cv/           # Distraction (YOLOv8), emotion (EfficientNet), rPPG
│   ├── distraction_yolov8.py
│   ├── emotion_recognition.py
│   └── rppg_extraction.py
├── phase4_fusion_model/          # Multimodal cross-attention Transformer
│   ├── encoders.py
│   ├── fusion_transformer.py
│   └── train.py
├── phase5_alert_system/          # Decision & alert engine, TTS/haptic/LED/OLED
│   └── alert_engine.py
├── phase6_cloud_emergency/       # FastAPI backend, emergency SOS, Grafana dashboard
│   ├── cloud_backend/main.py
│   ├── emergency_api.py
│   └── grafana/dashboard.json
├── phase7_validation/            # Threshold calibration, metrics, session logging
│   ├── metrics.py
│   └── validation_pipeline.py
├── main_pipeline.py               # End-to-end orchestrator (all phases wired together)
└── requirements.txt
```

## Phase-by-phase deliverables

| Phase | Deliverable | Entry point |
|---|---|---|
| 1 | Face detection + EAR/PERCLOS drowsiness script | `phase1_vision_setup/face_mesh_ear.py` |
| 2 | Synchronized physio data stream (HR/SpO2/GSR/IMU) | `phase2_physio_signals/signal_processor.py` (+ ESP32 firmware) |
| 3 | 3-model CV pipeline (distraction, emotion, rPPG) | `phase3_advanced_cv/*.py` |
| 4 | Multimodal fusion Transformer | `phase4_fusion_model/fusion_transformer.py`, `train.py` |
| 5 | Driver alert system (voice/haptic/visual/OLED) | `phase5_alert_system/alert_engine.py` |
| 6 | Emergency + cloud dashboard | `phase6_cloud_emergency/` |
| 7 | Threshold calibration, FP/FN reduction, robustness | `phase7_validation/` |

## Notes on claim discipline

Per the project overview: this system frames health-related output as
**"early anomaly detection and rapid automated response,"** never as a
predictive medical diagnosis (e.g. "predicting cardiac arrest 30-60s in
advance"). The emergency module (`phase6_cloud_emergency/emergency_api.py`)
and its alert message reflect that framing explicitly.

## Testing what's been built

Every module has a `if __name__ == "__main__":` smoke test at the
bottom that runs on synthetic data with no hardware attached:

```bash
python phase4_fusion_model/fusion_transformer.py   # random-tensor forward pass
python phase4_fusion_model/train.py                # 3-epoch training loop on synthetic data
python phase5_alert_system/alert_engine.py         # 3 example risk scenarios
python phase6_cloud_emergency/emergency_api.py     # simulated sustained health anomaly -> SOS
python phase7_validation/metrics.py                # threshold calibration on synthetic runs
python main_pipeline.py --duration-s 20            # full pipeline, simulated drowsy driver
```
