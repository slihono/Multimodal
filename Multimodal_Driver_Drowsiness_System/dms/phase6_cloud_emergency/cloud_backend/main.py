"""
Phase 6 - Cloud Integration: FastAPI backend
===============================================
Receives telemetry (FusionOutput + GPS) from the vehicle over 4G
(SIM7600G-H), stores it, and exposes it for the Grafana dashboard and
for the emergency API.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Requires: fastapi, uvicorn, pydantic
    pip install fastapi uvicorn pydantic --break-system-packages
"""

import time
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Driver Monitoring Cloud Backend", version="1.0")

# In-memory store for the reference implementation. Swap for
# InfluxDB/PostgreSQL for production (see requirements.txt / Section 12
# of the project doc: FastAPI + Grafana + InfluxDB stack).
_telemetry_log: List[dict] = []
_MAX_LOG_SIZE = 10_000


class TelemetryIn(BaseModel):
    vehicle_id: str
    timestamp: Optional[float] = None
    drowsiness_score: float
    health_anomaly_score: float
    distraction_score: float
    time_to_event_s: Optional[float] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    speed_kph: Optional[float] = None


class EmergencyTrigger(BaseModel):
    vehicle_id: str
    reason: str            # "health_anomaly", "manual_sos", "crash_detected"
    lat: float
    lon: float
    hr_bpm: Optional[float] = None
    spo2_pct: Optional[float] = None


@app.post("/telemetry")
def ingest_telemetry(payload: TelemetryIn):
    record = payload.dict()
    record["received_at"] = time.time()
    if record["timestamp"] is None:
        record["timestamp"] = record["received_at"]
    _telemetry_log.append(record)
    if len(_telemetry_log) > _MAX_LOG_SIZE:
        _telemetry_log.pop(0)
    return {"status": "ok", "stored_count": len(_telemetry_log)}


@app.get("/telemetry/{vehicle_id}")
def get_telemetry(vehicle_id: str, limit: int = 100):
    records = [r for r in _telemetry_log if r["vehicle_id"] == vehicle_id]
    return records[-limit:]


@app.get("/telemetry/{vehicle_id}/latest")
def get_latest(vehicle_id: str):
    records = [r for r in _telemetry_log if r["vehicle_id"] == vehicle_id]
    if not records:
        raise HTTPException(status_code=404, detail="No telemetry for this vehicle yet")
    return records[-1]


@app.get("/fleet/summary")
def fleet_summary():
    """Aggregate view for Grafana / fleet management dashboards
    (Target Markets: fleet management, ride-sharing, public transport)."""
    by_vehicle = {}
    for r in _telemetry_log:
        vid = r["vehicle_id"]
        by_vehicle.setdefault(vid, []).append(r)

    summary = []
    for vid, records in by_vehicle.items():
        latest = records[-1]
        avg_drowsy = sum(r["drowsiness_score"] for r in records) / len(records)
        summary.append({
            "vehicle_id": vid,
            "latest_drowsiness": latest["drowsiness_score"],
            "latest_health_anomaly": latest["health_anomaly_score"],
            "avg_drowsiness_session": avg_drowsy,
            "sample_count": len(records),
        })
    return summary


@app.post("/emergency/trigger")
def trigger_emergency(payload: EmergencyTrigger):
    """Called by the vehicle unit (or the emergency_api.py module directly)
    when a CRITICAL health-anomaly alert fires. Logs the event; actual
    dispatch to emergency services is handled in emergency_api.py, kept
    separate so this cloud endpoint can also notify emergency contacts,
    fleet dispatch, and insurance partners in parallel."""
    event = payload.dict()
    event["triggered_at"] = time.time()
    # In production: push to a message queue (e.g. SQS/PubSub) consumed by
    # the emergency-response worker, notification service, and audit log.
    print(f"[EMERGENCY] {event}")
    return {"status": "received", "event": event}


@app.get("/health")
def health_check():
    return {"status": "ok", "records_stored": len(_telemetry_log)}
