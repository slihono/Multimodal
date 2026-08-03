"""
Phase 6 - Cloud Integration and Emergency: Emergency API
===========================================================
Reads the SIM7600G-H (4G) module and the NEO-M8N GPS module, and when a
CRITICAL health-anomaly is confirmed by the Phase 4 fusion model +
Phase 5 alert engine, contacts emergency services with the driver's
location and vital signs.

⚠️ Claim discipline (see project overview): this module frames its
action as "automated emergency reporting following detected anomaly",
never as medical diagnosis. It notifies; it does not diagnose.

Two backends:
  - GPSReader: reads NMEA sentences from the NEO-M8N over UART
    (pyserial + pynmea2), or returns simulated coordinates.
  - CellularModem: sends SMS/calls via AT commands to the SIM7600G-H
    over UART, or simulates the exchange for development without
    hardware.
"""

import sys
import time
from dataclasses import dataclass
from typing import Optional

sys.path.append("..")
from common.schemas import PhysioSample  # noqa: E402


@dataclass
class GPSFix:
    lat: float
    lon: float
    speed_kph: float = 0.0
    valid: bool = True


class GPSReader:
    def __init__(self, port: Optional[str] = None, baud: int = 9600, simulate: bool = True):
        self.simulate = simulate or port is None
        self.ser = None
        if not self.simulate:
            import serial
            self.ser = serial.Serial(port, baud, timeout=1)

    def read_fix(self) -> GPSFix:
        if self.simulate:
            # Simulated fixed location for development/testing.
            return GPSFix(lat=33.749, lon=-84.388, speed_kph=95.0)

        import pynmea2
        line = self.ser.readline().decode(errors="ignore").strip()
        try:
            msg = pynmea2.parse(line)
            if hasattr(msg, "latitude") and hasattr(msg, "longitude"):
                return GPSFix(lat=msg.latitude, lon=msg.longitude, valid=True)
        except pynmea2.ParseError:
            pass
        return GPSFix(lat=0.0, lon=0.0, valid=False)


class CellularModem:
    """AT-command wrapper for the SIM7600G-H. Real deployment sends this
    over UART with pyserial; simulated backend just prints what would be
    sent, which is enough to validate the emergency-trigger logic in
    Phase 7 bench testing before wiring the real modem."""

    def __init__(self, port: Optional[str] = None, baud: int = 115200, simulate: bool = True):
        self.simulate = simulate or port is None
        self.ser = None
        if not self.simulate:
            import serial
            self.ser = serial.Serial(port, baud, timeout=2)

    def _send_at(self, command: str) -> str:
        if self.simulate:
            print(f"[MODEM-SIM] > {command}")
            return "OK"
        self.ser.write((command + "\r\n").encode())
        time.sleep(0.3)
        return self.ser.read(self.ser.in_waiting or 1).decode(errors="ignore")

    def send_sms(self, number: str, message: str):
        self._send_at('AT+CMGF=1')                         # text mode
        self._send_at(f'AT+CMGS="{number}"')
        self._send_at(message + chr(26))                   # Ctrl+Z terminates SMS body
        print(f"[MODEM] SMS sent to {number}")

    def dial_call(self, number: str):
        self._send_at(f'ATD{number};')
        print(f"[MODEM] Dialing {number}")


class EmergencyResponder:
    """Ties GPS + cellular modem + cloud backend together. Called by the
    Phase 5 AlertEngine (or directly by Phase 4 orchestration) when
    RiskLevel.CRITICAL health-anomaly is sustained for a confirmation
    window (to avoid false-positive 911 calls on a single noisy sample).
    """

    def __init__(self, emergency_number: str = "911",
                 emergency_contacts: list = None,
                 gps: GPSReader = None, modem: CellularModem = None,
                 cloud_backend_url: Optional[str] = None,
                 confirmation_window_s: float = 15.0):
        self.emergency_number = emergency_number
        self.emergency_contacts = emergency_contacts or []
        self.gps = gps or GPSReader(simulate=True)
        self.modem = modem or CellularModem(simulate=True)
        self.cloud_backend_url = cloud_backend_url
        self.confirmation_window_s = confirmation_window_s
        self._anomaly_since: Optional[float] = None

    def update(self, health_anomaly_score: float, threshold: float,
               physio: Optional[PhysioSample] = None) -> bool:
        """Call this on every fusion-model tick. Returns True if an
        emergency was just triggered."""
        now = time.time()
        if health_anomaly_score < threshold:
            self._anomaly_since = None
            return False

        if self._anomaly_since is None:
            self._anomaly_since = now
            return False

        if now - self._anomaly_since >= self.confirmation_window_s:
            self._trigger(physio)
            self._anomaly_since = None  # reset so we don't spam-trigger
            return True
        return False

    def _trigger(self, physio: Optional[PhysioSample]):
        fix = self.gps.read_fix()
        vitals = ""
        if physio:
            vitals = f" HR={physio.hr_bpm}bpm SpO2={physio.spo2_pct}%"

        message = (f"AUTOMATED ALERT: possible driver medical emergency detected. "
                   f"Location: {fix.lat:.5f},{fix.lon:.5f}.{vitals} "
                   f"This is an automated notification, not a confirmed diagnosis.")

        self.modem.send_sms(self.emergency_number, message)
        for contact in self.emergency_contacts:
            self.modem.send_sms(contact, message)

        if self.cloud_backend_url:
            self._post_to_cloud(fix, physio)

    def _post_to_cloud(self, fix: GPSFix, physio: Optional[PhysioSample]):
        try:
            import requests
            requests.post(f"{self.cloud_backend_url}/emergency/trigger", json={
                "vehicle_id": "vehicle-001",
                "reason": "health_anomaly",
                "lat": fix.lat, "lon": fix.lon,
                "hr_bpm": physio.hr_bpm if physio else None,
                "spo2_pct": physio.spo2_pct if physio else None,
            }, timeout=5)
        except Exception as e:
            print(f"[!] Could not reach cloud backend: {e}")


if __name__ == "__main__":
    responder = EmergencyResponder(confirmation_window_s=2.0)  # short window for the demo
    print("[*] Simulating a sustained health anomaly...")
    for i in range(5):
        triggered = responder.update(health_anomaly_score=0.9, threshold=0.8)
        print(f"  tick {i}: triggered={triggered}")
        time.sleep(1)
