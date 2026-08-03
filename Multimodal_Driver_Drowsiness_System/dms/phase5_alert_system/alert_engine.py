"""
Phase 5 - Driver Alert System and User Interface
===================================================
Decision & Alert Engine: takes the Phase 4 FusionOutput and drives all
output devices (voice/TTS via INMP441+speaker, haptic via DRV2605L,
WS2812B RGB LED strip, mini-OLED display) based on configurable risk
thresholds.

Each hardware interface below has a *simulated* backend (prints /
logs what would happen) and, where the Python driver library exists, a
*real* backend behind the same interface, so the exact same
AlertEngine code runs in dev (no hardware attached) and on the Jetson
with real peripherals wired up.
"""

import sys
import time
from dataclasses import dataclass
from enum import Enum

sys.path.append("..")
from common.schemas import FusionOutput  # noqa: E402


class RiskLevel(Enum):
    SAFE = 0
    CAUTION = 1
    WARNING = 2
    CRITICAL = 3


@dataclass
class AlertThresholds:
    caution: float = 0.35
    warning: float = 0.60
    critical: float = 0.80


def classify_risk(score: float, t: AlertThresholds) -> RiskLevel:
    if score >= t.critical:
        return RiskLevel.CRITICAL
    if score >= t.warning:
        return RiskLevel.WARNING
    if score >= t.caution:
        return RiskLevel.CAUTION
    return RiskLevel.SAFE


# ---------------------------------------------------------------- devices --

class TextToSpeechAlert:
    """INMP441 mic is for audio *input* (voice commands); output goes
    through a speaker via the MAX98357A I2S amp. Uses pyttsx3 (offline
    TTS, no cloud dependency -- important for an edge/privacy-preserving
    system) when available, otherwise logs the message."""

    def __init__(self, simulate: bool = True):
        self.simulate = simulate
        self.engine = None
        if not simulate:
            try:
                import pyttsx3
                self.engine = pyttsx3.init()
            except ImportError:
                print("[!] pyttsx3 not installed, falling back to simulation. "
                      "pip install pyttsx3 --break-system-packages")
                self.simulate = True

    def speak(self, message: str):
        if self.simulate or self.engine is None:
            print(f"[TTS-SIM] \"{message}\"")
        else:
            self.engine.say(message)
            self.engine.runAndWait()


class HapticFeedback:
    """DRV2605L haptic driver (I2C) driving an LRA motor in the seat/
    steering wheel. Real backend uses smbus2 if available."""

    def __init__(self, simulate: bool = True, i2c_addr: int = 0x5A):
        self.simulate = simulate
        self.i2c_addr = i2c_addr
        self.bus = None
        if not simulate:
            try:
                import smbus2
                self.bus = smbus2.SMBus(1)
            except (ImportError, FileNotFoundError):
                print("[!] smbus2/I2C bus not available, falling back to simulation.")
                self.simulate = True

    def pulse(self, intensity: str = "medium"):
        # DRV2605L effect library IDs (see datasheet section 11.13):
        effect_map = {"low": 1, "medium": 14, "high": 47}  # e.g. Strong Click, Buzz, Double Click
        effect_id = effect_map.get(intensity, 14)
        if self.simulate or self.bus is None:
            print(f"[HAPTIC-SIM] pulse intensity={intensity} (effect {effect_id})")
        else:
            self.bus.write_byte_data(self.i2c_addr, 0x01, 0x01)   # select ROM library
            self.bus.write_byte_data(self.i2c_addr, 0x04, effect_id)
            self.bus.write_byte_data(self.i2c_addr, 0x0C, 0x01)   # GO


class LEDStrip:
    """WS2812B RGB LED strip for ambient visual alerts. Real backend
    uses rpi_ws281x / neopixel; simulated backend just prints the color."""

    COLORS = {
        RiskLevel.SAFE: (0, 255, 0),
        RiskLevel.CAUTION: (255, 255, 0),
        RiskLevel.WARNING: (255, 128, 0),
        RiskLevel.CRITICAL: (255, 0, 0),
    }

    def __init__(self, simulate: bool = True, led_count: int = 30):
        self.simulate = simulate
        self.led_count = led_count
        self.strip = None
        if not simulate:
            try:
                from rpi_ws281x import PixelStrip, Color
                self.Color = Color
                self.strip = PixelStrip(led_count, 18)
                self.strip.begin()
            except ImportError:
                print("[!] rpi_ws281x not installed, falling back to simulation.")
                self.simulate = True

    def set_level(self, level: RiskLevel, blink: bool = False):
        r, g, b = self.COLORS[level]
        if self.simulate or self.strip is None:
            blink_str = " (blinking)" if blink else ""
            print(f"[LED-SIM] level={level.name} rgb=({r},{g},{b}){blink_str}")
        else:
            for i in range(self.led_count):
                self.strip.setPixelColor(i, self.Color(r, g, b))
            self.strip.show()


class OLEDDisplay:
    """SSD1306 mini-OLED for system status. Real backend uses
    luma.oled / adafruit_ssd1306; simulated backend prints the layout."""

    def __init__(self, simulate: bool = True):
        self.simulate = simulate
        self.device = None
        if not simulate:
            try:
                from luma.core.interface.serial import i2c
                from luma.oled.device import ssd1306
                serial = i2c(port=1, address=0x3C)
                self.device = ssd1306(serial)
            except ImportError:
                print("[!] luma.oled not installed, falling back to simulation.")
                self.simulate = True

    def show_status(self, drowsiness: float, health: float, distraction: float, level: RiskLevel):
        text = (f"Drowsy:{drowsiness:.0%} Health:{health:.0%} "
                f"Distr:{distraction:.0%} [{level.name}]")
        if self.simulate or self.device is None:
            print(f"[OLED-SIM] {text}")
        else:
            from luma.core.render import canvas
            with canvas(self.device) as draw:
                draw.text((0, 0), text, fill="white")


# ------------------------------------------------------------ alert engine --

class AlertEngine:
    def __init__(self, thresholds: AlertThresholds = None, simulate: bool = True):
        self.thresholds = thresholds or AlertThresholds()
        self.tts = TextToSpeechAlert(simulate=simulate)
        self.haptic = HapticFeedback(simulate=simulate)
        self.led = LEDStrip(simulate=simulate)
        self.oled = OLEDDisplay(simulate=simulate)
        self._last_spoken_at = 0.0
        self._speak_cooldown_s = 10.0

    def handle(self, fusion_out: FusionOutput):
        drowsy_level = classify_risk(fusion_out.drowsiness_score, self.thresholds)
        health_level = classify_risk(fusion_out.health_anomaly_score, self.thresholds)
        distraction_level = classify_risk(fusion_out.distraction_score, self.thresholds)
        overall = max(drowsy_level, health_level, distraction_level, key=lambda l: l.value)

        self.oled.show_status(fusion_out.drowsiness_score, fusion_out.health_anomaly_score,
                               fusion_out.distraction_score, overall)
        self.led.set_level(overall, blink=(overall.value >= RiskLevel.WARNING.value))

        if overall == RiskLevel.SAFE:
            return overall

        self.haptic.pulse(intensity={"CAUTION": "low", "WARNING": "medium", "CRITICAL": "high"}[overall.name])

        now = time.time()
        if now - self._last_spoken_at > self._speak_cooldown_s:
            message = self._message_for(overall, drowsy_level, health_level, distraction_level)
            self.tts.speak(message)
            self._last_spoken_at = now

        return overall

    @staticmethod
    def _message_for(overall, drowsy, health, distraction) -> str:
        if health.value >= RiskLevel.WARNING.value:
            return "Please pull over safely. A potential health issue has been detected."
        if drowsy.value >= RiskLevel.WARNING.value:
            return "You seem drowsy. Please consider taking a break."
        if distraction.value >= RiskLevel.WARNING.value:
            return "Please keep your eyes on the road."
        return "Stay alert. Consider taking a break soon."


if __name__ == "__main__":
    engine = AlertEngine(simulate=True)
    scenarios = [
        FusionOutput(timestamp=time.time(), drowsiness_score=0.10, health_anomaly_score=0.05,
                     distraction_score=0.10, time_to_event_s=None),
        FusionOutput(timestamp=time.time(), drowsiness_score=0.72, health_anomaly_score=0.10,
                     distraction_score=0.20, time_to_event_s=25.0),
        FusionOutput(timestamp=time.time(), drowsiness_score=0.30, health_anomaly_score=0.85,
                     distraction_score=0.15, time_to_event_s=8.0),
    ]
    for s in scenarios:
        print(f"\n--- scenario drowsy={s.drowsiness_score} health={s.health_anomaly_score} ---")
        engine.handle(s)
