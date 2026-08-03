/*
 * Phase 2 - Physiological Signal Acquisition
 * ===========================================
 * ESP32-S3 firmware.
 *
 * Sensors:
 *   - MAX30102 : Heart Rate (HR) + SpO2, I2C
 *   - GSR (Grove GSR V1.2) : Galvanic Skin Response, analog (ADC)
 *   - MPU6050 : 6-axis IMU (accel + gyro), I2C
 *
 * Transmits a JSON line per sample over UART (115200 baud) AND, if WiFi
 * credentials are set, publishes the same JSON to an MQTT broker on
 * topic "driver/physio".
 *
 * Libraries needed (Arduino IDE Library Manager):
 *   - SparkFun MAX3010x Pulse and Proximity Sensor Library
 *   - Adafruit MPU6050 + Adafruit Unified Sensor
 *   - PubSubClient (MQTT)
 *   - ArduinoJson
 */

#include <Wire.h>
#include "MAX30105.h"
#include "heartRate.h"
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

// ---------- Configuration ----------
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* MQTT_BROKER   = "192.168.1.100";
const int   MQTT_PORT     = 1883;
const char* MQTT_TOPIC    = "driver/physio";
const bool  USE_MQTT      = false;   // set true once WiFi/broker are configured

const int GSR_PIN = 34;              // ADC1 channel on ESP32-S3
const unsigned long SAMPLE_INTERVAL_MS = 40;   // 25 Hz

// ---------- Globals ----------
MAX30105 particleSensor;
Adafruit_MPU6050 mpu;
WiFiClient espClient;
PubSubClient mqttClient(espClient);

// Simple moving-average filter for GSR (matches the "moving average filter"
// task in Phase 2). Butterworth filtering for HR/PPG is done on the PC side
// (see signal_processor.py) where SciPy is available; this keeps the MCU
// firmware lightweight and real-time safe.
const int GSR_MA_SIZE = 8;
float gsrBuffer[GSR_MA_SIZE] = {0};
int gsrIdx = 0;

float movingAverage(float newSample) {
  gsrBuffer[gsrIdx] = newSample;
  gsrIdx = (gsrIdx + 1) % GSR_MA_SIZE;
  float sum = 0;
  for (int i = 0; i < GSR_MA_SIZE; i++) sum += gsrBuffer[i];
  return sum / GSR_MA_SIZE;
}

// Beat detection state for MAX30102 (algorithm from SparkFun heartRate.h)
long lastBeat = 0;
float beatsPerMinute = 0;
int beatAvg = 0;

void connectWiFiAndMQTT() {
  if (!USE_MQTT) return;
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 10000) {
    delay(250);
    Serial.print(".");
  }
  mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
}

void setup() {
  Serial.begin(115200);
  Wire.begin();

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("{\"error\":\"MAX30102 not found\"}");
  } else {
    particleSensor.setup();               // default config: red+IR, 100Hz
    particleSensor.setPulseAmplitudeRed(0x0A);
    particleSensor.setPulseAmplitudeGreen(0);
  }

  if (!mpu.begin()) {
    Serial.println("{\"error\":\"MPU6050 not found\"}");
  } else {
    mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  }

  pinMode(GSR_PIN, INPUT);
  connectWiFiAndMQTT();
}

void loop() {
  static unsigned long lastSample = 0;
  if (millis() - lastSample < SAMPLE_INTERVAL_MS) return;
  lastSample = millis();

  // --- Heart rate (MAX30102) ---
  long irValue = particleSensor.getIR();
  if (checkForBeat(irValue)) {
    long delta = millis() - lastBeat;
    lastBeat = millis();
    beatsPerMinute = 60.0 / (delta / 1000.0);
    if (beatsPerMinute > 20 && beatsPerMinute < 255) {
      beatAvg = (beatAvg == 0) ? beatsPerMinute : (beatAvg * 0.8 + beatsPerMinute * 0.2);
    }
  }
  // SpO2 requires the red/IR ratio-of-ratios algorithm (see SparkFun
  // spo2_algorithm.h in the full firmware repo); omitted here for brevity,
  // raw IR/RED are sent so the PC-side pipeline can compute it if needed.
  long redValue = particleSensor.getRed();

  // --- GSR ---
  int gsrRaw = analogRead(GSR_PIN);
  float gsrFiltered = movingAverage((float)gsrRaw);

  // --- IMU ---
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  // --- Build JSON payload ---
  StaticJsonDocument<256> doc;
  doc["t"] = millis();
  doc["hr_bpm"] = beatAvg;
  doc["ir_raw"] = irValue;
  doc["red_raw"] = redValue;
  doc["gsr_raw"] = gsrFiltered;
  doc["ax"] = a.acceleration.x;
  doc["ay"] = a.acceleration.y;
  doc["az"] = a.acceleration.z;
  doc["gx"] = g.gyro.x;
  doc["gy"] = g.gyro.y;
  doc["gz"] = g.gyro.z;

  char buf[256];
  size_t n = serializeJson(doc, buf);
  Serial.write(buf, n);
  Serial.write('\n');

  if (USE_MQTT && mqttClient.connected()) {
    mqttClient.publish(MQTT_TOPIC, buf, n);
  }
}
