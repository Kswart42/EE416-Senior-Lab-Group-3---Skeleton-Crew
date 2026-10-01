/*
  ESP32-S3-DevKitC-1 + Adafruit MPU-6050 (STEMMA QT) Motion Tracker
  -----------------------------------------------------------------
  Reads accel + gyro over I2C and estimates roll/pitch with a
  complementary filter. Yaw is gyro-only and WILL drift (the MPU-6050
  has no magnetometer).

  Wiring (DevKitC-1 has no STEMMA QT port, so use a STEMMA QT -> male
  header cable):
    MPU6050 SDA (blue)   -> GPIO8
    MPU6050 SCL (yellow) -> GPIO9
    MPU6050 VIN (red)    -> 3V3
    MPU6050 GND (black)  -> GND

  Serial output goes over the native port labeled "USB" (not "UART")
  because of ARDUINO_USB_CDC_ON_BOOT=1 in platformio.ini.
*/

#include <Arduino.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

Adafruit_MPU6050 mpu;

constexpr int SDA_PIN = 8;
constexpr int SCL_PIN = 9;

constexpr float    ALPHA       = 0.98f;   // gyro weight in complementary filter
constexpr uint32_t LOOP_US     = 10000;   // 100 Hz sensor/filter rate
constexpr uint8_t  PRINT_EVERY = 2;       // print every Nth sample (50 Hz)

// Filter state (radians)
float roll = 0.0f, pitch = 0.0f, yaw = 0.0f;

// Gyro bias (rad/s), measured at startup while the board is still
float gyroBiasX = 0.0f, gyroBiasY = 0.0f, gyroBiasZ = 0.0f;

uint32_t lastUpdateMicros = 0;

void calibrateGyro(int samples = 500) {
  while (true) {
    Serial.println("Calibrating gyro... keep the sensor still.");
    float sum[3] = {0, 0, 0};
    float mn[3]  = { 1e9f,  1e9f,  1e9f};
    float mx[3]  = {-1e9f, -1e9f, -1e9f};

    for (int i = 0; i < samples; i++) {
      sensors_event_t a, g, t;
      mpu.getEvent(&a, &g, &t);
      float v[3] = {g.gyro.x, g.gyro.y, g.gyro.z};
      for (int k = 0; k < 3; k++) {
        sum[k] += v[k];
        if (v[k] < mn[k]) mn[k] = v[k];
        if (v[k] > mx[k]) mx[k] = v[k];
      }
      delay(3);
    }

    // If the board was moved during calibration, the bias is garbage: retry.
    bool still = true;
    for (int k = 0; k < 3; k++) {
      if (mx[k] - mn[k] > 0.1f) still = false;  // ~6 deg/s peak-to-peak
    }
    if (!still) {
      Serial.println("Motion detected during calibration, retrying...");
      continue;
    }

    gyroBiasX = sum[0] / samples;
    gyroBiasY = sum[1] / samples;
    gyroBiasZ = sum[2] / samples;
    Serial.println("Calibration complete.");
    return;
  }
}

// Start the filter at the real orientation instead of 0,0
void initAnglesFromAccel() {
  sensors_event_t a, g, t;
  mpu.getEvent(&a, &g, &t);
  float ax = a.acceleration.x, ay = a.acceleration.y, az = a.acceleration.z;
  roll  = atan2f(ay, az);
  pitch = atan2f(-ax, sqrtf(ay * ay + az * az));
  yaw   = 0.0f;
}

void setup() {
  Serial.begin(115200);
  delay(2000);  // let USB CDC enumerate

  Serial.println("Initializing MPU6050...");

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);

  if (!mpu.begin()) {  // default address 0x68
    Serial.println("Failed to find MPU6050. Check wiring/address.");
    while (1) delay(10);
  }
  Serial.println("MPU6050 found!");

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  delay(100);
  calibrateGyro();
  initAnglesFromAccel();

  lastUpdateMicros = micros();
}

void loop() {
  uint32_t now = micros();
  if (now - lastUpdateMicros < LOOP_US) {
    delay(1);
    return;
  }
  float dt = (now - lastUpdateMicros) * 1e-6f;
  lastUpdateMicros = now;

  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  float gx = g.gyro.x - gyroBiasX;
  float gy = g.gyro.y - gyroBiasY;
  float gz = g.gyro.z - gyroBiasZ;

  float ax = a.acceleration.x, ay = a.acceleration.y, az = a.acceleration.z;

  // Accel angle estimates. Each one is paired with the gyro axis that
  // rotates about the SAME axis:
  //   roll  = rotation about X -> gx
  //   pitch = rotation about Y -> gy
  float accelRoll  = atan2f(ay, az);
  float accelPitch = atan2f(-ax, sqrtf(ay * ay + az * az));

  roll  = ALPHA * (roll  + gx * dt) + (1.0f - ALPHA) * accelRoll;
  pitch = ALPHA * (pitch + gy * dt) + (1.0f - ALPHA) * accelPitch;
  yaw  += gz * dt;  // gyro only: drifts over time

  static uint8_t n = 0;
  if (++n < PRINT_EVERY) return;
  n = 0;

  Serial.print("accelX:"); Serial.print(ax, 3); Serial.print(",");
  Serial.print("accelY:"); Serial.print(ay, 3); Serial.print(",");
  Serial.print("accelZ:"); Serial.print(az, 3); Serial.print(",");
  Serial.print("gyroX:");  Serial.print(gx, 3); Serial.print(",");
  Serial.print("gyroY:");  Serial.print(gy, 3); Serial.print(",");
  Serial.print("gyroZ:");  Serial.print(gz, 3); Serial.print(",");
  Serial.print("roll:");   Serial.print(roll  * RAD_TO_DEG, 2); Serial.print(",");
  Serial.print("pitch:");  Serial.print(pitch * RAD_TO_DEG, 2); Serial.print(",");
  Serial.print("yaw:");    Serial.print(yaw   * RAD_TO_DEG, 2);
  Serial.println();
}
