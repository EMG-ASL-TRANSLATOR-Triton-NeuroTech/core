// mock_arduino.ino
// Simple Arduino sketch that simulates a multi-channel EMG signal by printing comma-separated values
// Upload to any UNO/Nano and open serial at 115200 or 9600 as configured below.
void setup() {
  Serial.begin(115200);
}

void loop() {
  unsigned long t = millis();
  // create some fake channels (4 channels)
  float c1 = 0.5 + 0.5 * sin(t / 200.0);
  float c2 = 0.4 + 0.4 * sin(t / 180.0 + 0.5);
  float c3 = 0.3 + 0.3 * sin(t / 160.0 + 1.0);
  float c4 = 0.2 + 0.2 * sin(t / 140.0 + 1.5);
  // print comma separated channels
  Serial.print(c1, 6); Serial.print(',');
  Serial.print(c2, 6); Serial.print(',');
  Serial.print(c3, 6); Serial.print(',');
  Serial.println(c4, 6);
  delay(5); // ~200 Hz
}
