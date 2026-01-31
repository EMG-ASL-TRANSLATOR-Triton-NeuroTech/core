# Workshop 1 — Data Collection Foundations
Files included:
- brainflow_reader.py  : Ganglion streaming example (BrainFlow)
- mock_arduino.ino     : Arduino sketch simulating multi-channel EMG
- serial_logger.py     : Reads serial CSV rows and logs to CSV
- data_logger.py       : Reusable logger for labeled EMG data
- workshop_demo.py     : Switch between Ganglion and Serial modes and log data
- this README.md       : Setup notes

## Python environment
Recommended: Python 3.12 (per team). Install:
  pip install numpy scipy pandas matplotlib pyserial brainflow pylsl

## Quick start (Ganglion)
1. Install BrainFlow and drivers (OpenBCI GUI may be helpful to verify IP).
2. Run:
   python brainflow_reader.py --serial_port COM3 --duration 10 --out ganglion_sample.csv
(On macOS, use the device path like /dev/cu.usbserial-xxxx)

## Quick start (Arduino mock)
1. Upload mock_arduino.ino to your UNO/Nano.
2. Run:
   python serial_logger.py --port COM4 --baud 115200 --duration 10 --out emg_mock.csv

## Notes
- The scripts are intentionally simple for learning. They will need improvements for production (error handling, robust channel selection, timestamps synced to board).
- Use `data_logger.py` to standardize CSV format for downstream processing.
