"""serial_logger.py
Reads newline-delimited CSV rows from a serial port and logs them with timestamps into a CSV file.
Works on Windows and macOS (adjust COM port / device name).
Usage:
  python serial_logger.py --port COM4 --baud 115200 --out emg_serial.csv --duration 10
"""
import argparse, serial, csv, time
from datetime import datetime

def read_serial(port, baud, out_file, duration=10.0):
    with serial.Serial(port, baud, timeout=1) as ser, open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'chan_1', 'chan_2', 'chan_3', 'chan_4'])
        start = time.time()
        while time.time() - start < duration:
            line = ser.readline().decode(errors='ignore').strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(',') if p.strip()!='']
            if len(parts) < 4:
                # skip incomplete lines
                continue
            ts = time.time()
            writer.writerow([ts] + parts[:4])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', required=True, help='COM port or /dev device')
    parser.add_argument('--baud', type=int, default=115200)
    parser.add_argument('--out', type=str, default=f'emg_serial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv')
    parser.add_argument('--duration', type=float, default=10.0)
    args = parser.parse_args()
    read_serial(args.port, args.baud, args.out, args.duration)
