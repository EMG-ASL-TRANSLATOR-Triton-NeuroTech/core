"""
workshop_demo.py
Unified data collection demo.
Switch between BrainFlow (OpenBCI Ganglion) and Serial (Arduino) inputs
and log EMG data with DataLogger.

Usage examples:
  python workshop_demo.py --mode ganglion --board-id 1 --serial-port COM3 --duration 10
  python workshop_demo.py --mode serial --port COM4 --baud 115200 --duration 10
"""

import argparse
import time
from datetime import datetime
from data_logger import DataLogger
import pylsl;


# ---------------------------------------------------------------------
#  Ganglion (BrainFlow) Mode
# ---------------------------------------------------------------------
def ganglion_mode(args):
    try:
        from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    except Exception as e:
        print("BrainFlow not installed or not available:", e)
        return

    BoardShim.enable_dev_board_logger()

    # Initialize BrainFlow parameters
    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = args.serial_port
    params.mac_address = args.mac_address
    params.other_info = args.other_info
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout
    params.file = args.file
    params.master_board = args.master_board

    # Connect and start streaming
    board = BoardShim(args.board_id, params)
    board.prepare_session()
    board.start_stream()
    print(f"Ganglion stream started for {args.duration}s on port {args.serial_port}")

    dl = DataLogger(out_dir="data",
                    session_name="ganglion_" + time.strftime("%Y%m%d_%H%M%S"),
                    channels=4)

    start = time.time()
    while time.time() - start < args.duration:
        data = board.get_current_board_data(256)
        if data.size == 0:
            time.sleep(0.01)
            continue

        # transpose to get one row per sample
        for sample in data.T:
            ts = time.time()
            chans = [float(sample[i]) if i < len(sample) else 0.0 for i in range(4)]
            dl.write_row(ts, chans, label="")  # labels added later

    board.stop_stream()
    board.release_session()
    print("Ganglion stream complete.")


# ---------------------------------------------------------------------
#  Serial (Arduino) Mode
# ---------------------------------------------------------------------
def serial_mode(port, baud, duration):
    import serial

    dl = DataLogger(out_dir="data",
                    session_name="serial_" + time.strftime("%Y%m%d_%H%M%S"),
                    channels=4)

    with serial.Serial(port, baud, timeout=1) as ser:
        start = time.time()
        while time.time() - start < duration:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",") if p.strip() != ""]
            if len(parts) < 4:
                continue
            ts = time.time()
            dl.write_row(ts, parts[:4], label="")

    print("Serial logging complete.")


# ---------------------------------------------------------------------
#  Main CLI Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EMG Data Collection Demo")

    # --- Common parameters ---
    parser.add_argument("--mode", choices=["ganglion", "serial"], required=True)
    parser.add_argument("--duration", type=float, default=10.0)

    # --- BrainFlow-specific parameters ---
    parser.add_argument("--board-id", type=int, default=1, help="1 for Ganglion, 0 for Cyton")
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--ip-port", type=int, default=0)
    parser.add_argument("--ip-protocol", type=int, default=0)
    parser.add_argument("--ip-address", type=str, default="")
    parser.add_argument("--serial-port", type=str, default="")
    parser.add_argument("--mac-address", type=str, default="")
    parser.add_argument("--other-info", type=str, default="")
    parser.add_argument("--serial-number", type=str, default="")
    parser.add_argument("--file", type=str, default="")
    parser.add_argument("--master-board", type=int, default=0)

    # --- Serial-specific parameters ---
    parser.add_argument("--port", type=str, help="Serial port for Arduino")
    parser.add_argument("--baud", type=int, default=115200)

    args = parser.parse_args()

    # --- Run selected mode ---
    if args.mode == "ganglion":
        assert args.serial_port, "Provide --serial-port for Ganglion mode"
        ganglion_mode(args)
    else:
        assert args.port, "Provide --port for Serial mode"
        serial_mode(args.port, args.baud, args.duration)

