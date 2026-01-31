"""
brainflow_reader.py
Streams EMG data from OpenBCI Ganglion using BrainFlow.
Compatible with Python 3.12.

Usage example:
    python brainflow_reader.py --board-id 1 --serial-port COM3 --duration 10
On macOS:
    python brainflow_reader.py --board-id 1 --serial-port /dev/cu.usbserial-1420
"""

import argparse
import time
import csv
from datetime import datetime
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds


def main():
    BoardShim.enable_dev_board_logger()

    # -----------------------------
    # Command-line argument parser
    # -----------------------------
    parser = argparse.ArgumentParser(description="Stream EMG data from Ganglion using BrainFlow")
    parser.add_argument('--timeout', type=int, required=False, default=0)
    parser.add_argument('--ip-port', type=int, required=False, default=0)
    parser.add_argument('--ip-protocol', type=int, required=False, default=0)
    parser.add_argument('--ip-address', type=str, required=False, default='')
    parser.add_argument('--serial-port', type=str, required=False, default='')
    parser.add_argument('--mac-address', type=str, required=False, default='')
    parser.add_argument('--other-info', type=str, required=False, default='')
    parser.add_argument('--serial-number', type=str, required=False, default='')
    parser.add_argument('--file', type=str, required=False, default='')
    parser.add_argument('--master-board', type=int, required=False, default=BoardIds.NO_BOARD)
    parser.add_argument('--board-id', type=int, required=True, help='Board ID (1 for Ganglion, 0 for Cyton)')
    parser.add_argument('--duration', type=float, required=False, default=10.0, help='Recording duration in seconds')
    parser.add_argument('--out', type=str, required=False,
                        default=f'emg_ganglion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                        help='Output CSV filename')
    args = parser.parse_args()

    # -----------------------------
    # Initialize BrainFlow parameters
    # -----------------------------
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

    # -----------------------------
    # Start data streaming
    # -----------------------------
    board = BoardShim(args.board_id, params)
    board.prepare_session()
    board.start_stream()
    print(f"🔌 Streaming started for {args.duration} seconds on port {args.serial_port}")

    start = time.time()
    with open(args.out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp'] + [f'chan_{i+1}' for i in range(4)])

        while time.time() - start < args.duration:
            data = board.get_current_board_data(256)  # latest 256 samples
            if data.size == 0:
                time.sleep(0.01)
                continue

            # Transpose to iterate by sample
            for sample in data.T:
                ts = time.time()
                chans = [sample[i] if i < len(sample) else 0.0 for i in range(4)]
                writer.writerow([ts] + chans)

    # -----------------------------
    # Stop and release resources
    # -----------------------------
    board.stop_stream()
    board.release_session()
    print(f" Stream complete. Data saved to {args.out}")


if __name__ == "__main__":
    main()
