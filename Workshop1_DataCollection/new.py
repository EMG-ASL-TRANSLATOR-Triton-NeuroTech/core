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
import sys
from pylsl import StreamInlet, resolve_stream

# ---------------------------------------------------------------------
#  Ganglion (BrainFlow) Mode
# ---------------------------------------------------------------------
def ganglion_mode(args, inlet):
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
    print("Preparing BrainFlow session...", flush=True)
    board.prepare_session()
    print("BrainFlow session prepared.", flush=True)
    print("Starting BrainFlow stream...", flush=True)
    board.start_stream()
    # wait a short time to allow the board to start producing data
    time.sleep(0.5)
    print(f"Ganglion stream started for {args.duration}s on port {args.serial_port}")

    dl = DataLogger(out_dir="data",
                    session_name="ganglion_" + time.strftime("%Y%m%d_%H%M%S"),
                    channels=4)
    print(f"Writing ganglion data to: {dl.filename}", flush=True)
    
    dl_lsl  = DataLogger(out_dir="data",
              session_name="lsl_" + time.strftime("%Y%m%d_%H%M%S"),
              channels=4)
    print(f"Writing LSL data to: {dl_lsl.filename}", flush=True)

    # Counters for debug
    written_count = 0
    lsl_written_count = 0

    written_count = 0
    lsl_written_count = 0

    # Stream loop is wrapped in try/finally to ensure session cleanup
    start = time.time()
    try:
        while time.time() - start < args.duration:
            data = board.get_current_board_data(1)

            # Try to read from LSL if an inlet was provided
            if inlet is not None:
                try:
                    sample_lsl, timestamp_lsl = inlet.pull_sample(timeout=0.0)
                except Exception:
                    sample_lsl, timestamp_lsl = None, None

                if sample_lsl:
                    ts = timestamp_lsl
                    chans = [float(sample_lsl[i]) if i < len(sample_lsl) else 0.0 for i in range(4)]
                    dl_lsl.write_row(ts, chans, label="")  # labels added later
                    lsl_written_count += 1
                    # print a short progress every 50 LSL samples
                    if lsl_written_count % 50 == 0:
                        print(f"LSL rows written: {lsl_written_count}", flush=True)

            if data.size == 0:
                time.sleep(0.01)
                continue

            # transpose to get one row per sample
            for sample in data.T:
                ts = time.time()
                chans = [float(sample[i]) if i < len(sample) else 0.0 for i in range(4)]
                dl.write_row(ts, chans, label="")  # labels added later
                written_count += 1
                if written_count % 100 == 0:
                    print(f"Ganglion rows written: {written_count}", flush=True)

        
    except Exception as e:
        print("Error during streaming loop:", e, flush=True)
    finally:
        try:
            board.stop_stream()
            print("BrainFlow stream stopped.", flush=True)
        except Exception:
            pass
        try:
            board.release_session()
            print("BrainFlow session released.", flush=True)
        except Exception:
            pass
        print("Ganglion stream complete.")
        print(f"Total ganglion rows written: {written_count}", flush=True)
        print(f"Total LSL rows written: {lsl_written_count}", flush=True)

def get_signal(timeout=2.0):
    # Resolve EMG-type streams on the network using a polling loop
    # because `resolve_stream` does not accept a `timeout` kwarg on all pylsl versions.
    start = time.time()
    streams = []
    while time.time() - start < timeout:
        try:
            streams = resolve_stream('type', 'EMG')
        except Exception:
            streams = []

        if streams:
            break

        time.sleep(0.1)

    if not streams:
        print('No LSL EMG streams found.')
        return None

    # Prefer a named test stream, otherwise take the first available
    for stream in streams:
        print('Found LSL stream:', stream.name())
        if stream.name() == "EMG Test 1":
            inlet = StreamInlet(stream, max_buflen=1)
            print('received stream "EMG Test 1"')
            return inlet

    # Fallback to the first EMG stream
    first = streams[0]
    inlet = StreamInlet(first, max_buflen=1)
    print('received first EMG stream:', first.name())
    return inlet


def test_lsl(inlet, test_duration=5.0):
    """Pull samples from the provided LSL inlet for `test_duration` seconds
    and save them to a CSV using DataLogger. Returns the number of rows written.
    """
    if inlet is None:
        print("No inlet provided to test_lsl().", flush=True)
        return 0

    dl_test = DataLogger(out_dir="data",
                         session_name="lsl_test_" + time.strftime("%Y%m%d_%H%M%S"),
                         channels=4)
    print(f"Writing LSL test data to: {dl_test.filename}", flush=True)

    count = 0
    start = time.time()
    while time.time() - start < test_duration:
        try:
            sample, ts = inlet.pull_sample(timeout=1.0)
        except Exception:
            sample, ts = None, None

        if sample:
            chans = [float(sample[i]) if i < len(sample) else 0.0 for i in range(4)]
            dl_test.write_row(ts, chans, label="test")
            count += 1
            if count % 50 == 0:
                print(f"LSL test rows written: {count}", flush=True)

    print(f"LSL test finished, rows written: {count}", flush=True)
    return count

# ---------------------------------------------------------------------
#  Main CLI Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EMG Data Collection Demo")

    # --- Common parameters ---
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--lsl-test-duration", type=float, default=5.0,
                        help="Seconds to test and record LSL stream before BrainFlow")

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

    # Ensure serial port provided for BrainFlow Ganglion mode
    assert args.serial_port, "Provide --serial-port for Ganglion mode"

    # Try to obtain an LSL inlet (may return None if no stream is available)
    inlet = get_signal()

    # If an LSL inlet is available, run a short test record first so user
    # can confirm LSL is working independently of BrainFlow hardware.
    if inlet is not None:
        print(f"LSL inlet acquired. Running LSL test for {args.lsl_test_duration}s...", flush=True)
        test_count = test_lsl(inlet, test_duration=args.lsl_test_duration)
        print(f"LSL test recorded {test_count} rows. Proceeding to BrainFlow.", flush=True)
    else:
        print("No LSL inlet found. Skipping LSL test and proceeding to BrainFlow.", flush=True)

    ganglion_mode(args, inlet)




