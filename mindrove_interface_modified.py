"""
MindRove EMG Data Collection Script — High & Medium Impact Improvements

High impact:
- More trials per sign (15 instead of 5)
- Longer rest between gestures (3s instead of 2s)
- Longer preparation window (1.5s instead of 0.5s)

Medium impact:
- Signal quality check before each trial
- Countdown before each gesture
"""

import os
import time
import sys
from psychopy import visual, core, event
import numpy as np
import pandas as pd

from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds

# --- CONFIGURATION ---
signs = ['Close', 'Open', 'Spiderman', 'Peace', 'Okay']
trials_per_sign = 15          # was 5
recording_duration = 3.0
rest_duration = 3.0           # was 2.0
prep_wait = 1.5               # was 0.5
SIGNAL_QUALITY_THRESHOLD = 50

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, "hand-images")
MINDROVE_IP = "192.168.4.1"
MINDROVE_PORT_CANDIDATES = (4210, 8888)
MINDROVE_ORIGINAL_COLUMNS = [
    "Channel1", "Channel2", "Channel3", "Channel4", "Channel5", "Channel6", "Channel7", "Channel8",
    "FilteredChannel1", "FilteredChannel2", "FilteredChannel3", "FilteredChannel4", "FilteredChannel5", "FilteredChannel6", "FilteredChannel7", "FilteredChannel8",
    "GyroX", "GyroY", "GyroZ", "AccX", "AccY", "AccZ",
    "PPG1", "PPG2", "rawPPG1", "rawPPG2", "rawPPG3",
    "Hr", "Hrv", "Battery", "Trigger", "PhysicalTrigger", "AutoTrigger", "NumMeasurements", "Timestamp"
]

marker_map = {
    'Rest': 1.0, 'Close': 2.0, 'Open': 3.0,
    'Spiderman': 4.0, 'Peace': 5.0, 'Okay': 6.0
}


# --- HELPER: Signal quality check ---
def check_signal_quality(board, fs=500, window_sec=0.5, threshold=SIGNAL_QUALITY_THRESHOLD):
    """
    Peek at the last window_sec of data and check if signal amplitude looks reasonable.
    Returns (is_good: bool, amplitude: float)
    """
    try:
        n_samples = int(fs * window_sec)
        recent = board.get_current_board_data(n_samples)
        ch1 = recent[0]
        amplitude = float(np.max(ch1) - np.min(ch1))
        return amplitude > threshold, amplitude
    except Exception:
        return True, -1.0


# --- BOARD HELPERS ---
def _try_connect(board_id, params, label):
    board = BoardShim(board_id, params)
    print(f"Trying {label}...")
    board.prepare_session()
    board.start_stream(450000)
    print(f"Connected successfully with {label}!")
    return board


def _release_all_board_sessions_safely():
    try:
        BoardShim.release_all_sessions()
    except Exception:
        pass


def _build_labeled_dataframe(data, board):
    board_id = getattr(BoardIds, "MINDROVE_WIFI_BOARD", 0)
    try:
        board_id = board.get_board_id()
    except Exception:
        pass

    num_rows = data.shape[0]

    if num_rows == len(MINDROVE_ORIGINAL_COLUMNS):
        df = pd.DataFrame(data.T, columns=MINDROVE_ORIGINAL_COLUMNS)

        trigger_all_invalid = (df["Trigger"] == -1).all()
        hrv_has_markers = df["Hrv"].nunique(dropna=True) > 1
        if trigger_all_invalid and hrv_has_markers:
            df["Trigger"] = df["Hrv"]

        ts_all_invalid = (df["Timestamp"] == -1).all()
        hr_monotonic = (df["Hr"].diff().fillna(0) >= 0).mean() > 0.99
        hr_has_range = (df["Hr"].max() - df["Hr"].min()) > 1
        if ts_all_invalid and hr_monotonic and hr_has_range:
            df["Timestamp"] = df["Hr"]

        return df

    columns = [f"ch_{i}" for i in range(num_rows)]
    try:
        ts_idx = BoardShim.get_timestamp_channel(board_id)
        if 0 <= ts_idx < num_rows:
            columns[ts_idx] = "timestamp"
    except Exception:
        pass
    try:
        marker_idx = BoardShim.get_marker_channel(board_id)
        if 0 <= marker_idx < num_rows:
            columns[marker_idx] = "marker"
    except Exception:
        pass

    return pd.DataFrame(data.T, columns=columns)


def _build_marker_view_dataframe(df, board):
    marker_cols = [c for c in ["Timestamp", "Trigger", "timestamp", "marker"] if c in df.columns]
    board_id = getattr(BoardIds, "MINDROVE_WIFI_BOARD", 0)
    try:
        board_id = board.get_board_id()
    except Exception:
        pass
    try:
        eeg_idxs = BoardShim.get_eeg_channels(board_id)
        eeg_cols = [f"ch_{idx}" for idx in eeg_idxs if f"ch_{idx}" in df.columns]
    except Exception:
        eeg_cols = []
    selected = marker_cols + eeg_cols
    if not selected:
        return df.copy()
    return df[selected].copy()


def initialize_board():
    last_error = None
    print("Using official MindRove SDK backend.")
    board_id = getattr(BoardIds, "MINDROVE_WIFI_BOARD", 0)

    default_params = MindRoveInputParams()
    try:
        return _try_connect(board_id, default_params, "MINDROVE_WIFI_BOARD (default params)")
    except Exception as e:
        last_error = e
        print(f"MINDROVE_WIFI_BOARD default connection failed: {e}")
        _release_all_board_sessions_safely()

    print(f"Trying explicit MindRove WiFi settings at {MINDROVE_IP}...")
    for port in MINDROVE_PORT_CANDIDATES:
        params = MindRoveInputParams()
        params.ip_address = MINDROVE_IP
        params.ip_port = port
        params.timeout = 10
        try:
            return _try_connect(board_id, params, f"MINDROVE_WIFI_BOARD (ip={MINDROVE_IP}, port={port})")
        except Exception as e:
            last_error = e
            print(f"MINDROVE_WIFI_BOARD failed on port {port}: {e}")
            _release_all_board_sessions_safely()

    print("\n\nCRITICAL CONNECTION ERROR:")
    print("MindRove SDK backend could not connect to the device.")
    print(f"Last error details: {last_error}")
    print("---------------------------------------------------")
    print("CHECKLIST:")
    print("1. Make sure MindRove Connect GUI is fully closed before running this script.")
    print("2. Confirm laptop is connected to MindRove WiFi network.")
    print("3. On Windows firewall, allow python.exe on BOTH private and public networks.")
    print("4. Reconnect WiFi and restart device if incoming throughput is zero.")
    print("---------------------------------------------------")
    input("Press Enter to Exit...")
    sys.exit(1)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

# 1. Build experiment sequence
experiment_sequence = signs * trials_per_sign
np.random.shuffle(experiment_sequence)

# 2. Setup PsychoPy Window
win = visual.Window([800, 600], monitor="testMonitor", units="deg", color=[-1, -1, -1])
message = visual.TextStim(win, text="Ready?", height=2)
fixation = visual.TextStim(win, text="+", height=2)
quality_warn = visual.TextStim(win, text="", height=1.5, color=[1, 0.5, 0], pos=(0, -5))

# Pre-load images
stim_images = {}
extension = '.png'
print(f"Loading images from: {IMAGE_FOLDER}")
for sign in signs:
    image_path = os.path.join(IMAGE_FOLDER, sign.lower() + extension)
    if os.path.exists(image_path):
        stim_images[sign] = visual.ImageStim(win, image=image_path, size=(10, 10))
        print(f"  Loaded: {sign}")
    else:
        print(f"  WARNING: No image for '{sign}'. Will use text.")
        stim_images[sign] = None

board = None

try:
    # 3. Connect to board
    board = initialize_board()
    time.sleep(2)

    # 4. Main experiment loop
    for trial_num, trial_stimulus in enumerate(experiment_sequence, start=1):

        # --- SIGNAL QUALITY CHECK ---
        is_good, amplitude = check_signal_quality(board)
        if not is_good:
            quality_warn.text = f"Weak signal (amp={amplitude:.0f}). Check armband contact."
            fixation.draw()
            quality_warn.draw()
            win.flip()
            core.wait(prep_wait)
        else:
            fixation.draw()
            win.flip()
            core.wait(prep_wait)

        # --- COUNTDOWN ---
        for count in [3, 2, 1]:
            message.text = f"Get ready... {count}"
            message.pos = (0, 0)
            message.draw()
            win.flip()
            core.wait(1.0)

        # --- STIMULUS ---
        message.text = f"Make Sign: {trial_stimulus}  ({trial_num}/{len(experiment_sequence)})"
        if stim_images.get(trial_stimulus):
            message.pos = (0, -7)
            stim_images[trial_stimulus].draw()
        else:
            message.pos = (0, 0)
        message.draw()
        win.flip()

        if board:
            board.insert_marker(marker_map[trial_stimulus])

        core.wait(recording_duration)

        # --- REST ---
        message.text = "Rest"
        message.pos = (0, 0)
        message.draw()
        win.flip()

        if board:
            board.insert_marker(marker_map['Rest'])

        core.wait(rest_duration)

        if 'escape' in event.getKeys():
            print("Escape pressed. Exiting...")
            break

except Exception as e:
    print(f"\nAn error occurred during the experiment: {e}")
    import traceback
    traceback.print_exc()

finally:
    print("Stopping Stream and Saving Data...")

    if board:
        try:
            data = board.get_board_data()
            board.stop_stream()
            board.release_session()

            df = _build_labeled_dataframe(data, board)
            df.to_csv("EMG_data.csv", sep='\t', index=False)

            marker_df = _build_marker_view_dataframe(df, board)
            marker_df.to_csv("EMG_training_markers.csv", sep='\t', index=False)

            print(f"Data saved successfully! Full shape: {df.shape}")
            print(f"Marker view saved: 'EMG_training_markers.csv' Shape: {marker_df.shape}")

        except Exception as e:
            print(f"Error saving data: {e}")

    win.close()
    print("Done.")
