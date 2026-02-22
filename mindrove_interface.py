import os
import time
import sys
from psychopy import visual, core, event
import numpy as np
import pandas as pd

from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds

# --- CONFIGURATION ---
signs = ['Close', 'Open', 'Spiderman', 'Peace', 'Okay']
trials_per_sign = 5
recording_duration = 3.0
rest_duration = 2.0
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
    """Return a dataframe with named columns including timestamp/marker when available."""
    board_id = getattr(BoardIds, "MINDROVE_WIFI_BOARD", 0)
    try:
        board_id = board.get_board_id()
    except Exception:
        pass

    num_rows = data.shape[0]

    # Prefer original MindRove schema when dimensions match exactly.
    if num_rows == len(MINDROVE_ORIGINAL_COLUMNS):
        df = pd.DataFrame(data.T, columns=MINDROVE_ORIGINAL_COLUMNS)

        # Trigger/Timestamp remain constant placeholders (-1).
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
    """Return compact dataframe focused on marker timing."""
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

    # First attempt exactly as docs (defaults), then explicit IP/port candidates.
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

# --- MAIN EXECUTION ---

# 1. Initialize data/sequence
experiment_sequence = signs * trials_per_sign
np.random.shuffle(experiment_sequence)
marker_map = {'Rest': 1.0, 'Close': 2.0, 'Open': 3.0, 'Spiderman': 4.0, 'Peace': 5.0, 'Okay': 6.0}

# 2. Setup PsychoPy Window
# NOTE: defined outside try block so we can close it safely
win = visual.Window([800, 600], monitor="testMonitor", units="deg", color=[-1, -1, -1])
message = visual.TextStim(win, text="Ready?", height=2)
fixation = visual.TextStim(win, text="+", height=2)

# Pre-load images so stimulus timing doesn't stutter.
stim_images = {}
extension = '.png'
print(f"Loading images from: {IMAGE_FOLDER}")
for sign in signs:
    image_path = os.path.join(IMAGE_FOLDER, sign.lower() + extension)
    if os.path.exists(image_path):
        stim_images[sign] = visual.ImageStim(win, image=image_path, size=(10, 10))
        print(f"Loaded: {sign}")
    else:
        print(f"WARNING: No image found for '{sign}'. Will use text.")
        stim_images[sign] = None

board = None  # Initialize to None for safety

try:
    # 3. Initialize Board
    board = initialize_board()
    time.sleep(2) # Let signals settle

    # 4. Start Experiment Loop
    for trial_stimulus in experiment_sequence:
        
        # --- PHASE 1: PREPARATION ---
        fixation.draw()
        win.flip()
        core.wait(0.5)
        
        # --- PHASE 2: STIMULUS ---
        message.text = f"Make Sign: {trial_stimulus}"

        # Draw image if available, otherwise fall back to text-only display.
        if stim_images.get(trial_stimulus) is not None:
            message.pos = (0, -7)
            stim_images[trial_stimulus].draw()
        else:
            message.pos = (0, 0)

        message.draw()
        win.flip() 
        
        # Send Marker
        if board:
            board.insert_marker(marker_map[trial_stimulus])
        
        core.wait(recording_duration)
        
        # --- PHASE 3: REST ---
        message.text = "Rest"
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
    
    # Safely close board connection
    if board:
        try:
            data = board.get_board_data()
            board.stop_stream()
            board.release_session()
            
            # Save full labeled data
            df = _build_labeled_dataframe(data, board)
            df.to_csv("EMG_data.csv", sep='\t', index=False)

            # Save marker/timestamp-focused view (+ EEG channels when available)
            marker_df = _build_marker_view_dataframe(df, board)
            marker_df.to_csv("EMG_training_markers.csv", sep='\t', index=False)

            print(f"Data saved successfully! Full shape: {df.shape}")
            print(f"Marker view saved: 'EMG_training_markers.csv' Shape: {marker_df.shape}")
            
        except Exception as e:
            print(f"Error saving data: {e}")
            
    # Close Window
    win.close()
    print("Done.")