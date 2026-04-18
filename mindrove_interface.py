import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mindrove.board_shim import BoardIds, BoardShim, MindRoveInputParams


BASE_DIR = Path(__file__).resolve().parent
IMAGE_FOLDER = BASE_DIR / "hand-images"
SESSION_ROOT = BASE_DIR / "mindrove_sessions"
TRAINING_ROOT = BASE_DIR / "CSV-Files"
MINDROVE_IP = "192.168.4.1"
MINDROVE_PORT_CANDIDATES = (4210, 8888)
TRIALS_PER_GESTURE = 3
RECORDING_DURATION = 3.0
REST_DURATION = 2.0
SAMPLE_RATE_HZ = 500

GESTURES = [
    {"cue": "Close", "slug": "closed-hand", "marker": 2.0, "image": "close.png"},
    {"cue": "Open", "slug": "opened-hand", "marker": 3.0, "image": "open.png"},
    {"cue": "Spiderman", "slug": "spider-man", "marker": 4.0, "image": "spiderman.png"},
    {"cue": "Peace", "slug": "peace", "marker": 5.0, "image": "peace.png"},
    # Keep the existing "okay" artwork as a temporary visual alias for hang-loose.
    {"cue": "Hang Loose", "slug": "hang-loose", "marker": 6.0, "image": "okay.png"},
]

REST_MARKER = 1.0
MINDROVE_ORIGINAL_COLUMNS = [
    "Channel1", "Channel2", "Channel3", "Channel4", "Channel5", "Channel6", "Channel7", "Channel8",
    "FilteredChannel1", "FilteredChannel2", "FilteredChannel3", "FilteredChannel4", "FilteredChannel5", "FilteredChannel6", "FilteredChannel7", "FilteredChannel8",
    "GyroX", "GyroY", "GyroZ", "AccX", "AccY", "AccZ",
    "PPG1", "PPG2", "rawPPG1", "rawPPG2", "rawPPG3",
    "Hr", "Hrv", "Battery", "Trigger", "PhysicalTrigger", "AutoTrigger", "NumMeasurements", "Timestamp",
]
MARKER_TO_SLUG = {gesture["marker"]: gesture["slug"] for gesture in GESTURES}
SLUG_TO_CUE = {gesture["slug"]: gesture["cue"] for gesture in GESTURES}


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


def _detect_marker_column(df):
    for candidate in ("Trigger", "marker", "Hrv", "PhysicalTrigger", "AutoTrigger"):
        if candidate in df.columns:
            series = pd.to_numeric(df[candidate], errors="coerce").fillna(-1)
            if series.isin(list(MARKER_TO_SLUG) + [REST_MARKER]).any():
                return candidate
    raise ValueError("No marker column found in MindRove dataframe.")


def _extract_marker_events(df):
    marker_col = _detect_marker_column(df)
    marker_series = pd.to_numeric(df[marker_col], errors="coerce").fillna(-1).to_numpy(dtype=float)
    events = []
    previous_value = None

    for idx, value in enumerate(marker_series):
        rounded = round(float(value), 3)
        if rounded not in MARKER_TO_SLUG and rounded != REST_MARKER:
            continue
        if previous_value == rounded:
            continue
        previous_value = rounded
        events.append({"index": idx, "marker": rounded})

    return marker_col, events


def split_trials_from_dataframe(df, session_id, output_root=TRAINING_ROOT):
    """Split one full MindRove session into per-trial gesture files."""
    output_root = Path(output_root)
    marker_col, events = _extract_marker_events(df)
    saved_paths = []
    trial_counts = {gesture["slug"]: 0 for gesture in GESTURES}

    # Identify the timestamp column name if it exists
    ts_col = "Timestamp" if "Timestamp" in df.columns else None

    for current, nxt in zip(events, events[1:]):
        current_marker = current["marker"]
        next_marker = nxt["marker"]

        if current_marker not in MARKER_TO_SLUG or next_marker != REST_MARKER:
            continue

        slug = MARKER_TO_SLUG[current_marker]
        start_idx = int(current["index"])
        end_idx = int(nxt["index"])
        if end_idx <= start_idx:
            continue

        trial_df = df.iloc[start_idx:end_idx].copy()
        if trial_df.empty:
            continue

        trial_counts[slug] += 1
        trial_df["GestureLabel"] = slug
        trial_df["CueLabel"] = SLUG_TO_CUE[slug]
        trial_df["TrialNumber"] = trial_counts[slug]
        trial_df["SessionId"] = session_id
        trial_df["MarkerColumn"] = marker_col

        gesture_dir = output_root / slug
        gesture_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp-based filename
        # Fallback to session_id if timestamp extraction fails
        time_str = session_id

        if ts_col and ts_col in trial_df.columns:
            try:
                # Get the first valid timestamp from the trial
                ts_series = trial_df[ts_col].dropna()
                if not ts_series.empty:
                    ts_val = float(ts_series.iloc[0])

                    # Convert to datetime
                    # MindRove timestamps are usually Unix epoch (seconds)
                    # If value is very large (> 1e12), assume milliseconds
                    if ts_val > 1e12:
                        ts_val /= 1000.0

                    dt_obj = datetime.fromtimestamp(ts_val)
                    # Format: YYYYMMDD_HHMMSS_milliseconds
                    # Using milliseconds to ensure unique filenames if trials are close in time
                    time_str = dt_obj.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            except Exception:
                pass  # Keep fallback session_id if conversion fails

        # Filename: gesture_type_timestamp.csv
        # Example: closed-hand_20260411_130301_123.csv
        out_filename = f"{slug}_{time_str}.csv"
        out_path = gesture_dir / out_filename

        trial_df.to_csv(out_path, sep="\t", index=False)
        saved_paths.append(out_path)

    return saved_paths, trial_counts


def save_session_outputs(df, board, session_id, experiment_sequence):
    session_dir = SESSION_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    full_path = session_dir / "EMG_data.tsv"
    marker_path = session_dir / "EMG_training_markers.tsv"
    metadata_path = session_dir / "metadata.json"

    df.to_csv(full_path, sep="\t", index=False)
    marker_df = _build_marker_view_dataframe(df, board)
    marker_df.to_csv(marker_path, sep="\t", index=False)

    trial_paths, trial_counts = split_trials_from_dataframe(df, session_id=session_id)

    metadata = {
        "session_id": session_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sampling_rate_hz": SAMPLE_RATE_HZ,
        "recording_duration_s": RECORDING_DURATION,
        "rest_duration_s": REST_DURATION,
        "trials_per_gesture": TRIALS_PER_GESTURE,
        "gesture_order": experiment_sequence,
        "trial_counts": trial_counts,
        "full_data_file": str(full_path.relative_to(BASE_DIR)),
        "marker_file": str(marker_path.relative_to(BASE_DIR)),
        "trial_files": [str(path.relative_to(BASE_DIR)) for path in trial_paths],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Full session saved: {full_path}")
    print(f"Marker view saved: {marker_path}")
    print(f"Per-trial files written: {len(trial_paths)}")
    for slug, count in trial_counts.items():
        print(f"  - {slug}: {count}")


def initialize_board():
    last_error = None
    print("Using official MindRove SDK backend.")
    board_id = getattr(BoardIds, "MINDROVE_WIFI_BOARD", 0)

    default_params = MindRoveInputParams()
    try:
        return _try_connect(board_id, default_params, "MINDROVE_WIFI_BOARD (default params)")
    except Exception as exc:
        last_error = exc
        print(f"MINDROVE_WIFI_BOARD default connection failed: {exc}")
        _release_all_board_sessions_safely()

    print(f"Trying explicit MindRove WiFi settings at {MINDROVE_IP}...")
    for port in MINDROVE_PORT_CANDIDATES:
        params = MindRoveInputParams()
        params.ip_address = MINDROVE_IP
        params.ip_port = port
        params.timeout = 10
        try:
            return _try_connect(board_id, params, f"MINDROVE_WIFI_BOARD (ip={MINDROVE_IP}, port={port})")
        except Exception as exc:
            last_error = exc
            print(f"MINDROVE_WIFI_BOARD failed on port {port}: {exc}")
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


def main():
    from psychopy import core, event, visual

    experiment_sequence = [gesture["cue"] for gesture in GESTURES] * TRIALS_PER_GESTURE
    np.random.shuffle(experiment_sequence)
    marker_map = {"Rest": REST_MARKER, **{gesture["cue"]: gesture["marker"] for gesture in GESTURES}}
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    win = visual.Window([800, 600], monitor="testMonitor", units="deg", color=[-1, -1, -1])
    message = visual.TextStim(win, text="Ready?", height=2)
    fixation = visual.TextStim(win, text="+", height=2)

    stim_images = {}
    print(f"Loading images from: {IMAGE_FOLDER}")
    for gesture in GESTURES:
        image_path = IMAGE_FOLDER / gesture["image"]
        if image_path.exists():
            stim_images[gesture["cue"]] = visual.ImageStim(win, image=str(image_path), size=(10, 10))
            print(f"Loaded: {gesture['cue']}")
        else:
            print(f"WARNING: No image found for '{gesture['cue']}'. Will use text.")
            stim_images[gesture["cue"]] = None

    board = None

    try:
        board = initialize_board()
        time.sleep(2)

        for trial_stimulus in experiment_sequence:
            fixation.draw()
            win.flip()
            core.wait(0.5)

            message.text = f"Make Sign: {trial_stimulus}"
            if stim_images.get(trial_stimulus) is not None:
                message.pos = (0, -7)
                stim_images[trial_stimulus].draw()
            else:
                message.pos = (0, 0)

            message.draw()
            win.flip()

            if board:
                board.insert_marker(marker_map[trial_stimulus])

            core.wait(RECORDING_DURATION)

            message.text = "Rest"
            message.pos = (0, 0)
            message.draw()
            win.flip()

            if board:
                board.insert_marker(marker_map["Rest"])

            core.wait(REST_DURATION)

            if "escape" in event.getKeys():
                print("Escape pressed. Exiting...")
                break

    except Exception as exc:
        print(f"\nAn error occurred during the experiment: {exc}")
        import traceback

        traceback.print_exc()

    finally:
        print("Stopping Stream and Saving Data...")

        if board:
            try:
                data = board.get_board_data()
                df = _build_labeled_dataframe(data, board)
                save_session_outputs(df, board, session_id=session_id, experiment_sequence=experiment_sequence)
                board.stop_stream()
                board.release_session()
                print(f"Data saved successfully! Full shape: {df.shape}")
            except Exception as exc:
                print(f"Error saving data: {exc}")

        win.close()
        print("Done.")


if __name__ == "__main__":
    main()
