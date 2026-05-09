import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mindrove.board_shim import BoardIds, BoardShim, MindRoveInputParams


BASE_DIR = Path(__file__).resolve().parent
IMAGE_FOLDER = BASE_DIR / "hand-images"
TRAINING_ROOT = BASE_DIR / "CSV-Files"
MINDROVE_IP = "192.168.4.1"
MINDROVE_PORT_CANDIDATES = (4210, 8888)

SAMPLES_PER_GESTURE = 10
ACTION_DURATION = 5.0
READY_DURATION = 3.0
REST_DURATION = 2.0

GESTURES = [
    {"cue": "Close", "slug": "closed-hand", "marker": 2.0, "image": "close.png", "abbr": "CH"},
    {"cue": "Open", "slug": "opened-hand", "marker": 3.0, "image": "open.png", "abbr": "OH"},
    {"cue": "Spiderman", "slug": "spider-man", "marker": 4.0, "image": "spiderman.png", "abbr": "SP"},
    {"cue": "Peace", "slug": "peace", "marker": 5.0, "image": "peace.png", "abbr": "PE"},
    {"cue": "Hang Loose", "slug": "hang-loose", "marker": 6.0, "image": "okay.png", "abbr": "HL"},
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


def _extract_trial_by_marker(df, marker_value):
    marker_col, events = _extract_marker_events(df)
    for current, nxt in zip(events, events[1:]):
        if current["marker"] == marker_value and nxt["marker"] == REST_MARKER:
            start_idx = int(current["index"])
            end_idx = int(nxt["index"])
            if end_idx > start_idx:
                trial_df = df.iloc[start_idx:end_idx].copy()
                trial_df["MarkerColumn"] = marker_col
                return trial_df
    raise RuntimeError("Could not extract a valid trial segment from captured data.")


def _next_sample_number(gesture_dir: Path, abbr: str) -> int:
    pattern = re.compile(rf"^E{abbr}(\d{{2}})-\d{{4}}\.csv$", re.IGNORECASE)
    max_num = 0
    for existing in gesture_dir.glob(f"E{abbr}*.csv"):
        match = pattern.match(existing.name)
        if not match:
            continue
        max_num = max(max_num, int(match.group(1)))
    return max_num + 1


def _save_legacy_trial_file(trial_df: pd.DataFrame, slug: str, abbr: str, sample_number: int, mmdd: str) -> Path:
    gesture_dir = TRAINING_ROOT / slug
    gesture_dir.mkdir(parents=True, exist_ok=True)
    filename = f"E{abbr}{sample_number:02d}-{mmdd}.csv"
    out_path = gesture_dir / filename
    trial_df.to_csv(out_path, sep="\t", index=False)
    return out_path


def _capture_single_trial_dataframe(
    board,
    win,
    fixation,
    message,
    stim,
    cue_text,
    marker_value,
):
    from psychopy import core

    try:
        board.get_board_data()
    except Exception:
        pass

    message.text = f"Get Ready: {cue_text}"
    message.pos = (0, 0)
    message.draw()
    win.flip()
    core.wait(READY_DURATION)

    fixation.draw()
    win.flip()
    core.wait(0.3)

    message.text = f"Make Sign: {cue_text}"
    if stim is not None:
        message.pos = (0, -7)
        stim.draw()
    else:
        message.pos = (0, 0)
    message.draw()
    win.flip()

    board.insert_marker(marker_value)
    core.wait(ACTION_DURATION)

    message.text = "Rest"
    message.pos = (0, 0)
    message.draw()
    win.flip()
    board.insert_marker(REST_MARKER)
    core.wait(REST_DURATION)

    trial_data = board.get_board_data()
    return _build_labeled_dataframe(trial_data, board)


def run_legacy_collection(board, win, fixation, message, stim_images, marker_map):
    mmdd = datetime.now().strftime("%m%d")
    saved_files = []

    print("\n=== Legacy-Format Data Collection ===")
    print(f"Collecting {SAMPLES_PER_GESTURE} samples per gesture.")
    print(f"Each sample: {ACTION_DURATION:.0f}s action + {REST_DURATION:.0f}s rest.")

    for gesture in GESTURES:
        cue = gesture["cue"]
        slug = gesture["slug"]
        abbr = gesture["abbr"]
        marker_value = marker_map[cue]

        gesture_dir = TRAINING_ROOT / slug
        gesture_dir.mkdir(parents=True, exist_ok=True)
        next_number = _next_sample_number(gesture_dir, abbr)

        print(f"\nGesture: {cue} ({abbr})")
        for sample_idx in range(SAMPLES_PER_GESTURE):
            display_num = sample_idx + 1
            print(f"  Recording sample {display_num}/{SAMPLES_PER_GESTURE}...")

            captured = _capture_single_trial_dataframe(
                board=board,
                win=win,
                fixation=fixation,
                message=message,
                stim=stim_images.get(cue),
                cue_text=cue,
                marker_value=marker_value,
            )
            trial_df = _extract_trial_by_marker(captured, marker_value)
            trial_df["GestureLabel"] = slug
            trial_df["CueLabel"] = cue
            trial_df["SampleNumber"] = next_number + sample_idx
            trial_df["CollectionDate"] = mmdd

            saved = _save_legacy_trial_file(
                trial_df=trial_df,
                slug=slug,
                abbr=abbr,
                sample_number=next_number + sample_idx,
                mmdd=mmdd,
            )
            saved_files.append(saved)
            print(f"    Saved: {saved}")

    print(f"\nFinished. Total files saved: {len(saved_files)}")
    return saved_files


def main():
    from psychopy import visual

    marker_map = {"Rest": REST_MARKER, **{gesture["cue"]: gesture["marker"] for gesture in GESTURES}}

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
        run_legacy_collection(
            board=board,
            win=win,
            fixation=fixation,
            message=message,
            stim_images=stim_images,
            marker_map=marker_map,
        )
    except Exception as exc:
        print(f"\nAn error occurred during collection: {exc}")
        import traceback

        traceback.print_exc()
    finally:
        print("Stopping stream...")
        if board:
            try:
                board.stop_stream()
                board.release_session()
            except Exception:
                pass
        win.close()
        print("Done.")


if __name__ == "__main__":
    main()
