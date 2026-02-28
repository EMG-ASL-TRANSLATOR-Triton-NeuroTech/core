"""
Prepare RandomForest EMG dataset from gesture CSV files.

Input:
- Tab/comma delimited MindRove gesture CSV files in CSV-Files/

Output:
- RandomForest dataset: RandomForest/DATASET EMG MINDROVE/FORMATTED/subjek_FORMATTED.csv
  Columns: CH1..CH8,Target
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Ordered from most specific to least specific for safer matching.
LABEL_KEYWORDS = [
    ("opened-hand", 2),
    ("open-hand", 2),
    ("open_hand", 2),
    ("closed-hand", 1),
    ("closed_hand", 1),
    ("index_finger", 3),
    ("ring_finger", 4),
    ("pinky_finger", 5),
    ("spider-man", 6),
    ("spiderman", 6),
    ("peace", 7),
    ("hang-loose", 8),
    ("hang_loose", 8),
    ("open", 2),
    ("closed", 1),
    ("index", 3),
    ("ring", 4),
    ("pinky", 5),
]

RAW_CHANNELS = [f"Channel{i}" for i in range(1, 9)]
GESTURE_NAMES = {
    1: "closed_hand",
    2: "open_hand",
    3: "index_finger",
    4: "ring_finger",
    5: "pinky_finger",
    6: "spider_man",
    7: "peace",
    8: "hang_loose",
}


def infer_label(csv_path: Path, input_root: Path) -> int | None:
    rel_parent = str(csv_path.parent.relative_to(input_root)).lower()
    name = csv_path.stem.lower()
    search_space = f"{rel_parent}/{name}"
    for keyword, label in LABEL_KEYWORDS:
        if keyword in search_space:
            return label
    return None


def read_emg_csv(csv_path: Path) -> pd.DataFrame:
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
    sep = "\t" if "\t" in first_line else ","
    return pd.read_csv(csv_path, sep=sep)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    random_forest_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Build RandomForest-ready EMG dataset")
    parser.add_argument(
        "--input-dir",
        default=str(project_root / "CSV-Files"),
        help="Source folder with raw gesture CSVs",
    )
    parser.add_argument(
        "--rf-output",
        default=str(random_forest_dir / "DATASET EMG MINDROVE" / "FORMATTED" / "subjek_FORMATTED.csv"),
        help="Output CSV for RandomForest training",
    )
    parser.add_argument(
        "--gesture-dir",
        default=str(random_forest_dir / "DATASET EMG MINDROVE" / "FORMATTED" / "by_gesture"),
        help="Output directory for per-gesture CSV files",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    rf_output = Path(args.rf_output)
    gesture_dir = Path(args.gesture_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    rf_output.parent.mkdir(parents=True, exist_ok=True)
    gesture_dir.mkdir(parents=True, exist_ok=True)
    for old_file in gesture_dir.glob("*.csv"):
        old_file.unlink()

    csv_files = sorted(input_dir.rglob("*.csv"))
    rf_frames: list[pd.DataFrame] = []
    gesture_frames: dict[int, list[pd.DataFrame]] = {k: [] for k in GESTURE_NAMES}

    used_files = 0
    skipped_files = []

    for csv_path in csv_files:
        label = infer_label(csv_path, input_dir)
        if label is None:
            skipped_files.append(str(csv_path.relative_to(input_dir)))
            continue

        df = read_emg_csv(csv_path)
        missing = [c for c in RAW_CHANNELS if c not in df.columns]
        if missing:
            skipped_files.append(f"{csv_path.relative_to(input_dir)} (missing columns: {missing})")
            continue

        used_files += 1

        # Build RandomForest frame (sample-level labels)
        rf_df = df[RAW_CHANNELS].copy()
        rf_df.columns = [f"CH{i}" for i in range(1, 9)]
        rf_df["Target"] = int(label)
        rf_frames.append(rf_df)
        if label in gesture_frames:
            gesture_frames[label].append(rf_df)

    if not rf_frames:
        raise RuntimeError(
            "No labeled source files were processed. Check filenames include gesture keywords "
            "(open/closed/index/ring/pinky)."
        )

    rf_master = pd.concat(rf_frames, ignore_index=True)
    rf_master.to_csv(rf_output, index=False)

    gesture_written = []
    for label, frames in gesture_frames.items():
        if not frames:
            continue
        gesture_df = pd.concat(frames, ignore_index=True)
        out_path = gesture_dir / f"{GESTURE_NAMES[label]}.csv"
        gesture_df.to_csv(out_path, index=False)
        gesture_written.append((label, out_path, len(gesture_df)))

    print("=== Dataset Formatting Complete ===")
    print(f"Input directory: {input_dir}")
    print(f"Source files used: {used_files}")
    print(f"Skipped files: {len(skipped_files)}")
    if skipped_files:
        for item in skipped_files:
            print(f"  - {item}")

    print(f"\nRandomForest output: {rf_output}")
    print(f"Rows: {len(rf_master)} | Columns: {list(rf_master.columns)}")
    print("Target counts:")
    print(rf_master["Target"].value_counts().sort_index())

    print(f"\nPer-gesture outputs: {gesture_dir}")
    for label, out_path, rows in gesture_written:
        print(f"  - Target {label} ({GESTURE_NAMES[label]}): {rows} rows -> {out_path.name}")


if __name__ == "__main__":
    main()
