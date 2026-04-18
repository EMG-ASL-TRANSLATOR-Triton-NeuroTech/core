"""
Train a leak-resistant RandomForest gesture classifier from MindRove trial files.

This pipeline reads per-trial CSV/TSV files from ../CSV-Files/<gesture>/,
extracts EMG features within each trial, evaluates with grouped splits so
windows from the same trial never land in both train and test, and saves a
deployable model bundle.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


MINDROVE_SAMPLING_RATE = 500
WINDOW_MS = 250.0
STRIDE_MS = 50.0
RAW_CHANNELS = [f"Channel{i}" for i in range(1, 9)]
FALLBACK_CHANNELS = [f"CH{i}" for i in range(1, 9)]
LABEL_TO_ID = {
    "closed-hand": 1,
    "opened-hand": 2,
    "spider-man": 3,
    "peace": 4,
    "hang-loose": 5,
    "okay": 5,
}
ID_TO_LABEL = {
    1: "closed-hand",
    2: "opened-hand",
    3: "spider-man",
    4: "peace",
    5: "hang-loose",
}


@dataclass
class TrialRecord:
    path: Path
    label_name: str
    label_id: int
    group_id: str
    subject_id: str
    session_id: str


def infer_label_from_path(path: Path, input_root: Path) -> str | None:
    search_parts = [part.lower() for part in path.relative_to(input_root).parts]
    for part in reversed(search_parts[:-1]):
        if part in LABEL_TO_ID:
            return "hang-loose" if part == "okay" else part

    stem = path.stem.lower()
    for key in LABEL_TO_ID:
        if key in stem:
            return "hang-loose" if key == "okay" else key
    return None


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        first_line = handle.readline()
    return "\t" if "\t" in first_line else ","


def read_trial(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=detect_delimiter(path))


def get_channel_columns(df: pd.DataFrame) -> list[str]:
    if all(column in df.columns for column in RAW_CHANNELS):
        return RAW_CHANNELS
    if all(column in df.columns for column in FALLBACK_CHANNELS):
        return FALLBACK_CHANNELS
    raise ValueError(f"Could not find expected EMG channels in columns: {list(df.columns)}")


def collect_trial_files(input_root: Path) -> list[TrialRecord]:
    trial_files = []
    for path in sorted(input_root.rglob("*.csv")):
        label_name = infer_label_from_path(path, input_root)
        if label_name is None:
            continue
        relative_parts = path.relative_to(input_root).parts
        subject_id = relative_parts[0] if len(relative_parts) > 2 else "unknown"
        session_id = path.stem.split("_trial")[0]
        trial_files.append(
            TrialRecord(
                path=path,
                label_name=label_name,
                label_id=LABEL_TO_ID[label_name],
                group_id=str(path.relative_to(input_root)),
                subject_id=subject_id,
                session_id=session_id,
            )
        )
    if not trial_files:
        raise FileNotFoundError(f"No trial CSV files found under {input_root}")
    return trial_files


def preprocess_signals(
    data: np.ndarray,
    fs: float = MINDROVE_SAMPLING_RATE,
    bandpass_low: float = 20.0,
    bandpass_high: float = 200.0,
    notch_freq: float = 60.0,
) -> np.ndarray:
    nyquist = fs / 2
    high_cutoff = min(bandpass_high, nyquist - 1)
    if high_cutoff <= bandpass_low:
        raise ValueError("Invalid bandpass settings for the configured sampling rate.")

    b_bp, a_bp = butter(4, [bandpass_low / nyquist, high_cutoff / nyquist], btype="band")
    filtered = filtfilt(b_bp, a_bp, data, axis=0)

    b_notch, a_notch = iirnotch(notch_freq, 30, fs)
    return filtfilt(b_notch, a_notch, filtered, axis=0)


def extract_features_from_window(window: np.ndarray, prefix: str) -> dict[str, float]:
    features = {}
    for idx in range(window.shape[1]):
        channel = window[:, idx]
        channel_prefix = f"{prefix}{idx + 1}"
        features[f"{channel_prefix}_RMS"] = float(np.sqrt(np.mean(channel ** 2)))
        features[f"{channel_prefix}_VAR"] = float(np.var(channel))
        features[f"{channel_prefix}_MAV"] = float(np.mean(np.abs(channel)))
        features[f"{channel_prefix}_WL"] = float(np.sum(np.abs(np.diff(channel))))
        features[f"{channel_prefix}_IEMG"] = float(np.sum(np.abs(channel)))
        features[f"{channel_prefix}_ZC"] = float(np.sum((channel[:-1] * channel[1:]) < 0))
        features[f"{channel_prefix}_SSC"] = float(
            np.sum(((channel[1:-1] - channel[:-2]) * (channel[1:-1] - channel[2:])) > 0)
        )
    return features


def build_feature_table(
    trial_records: list[TrialRecord],
    fs: float = MINDROVE_SAMPLING_RATE,
    window_ms: float = WINDOW_MS,
    stride_ms: float = STRIDE_MS,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    window_samples = max(1, int(window_ms / 1000 * fs))
    stride_samples = max(1, int(stride_ms / 1000 * fs))

    for record in trial_records:
        df = read_trial(record.path)
        channel_cols = get_channel_columns(df)
        filtered = preprocess_signals(df[channel_cols].to_numpy(dtype=float), fs=fs)
        if len(filtered) < window_samples:
            continue

        for start in range(0, len(filtered) - window_samples + 1, stride_samples):
            stop = start + window_samples
            window = filtered[start:stop]
            feature_row = extract_features_from_window(window, prefix="CH")
            feature_row["Target"] = record.label_id
            feature_row["LabelName"] = record.label_name
            feature_row["GroupId"] = record.group_id
            feature_row["SubjectId"] = record.subject_id
            feature_row["SessionId"] = record.session_id
            rows.append(feature_row)

    if not rows:
        raise RuntimeError("No feature windows were extracted. Check input files and sampling settings.")
    return pd.DataFrame(rows)


def evaluate_with_group_holdout(
    features_df: pd.DataFrame,
    random_state: int = 42,
) -> tuple[float, pd.DataFrame, dict[str, int]]:
    feature_cols = [col for col in features_df.columns if col.startswith("CH")]
    X = features_df[feature_cols]
    y = features_df["Target"]
    groups = features_df["GroupId"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    labels = sorted(y.unique())
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    confusion_df = pd.DataFrame(matrix, index=labels, columns=labels)
    counts = y.value_counts().sort_index().to_dict()
    return accuracy, confusion_df, counts


def cross_validate_by_group(
    features_df: pd.DataFrame,
    random_state: int = 42,
) -> list[float]:
    feature_cols = [col for col in features_df.columns if col.startswith("CH")]
    X = features_df[feature_cols]
    y = features_df["Target"]
    groups = features_df["GroupId"]
    unique_groups = pd.Series(groups.unique())
    n_splits = min(5, len(unique_groups))
    if n_splits < 2:
        return []

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    for train_idx, test_idx in splitter.split(X, y, groups):
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        predictions = model.predict(X.iloc[test_idx])
        scores.append(float(accuracy_score(y.iloc[test_idx], predictions)))
    return scores


def fit_full_model(features_df: pd.DataFrame, random_state: int = 42) -> tuple[RandomForestClassifier, list[str]]:
    feature_cols = [col for col in features_df.columns if col.startswith("CH")]
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(features_df[feature_cols], features_df["Target"])
    return model, feature_cols


def save_model_bundle(
    model: RandomForestClassifier,
    feature_columns: list[str],
    output_path: Path,
    window_ms: float,
    stride_ms: float,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "sampling_rate_hz": MINDROVE_SAMPLING_RATE,
        "window_ms": window_ms,
        "stride_ms": stride_ms,
        "label_map": ID_TO_LABEL,
        "feature_columns": feature_columns,
    }
    joblib.dump({"model": model, "metadata": metadata}, output_path)
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent

    parser = argparse.ArgumentParser(description="Train a grouped RandomForest model from MindRove trial files.")
    parser.add_argument(
        "--input-dir",
        default=str(project_root / "CSV-Files"),
        help="Root folder containing per-gesture MindRove trial files.",
    )
    parser.add_argument(
        "--model-out",
        default=str(base_dir / "models" / "mindrove_rf.joblib"),
        help="Output path for the trained model bundle.",
    )
    parser.add_argument(
        "--window-ms",
        type=float,
        default=WINDOW_MS,
        help="Window length in milliseconds for feature extraction.",
    )
    parser.add_argument(
        "--stride-ms",
        type=float,
        default=STRIDE_MS,
        help="Stride length in milliseconds for feature extraction.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_dir)
    model_out = Path(args.model_out)

    trial_records = collect_trial_files(input_root)
    features_df = build_feature_table(trial_records, window_ms=args.window_ms, stride_ms=args.stride_ms)

    print("=== MindRove RandomForest Training ===")
    print(f"Trial files used: {len(trial_records)}")
    print(f"Windowed samples: {len(features_df)}")
    print("Class counts:")
    print(features_df["LabelName"].value_counts().sort_index())

    holdout_accuracy, confusion_df, class_counts = evaluate_with_group_holdout(features_df)
    cv_scores = cross_validate_by_group(features_df)

    print(f"\nGrouped holdout accuracy: {holdout_accuracy:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_df)
    if cv_scores:
        print(f"\nGrouped CV accuracy: mean={np.mean(cv_scores):.4f} std={np.std(cv_scores):.4f}")
        print("Fold scores:", ", ".join(f"{score:.4f}" for score in cv_scores))
    else:
        print("\nGrouped CV accuracy: skipped (not enough unique trial groups).")

    model, feature_columns = fit_full_model(features_df)
    save_model_bundle(model, feature_columns, model_out, window_ms=args.window_ms, stride_ms=args.stride_ms)
    print(f"\nSaved model bundle: {model_out}")
    print(f"Saved metadata: {model_out.with_suffix('.json')}")
    print("Numeric class counts:", class_counts)


if __name__ == "__main__":
    main()
