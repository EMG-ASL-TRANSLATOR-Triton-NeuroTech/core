"""
Train or load a base model from legacy trials, then evaluate on new MindRove trials.

Expected workflow:
1) Legacy files (ECH/EHL/EOH/EPE/ESP) define the base model.
2) New calibration files from mindrove_interface are evaluated against that model.
3) Report window-level and trial-level accuracy, and per-trial gesture predictions.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


MINDROVE_SAMPLING_RATE = 500
WINDOW_MS = 250.0
STRIDE_MS = 50.0
RAW_CHANNELS = [f"Channel{i}" for i in range(1, 9)]
FALLBACK_CHANNELS = [f"CH{i}" for i in range(1, 9)]
INDEXED_CHANNELS = [f"ch_{i}" for i in range(8)]
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
LEGACY_TRAIN_FILE_PATTERN = re.compile(r"^(ECH|EHL|EOH|EPE|ESP)\d{2}-\d{4}\.csv$", re.IGNORECASE)
NEW_TRIAL_FILE_PATTERNS = (
    re.compile(r"^\d{8}_\d{6}_[a-z-]+_trial\d+\.csv$", re.IGNORECASE),
    re.compile(r"^[a-z-]+_\d{8}_\d{6}(?:_\d{3})?\.csv$", re.IGNORECASE),
)


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
    # Support mixed datasets where channel naming differs by export path.
    column_lookup = {column.lower(): column for column in df.columns}

    for candidate_set in (RAW_CHANNELS, FALLBACK_CHANNELS, INDEXED_CHANNELS):
        if all(column.lower() in column_lookup for column in candidate_set):
            return [column_lookup[column.lower()] for column in candidate_set]

    raise ValueError(f"Could not find expected EMG channels in columns: {list(df.columns)}")


def is_legacy_trial_file(path: Path) -> bool:
    return bool(LEGACY_TRAIN_FILE_PATTERN.match(path.name))


def is_new_mindrove_trial_file(path: Path, input_root: Path) -> bool:
    if is_legacy_trial_file(path):
        return False

    relative_parts = path.relative_to(input_root).parts
    if len(relative_parts) < 2:
        return False

    parent_slug = relative_parts[-2].lower()
    if parent_slug not in LABEL_TO_ID:
        return False

    return any(pattern.match(path.name) for pattern in NEW_TRIAL_FILE_PATTERNS)


def collect_trial_files(input_root: Path, dataset: Literal["legacy", "new"]) -> list[TrialRecord]:
    trial_files = []
    for path in sorted(input_root.rglob("*.csv")):
        if dataset == "legacy" and not is_legacy_trial_file(path):
            continue
        if dataset == "new" and not is_new_mindrove_trial_file(path, input_root):
            continue

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
        raise FileNotFoundError(f"No {dataset} trial CSV files found under {input_root}")
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
    eps = 1e-10

    for idx in range(window.shape[1]):
        channel = window[:, idx]
        abs_channel = np.abs(channel)
        diff_channel = np.diff(channel)
        channel_prefix = f"{prefix}{idx + 1}"

        # Existing EMG staples
        features[f"{channel_prefix}_RMS"] = float(np.sqrt(np.mean(channel ** 2)))
        features[f"{channel_prefix}_VAR"] = float(np.var(channel))
        features[f"{channel_prefix}_MAV"] = float(np.mean(abs_channel))
        features[f"{channel_prefix}_WL"] = float(np.sum(np.abs(diff_channel)))
        features[f"{channel_prefix}_IEMG"] = float(np.sum(abs_channel))
        features[f"{channel_prefix}_ZC"] = float(np.sum((channel[:-1] * channel[1:]) < 0))
        features[f"{channel_prefix}_SSC"] = float(
            np.sum(((channel[1:-1] - channel[:-2]) * (channel[1:-1] - channel[2:])) > 0)
        )

        # Robust amplitude descriptors (better separation for subtle finger differences)
        channel_median = float(np.median(channel))
        features[f"{channel_prefix}_MEDABS"] = float(np.median(abs_channel))
        features[f"{channel_prefix}_MAD"] = float(np.median(np.abs(channel - channel_median)))
        features[f"{channel_prefix}_IQR"] = float(np.percentile(channel, 75) - np.percentile(channel, 25))
        features[f"{channel_prefix}_LOGDET"] = float(np.exp(np.mean(np.log(abs_channel + eps))))
        features[f"{channel_prefix}_DASDV"] = float(np.sqrt(np.mean(diff_channel ** 2)))

        # Frequency descriptors (captures activation pattern differences across gestures)
        power = np.abs(np.fft.rfft(channel)) ** 2
        freqs = np.fft.rfftfreq(len(channel), d=1.0 / MINDROVE_SAMPLING_RATE)
        power_sum = float(np.sum(power))
        if power_sum <= eps:
            features[f"{channel_prefix}_MNF"] = 0.0
            features[f"{channel_prefix}_MDF"] = 0.0
            features[f"{channel_prefix}_PKF"] = 0.0
        else:
            norm_power = power / power_sum
            cumulative = np.cumsum(norm_power)
            mdf_idx = int(np.searchsorted(cumulative, 0.5))
            mdf_idx = min(max(mdf_idx, 0), len(freqs) - 1)
            features[f"{channel_prefix}_MNF"] = float(np.sum(freqs * norm_power))
            features[f"{channel_prefix}_MDF"] = float(freqs[mdf_idx])
            features[f"{channel_prefix}_PKF"] = float(freqs[int(np.argmax(power))])

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


def _fit_model_with_params(
    features_df: pd.DataFrame,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    random_state: int = 42,
) -> tuple[RandomForestClassifier, list[str]]:
    feature_cols = [col for col in features_df.columns if col.startswith("CH")]
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(features_df[feature_cols], features_df["Target"])
    return model, feature_cols


def _split_legacy_train_validation(
    records: list[TrialRecord],
    train_ratio: float = 0.8,
    random_state: int = 42,
) -> tuple[list[TrialRecord], list[TrialRecord]]:
    rng = np.random.default_rng(random_state)
    train_records: list[TrialRecord] = []
    val_records: list[TrialRecord] = []

    by_label: dict[str, list[TrialRecord]] = {}
    for record in records:
        by_label.setdefault(record.label_name, []).append(record)

    for label_records in by_label.values():
        items = list(label_records)
        rng.shuffle(items)
        if len(items) == 1:
            train_records.extend(items)
            continue

        split_idx = int(round(len(items) * train_ratio))
        split_idx = min(max(split_idx, 1), len(items) - 1)
        train_records.extend(items[:split_idx])
        val_records.extend(items[split_idx:])

    return train_records, val_records


def _build_records_from_calibration_paths(paths: Sequence[Path], input_root: Path) -> list[TrialRecord]:
    records: list[TrialRecord] = []
    for path in paths:
        resolved = Path(path).resolve()
        if not resolved.exists():
            continue
        try:
            relative = resolved.relative_to(input_root)
        except ValueError:
            continue

        label_name = infer_label_from_path(resolved, input_root)
        if label_name is None:
            continue

        relative_parts = relative.parts
        subject_id = relative_parts[0] if len(relative_parts) > 2 else "unknown"
        session_id = resolved.stem.split("_trial")[0]
        records.append(
            TrialRecord(
                path=resolved,
                label_name=label_name,
                label_id=LABEL_TO_ID[label_name],
                group_id=str(relative),
                subject_id=subject_id,
                session_id=session_id,
            )
        )
    return records


def train_model_with_legacy_and_calibration(
    input_root: Path,
    calibration_paths: Sequence[Path],
    model_out: Path,
    window_ms: float = WINDOW_MS,
    stride_ms: float = STRIDE_MS,
    random_state: int = 42,
) -> tuple[RandomForestClassifier, list[str], dict[str, object]]:
    legacy_records = collect_trial_files(input_root, dataset="legacy")
    legacy_train_records, legacy_val_records = _split_legacy_train_validation(
        legacy_records,
        train_ratio=0.8,
        random_state=random_state,
    )
    calibration_records = _build_records_from_calibration_paths(calibration_paths, input_root)

    legacy_train_df = build_feature_table(
        legacy_train_records,
        window_ms=window_ms,
        stride_ms=stride_ms,
    )
    legacy_val_df = build_feature_table(
        legacy_val_records,
        window_ms=window_ms,
        stride_ms=stride_ms,
    )
    calibration_df = build_feature_table(
        calibration_records,
        window_ms=window_ms,
        stride_ms=stride_ms,
    )

    validation_df = pd.concat([legacy_val_df, calibration_df], ignore_index=True)

    candidates = [
        {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 2},
        {"n_estimators": 300, "max_depth": 20, "min_samples_leaf": 2},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
    ]

    best_model: RandomForestClassifier | None = None
    best_feature_cols: list[str] = []
    best_params: dict[str, object] = {}
    best_window_accuracy = -1.0

    for params in candidates:
        candidate_model, feature_cols = _fit_model_with_params(
            legacy_train_df,
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=random_state,
        )
        window_acc, _, _, _, _ = evaluate_model_on_new_data(candidate_model, feature_cols, validation_df)
        if window_acc > best_window_accuracy:
            best_window_accuracy = window_acc
            best_model = candidate_model
            best_feature_cols = feature_cols
            best_params = params

    assert best_model is not None

    adapted_train_df = pd.concat([legacy_train_df, calibration_df], ignore_index=True)
    adapted_model, adapted_feature_cols = _fit_model_with_params(
        adapted_train_df,
        n_estimators=int(best_params["n_estimators"]),
        max_depth=best_params["max_depth"],
        min_samples_leaf=int(best_params["min_samples_leaf"]),
        random_state=random_state,
    )
    save_model_bundle(
        adapted_model,
        adapted_feature_cols,
        model_out,
        window_ms=window_ms,
        stride_ms=stride_ms,
    )

    post_window_acc, post_confusion, post_trial_acc, post_trial_summary, _ = evaluate_model_on_new_data(
        adapted_model,
        adapted_feature_cols,
        validation_df,
    )

    report = {
        "legacy_total_files": len(legacy_records),
        "legacy_train_files": len(legacy_train_records),
        "legacy_validation_files": len(legacy_val_records),
        "calibration_files": len(calibration_records),
        "legacy_train_windows": int(len(legacy_train_df)),
        "legacy_validation_windows": int(len(legacy_val_df)),
        "calibration_windows": int(len(calibration_df)),
        "validation_windows_total": int(len(validation_df)),
        "selected_params": best_params,
        "pre_adaptation_window_accuracy": float(best_window_accuracy),
        "post_adaptation_window_accuracy": float(post_window_acc),
        "post_adaptation_trial_accuracy": float(post_trial_acc),
        "post_adaptation_confusion": post_confusion,
        "post_adaptation_trial_summary": post_trial_summary,
    }
    return adapted_model, adapted_feature_cols, report


def predict_trial_dataframe(
    model: RandomForestClassifier,
    feature_columns: list[str],
    trial_df: pd.DataFrame,
    fs: float = MINDROVE_SAMPLING_RATE,
    window_ms: float = WINDOW_MS,
    stride_ms: float = STRIDE_MS,
) -> tuple[int, str, float]:
    channel_cols = get_channel_columns(trial_df)
    filtered = preprocess_signals(trial_df[channel_cols].to_numpy(dtype=float), fs=fs)

    window_samples = max(1, int(window_ms / 1000 * fs))
    stride_samples = max(1, int(stride_ms / 1000 * fs))
    feature_rows: list[dict[str, float]] = []
    for start in range(0, len(filtered) - window_samples + 1, stride_samples):
        window = filtered[start : start + window_samples]
        feature_rows.append(extract_features_from_window(window, prefix="CH"))

    if not feature_rows:
        raise RuntimeError("Trial is too short to extract prediction windows.")

    features = pd.DataFrame(feature_rows)
    for column in feature_columns:
        if column not in features.columns:
            features[column] = 0.0
    features = features[feature_columns]

    predictions = model.predict(features)
    vote_counts = pd.Series(predictions).value_counts(normalize=True)
    pred_id = int(vote_counts.index[0])
    confidence = float(vote_counts.iloc[0])
    return pred_id, ID_TO_LABEL.get(pred_id, str(pred_id)), confidence


def evaluate_model_on_new_data(
    model: RandomForestClassifier,
    feature_columns: list[str],
    features_df: pd.DataFrame,
) -> tuple[float, pd.DataFrame, float, pd.DataFrame, dict[str, int]]:
    X_new = features_df[feature_columns]
    y_true = features_df["Target"]
    y_pred = model.predict(X_new)

    window_accuracy = float(accuracy_score(y_true, y_pred))
    labels = sorted(y_true.unique())
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    window_confusion_df = pd.DataFrame(matrix, index=labels, columns=labels)

    grouped = pd.DataFrame(
        {
            "GroupId": features_df["GroupId"],
            "TrueTarget": y_true,
            "PredTarget": y_pred,
        }
    )
    trial_summary = (
        grouped.groupby("GroupId")
        .agg(
            TrueTarget=("TrueTarget", "first"),
            PredTarget=("PredTarget", lambda s: int(pd.Series(s).value_counts().idxmax())),
            WindowCount=("PredTarget", "size"),
        )
        .reset_index()
    )
    trial_accuracy = float((trial_summary["TrueTarget"] == trial_summary["PredTarget"]).mean())
    class_counts = y_true.value_counts().sort_index().to_dict()

    return window_accuracy, window_confusion_df, trial_accuracy, trial_summary, class_counts


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

    parser = argparse.ArgumentParser(
        description="Train/load a legacy base model and evaluate new MindRove calibration files."
    )
    parser.add_argument(
        "--input-dir",
        default=str(project_root / "CSV-Files"),
        help="Root folder containing both legacy and new trial files.",
    )
    parser.add_argument(
        "--model-out",
        default=str(base_dir / "models" / "mindrove_rf.joblib"),
        help="Output path for the trained model bundle (used when --base-model-in is not provided).",
    )
    parser.add_argument(
        "--base-model-in",
        default="",
        help="Optional path to an existing model bundle. If set, legacy retraining is skipped.",
    )
    parser.add_argument(
        "--trial-predictions-out",
        default=str(base_dir / "models" / "mindrove_new_trial_predictions.csv"),
        help="CSV output path for per-trial prediction summary on new calibration data.",
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
    base_model_in = Path(args.base_model_in) if args.base_model_in else None
    trial_predictions_out = Path(args.trial_predictions_out)

    if base_model_in:
        loaded = joblib.load(base_model_in)
        model = loaded["model"]
        feature_columns = loaded["metadata"]["feature_columns"]
        print("=== Base Model ===")
        print(f"Loaded model bundle: {base_model_in}")
    else:
        legacy_records = collect_trial_files(input_root, dataset="legacy")
        legacy_features_df = build_feature_table(legacy_records, window_ms=args.window_ms, stride_ms=args.stride_ms)
        model, feature_columns = fit_full_model(legacy_features_df)
        save_model_bundle(model, feature_columns, model_out, window_ms=args.window_ms, stride_ms=args.stride_ms)
        print("=== Base Model ===")
        print("Trained from legacy files")
        print(f"Legacy trial files used: {len(legacy_records)}")
        print(f"Legacy windowed samples: {len(legacy_features_df)}")
        print(f"Saved model bundle: {model_out}")
        print(f"Saved metadata: {model_out.with_suffix('.json')}")

    new_records = collect_trial_files(input_root, dataset="new")
    new_features_df = build_feature_table(new_records, window_ms=args.window_ms, stride_ms=args.stride_ms)

    print("\n=== New MindRove Calibration Evaluation ===")
    print(f"New trial files used: {len(new_records)}")
    print(f"New windowed samples: {len(new_features_df)}")
    print("Class counts:")
    print(new_features_df["LabelName"].value_counts().sort_index())

    window_accuracy, confusion_df, trial_accuracy, trial_summary, class_counts = evaluate_model_on_new_data(
        model,
        feature_columns,
        new_features_df,
    )

    trial_predictions_out.parent.mkdir(parents=True, exist_ok=True)
    trial_summary.to_csv(trial_predictions_out, index=False)

    print(f"\nWindow-level accuracy on new data: {window_accuracy:.4f}")
    print("Window-level confusion matrix (rows=true, cols=pred):")
    print(confusion_df)
    print(f"\nTrial-level accuracy on new data (majority vote): {trial_accuracy:.4f}")

    printable_trials = trial_summary.copy()
    printable_trials["TrueLabel"] = printable_trials["TrueTarget"].map(ID_TO_LABEL)
    printable_trials["PredLabel"] = printable_trials["PredTarget"].map(ID_TO_LABEL)
    print("Per-trial predictions:")
    print(printable_trials[["GroupId", "WindowCount", "TrueLabel", "PredLabel"]].to_string(index=False))

    print(f"\nSaved per-trial prediction summary: {trial_predictions_out}")
    print("Numeric class counts (new data):", class_counts)


if __name__ == "__main__":
    main()
