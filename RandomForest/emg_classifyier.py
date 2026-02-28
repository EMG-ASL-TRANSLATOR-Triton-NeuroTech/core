"""
EMG Classifier - Load MindRove EMG data and verify with plotting.
Supports resampling for datasets with different sampling rates (e.g., Ninapro 200Hz).
"""

import glob
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "DATASET EMG MINDROVE" / "FORMATTED" / "subjek_FORMATTED.csv"
DATASET_ROOT = BASE_DIR / "DATASET EMG MINDROVE" / "FORMATTED" / "by_gesture"
MINDROVE_SAMPLING_RATE = 500  # Hz


def load_master_dataframe(dataset_root: str) -> pd.DataFrame:
    """
    Find all .csv files recursively, load each, add Subject from folder name, concatenate.
    """
    root = Path(dataset_root)
    # Load only direct CSVs from the selected dataset folder.
    csv_files = [str(p) for p in root.glob("*.csv")]
    if not csv_files:
        raise FileNotFoundError(f"No .csv files found under {dataset_root}")

    dfs = []
    for filepath in csv_files:
        df = pd.read_csv(filepath)
        subject = Path(filepath).parent.name
        df["Subject"] = subject
        dfs.append(df)

    master = pd.concat(dfs, ignore_index=True)
    print(f"Master DataFrame shape: {master.shape}")
    print(f"Unique Subjects: {sorted(master['Subject'].unique())}")
    return master


def load_emg_data(filepath: str) -> pd.DataFrame:
    """Load EMG CSV with columns CH1..CH8 and Target."""
    df = pd.read_csv(filepath)
    return df


def align_sampling_rate(
    data: np.ndarray,
    original_fs: float,
    target_fs: float = MINDROVE_SAMPLING_RATE,
) -> np.ndarray:
    """
    Resample data to target sampling rate using scipy.signal.resample.
    
    Args:
        data: 1D or 2D array (samples, [channels]). For 2D, resampling is along axis=0.
        original_fs: Original sampling rate in Hz (e.g., 200 for Ninapro).
        target_fs: Target sampling rate in Hz (default 500 for MindRove).
    
    Returns:
        Resampled array at target_fs Hz.
    """
    if original_fs == target_fs:
        return data
    
    n_samples = data.shape[0]
    duration_sec = n_samples / original_fs
    n_target = int(duration_sec * target_fs)
    
    if data.ndim == 1:
        return resample(data, n_target)
    return resample(data, n_target, axis=0)


def preprocess_signals(
    data: np.ndarray,
    fs: float = MINDROVE_SAMPLING_RATE,
    bandpass_low: float = 20.0,
    bandpass_high: float = 450.0,
    notch_freq: float = 60.0,
) -> np.ndarray:
    """
    Preprocess EMG signals: bandpass filter + 60Hz notch for power line removal.
    
    Args:
        data: 1D or 2D array (samples, [channels]). Filtering is along axis=0.
        fs: Sampling rate in Hz.
        bandpass_low: Bandpass low cutoff (Hz).
        bandpass_high: Bandpass high cutoff (Hz). Capped at Nyquist-1 for validity.
        notch_freq: Notch filter center frequency (Hz), typically 60 for power line.
    
    Returns:
        Filtered array, same shape as input.
    """
    nyq = fs / 2
    high_cutoff = min(bandpass_high, nyq - 1)

    # Bandpass 20–450 Hz (high capped at Nyquist)
    b_bp, a_bp = butter(4, [bandpass_low / nyq, high_cutoff / nyq], btype="band")
    filtered = filtfilt(b_bp, a_bp, data, axis=0)

    # Notch at 60 Hz to remove power line interference
    q = 30
    b_notch, a_notch = iirnotch(notch_freq, q, fs)
    filtered = filtfilt(b_notch, a_notch, filtered, axis=0)

    return filtered


def plot_raw_vs_filtered(
    raw: np.ndarray,
    filtered: np.ndarray,
    n_samples: int = 1000,
    fs: float = MINDROVE_SAMPLING_RATE,
    channel_name: str = "CH1",
) -> None:
    """Plot first n_samples of raw vs preprocessed signal."""
    raw = np.asarray(raw).flatten()
    filtered = np.asarray(filtered).flatten()
    n = min(n_samples, len(raw), len(filtered))
    t = np.arange(n) / fs

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
    axes[0].plot(t, raw[:n])
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"Raw {channel_name}")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, filtered[:n])
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title(f"Filtered {channel_name} (bandpass 20–450Hz + 60Hz notch)")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def extract_features(
    df: pd.DataFrame,
    fs: float = MINDROVE_SAMPLING_RATE,
    window_ms: float = 250.0,
    transition_threshold: float = 0.1,
) -> pd.DataFrame:
    """
    Pure-State Filter: group by Subject, slide 250ms window; drop windows where
    max(target)-min(target) > threshold (transition). Otherwise round(mean(target))
    and extract RMS, VAR, MAV per channel.
    """
    channel_cols = [f"CH{i}" for i in range(1, 9) if f"CH{i}" in df.columns]
    if not channel_cols:
        channel_cols = [c for c in df.columns if c == "CH1" or (isinstance(c, str) and c.startswith("CH") and len(c) <= 4)]
    if not channel_cols:
        channel_cols = [f"CH{i}" for i in range(1, 9)]
    target_col = "Target"
    subject_col = "Subject"
    window_samples = max(1, int(window_ms / 1000 * fs))
    n_channels = len(channel_cols)
    all_rows = []

    for _subject, group in df.groupby(subject_col):
        data = group[channel_cols].values
        targets = group[target_col].values
        n_samples = len(data)
        for start in range(0, n_samples - window_samples + 1, window_samples):
            end = start + window_samples
            window = data[start:end]
            window_targets = targets[start:end]
            diff = float(np.max(window_targets) - np.min(window_targets))
            if diff > transition_threshold:
                continue
            label = int(round(np.mean(window_targets)))
            feats = {"Subject": _subject, "Target": label}
            for c in range(n_channels):
                x = window[:, c]
                feats[f"CH{c+1}_RMS"] = np.sqrt(np.mean(x**2))
                feats[f"CH{c+1}_VAR"] = np.var(x)
                feats[f"CH{c+1}_MAV"] = np.mean(np.abs(x))
            all_rows.append(feats)

    out = pd.DataFrame(all_rows)
    out["Target"] = out["Target"].astype(int)
    return out


def normalize_and_encode(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Subject-wise normalization: StandardScaler on RMS/VAR/MAV per Subject.
    Then one-hot encode Subject (context injection) via pd.get_dummies.
    """
    feat_cols = [c for c in features_df.columns if c.endswith("_RMS") or c.endswith("_VAR") or c.endswith("_MAV")]
    if not feat_cols:
        feat_cols = [c for c in features_df.columns if c not in ("Subject", "Target")]
    subject_col = "Subject"
    target_col = "Target"
    parts = []
    for _subject, group in features_df.groupby(subject_col):
        scaler = StandardScaler()
        X = group[feat_cols]
        group_scaled = group.copy()
        group_scaled[feat_cols] = scaler.fit_transform(X)
        parts.append(group_scaled)
    scaled_df = pd.concat(parts, ignore_index=True)
    out = pd.get_dummies(scaled_df, columns=[subject_col])
    return out


def train_and_evaluate(
    features_df: pd.DataFrame,
    label_col: str = "Target",
    test_size: float = 0.8,
    random_state: int = 42,
    min_samples_per_class: int = 10,
) -> RandomForestClassifier:
    """
    Split features 80/20, train RandomForestClassifier, evaluate on test set.
    Prints accuracy score and confusion matrix.
    Filters out classes with fewer than min_samples_per_class before splitting.
    """
    X = features_df.drop(columns=[label_col])
    y = features_df[label_col]

    print(f"{label_col} value counts (before filtering):")
    print(y.value_counts())

    value_counts = y.value_counts()
    valid_classes = value_counts[value_counts >= min_samples_per_class].index
    mask = y.isin(valid_classes)
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    if len(valid_classes) < len(value_counts):
        print(f"\nFiltered out classes with < {min_samples_per_class} samples. Remaining: {list(valid_classes)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=random_state),
        param_grid,
        cv=cv,
        scoring="accuracy",
    )
    grid_search.fit(X_train, y_train)

    print("Best hyperparameters:", grid_search.best_params_)

    model = grid_search.best_estimator_
    y_pred = model.predict(X_test)
    print("Final global accuracy (test set):", accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
    print("\nConfusion Matrix (rows = true, cols = predicted):")
    print(cm_df)

    # Feature importance: top 10 horizontal bar chart
    importance_df = pd.DataFrame(
        {"feature": X_train.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    top10 = importance_df.head(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(top10)), top10["importance"].values, align="center")
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Top 10 Most Important Features")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.show()

    return model


def plot_channel_1_first_n_seconds(
    df: pd.DataFrame,
    channel_col: str = "CH1",
    sampling_rate: float = MINDROVE_SAMPLING_RATE,
    seconds: float = 5.0,
) -> None:
    """Plot the first N seconds of a channel to verify data loaded correctly."""
    signal = df[channel_col].values
    n_samples = int(seconds * sampling_rate)
    signal = signal[:n_samples]
    time_axis = np.arange(len(signal)) / sampling_rate

    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, signal)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"First {seconds} seconds of {channel_col} ({sampling_rate} Hz)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load from multi-subject dataset or single CSV
    if Path(DATASET_ROOT).exists():
        df = load_master_dataframe(DATASET_ROOT)
    else:
        df = load_emg_data(CSV_PATH)
        df["Subject"] = "single"
        print(f"Loaded {len(df)} samples, {len(df.columns)} columns: {list(df.columns)}")

    channel_cols = [f"CH{i}" for i in range(1, 9) if f"CH{i}" in df.columns]
    if not channel_cols:
        channel_cols = [c for c in df.columns if c not in ("Subject", "Target") and not str(c).startswith("Unnamed")]

    # Preprocess per subject so Subject stays aligned
    parts = []
    for _subject, group in df.groupby("Subject"):
        ch_data = group[channel_cols].values
        filtered_ch = preprocess_signals(ch_data, fs=MINDROVE_SAMPLING_RATE)
        part = group.copy()
        part[channel_cols] = filtered_ch
        parts.append(part)
    filtered_df = pd.concat(parts, ignore_index=True)

    # 1. Pure-State extraction (250ms windows, drop transitions)
    features_df = extract_features(filtered_df, fs=MINDROVE_SAMPLING_RATE, window_ms=250)
    print(f"Features after Pure-State filter: {features_df.shape}")

    # 2. Subject-wise normalization + 3. Context injection (one-hot Subject)
    features_df = normalize_and_encode(features_df)
    print(f"Features after normalize + encode: {features_df.shape}")

    # 4. Train global model (GridSearchCV), print accuracy and confusion matrix
    model = train_and_evaluate(features_df)
    print("Global model training complete.")
