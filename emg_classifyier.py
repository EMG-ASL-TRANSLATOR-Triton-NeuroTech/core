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
from scipy.stats import mode as scipy_mode
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


# --- Configuration ---
CSV_PATH = "dataset_emg_mindrove.csv"  # Update this to your CSV file path
DATASET_ROOT = "DATASET EMG MINDROVE"  # Folder with subject subfolders (AYU, DANIEL, etc.)
MINDROVE_SAMPLING_RATE = 500  # Hz


def load_master_dataframe(dataset_root: str) -> pd.DataFrame:
    """
    Find all .csv files recursively, load each, add Subject from folder name, concatenate.
    """
    pattern = str(Path(dataset_root) / "**" / "*.csv")
    csv_files = glob.glob(pattern)
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
    filtered_data: np.ndarray,
    labels: np.ndarray,
    fs: float = 200.0,
    window_ms: float = 200.0,
) -> pd.DataFrame:
    """
    Slice filtered data into windows and extract RMS, VAR, MAV per channel.
    
    Args:
        filtered_data: 2D array (samples, channels) or 1D (samples,).
        labels: 1D array of labels (e.g. 'Fist', 'Rest') for each sample.
        fs: Sampling rate in Hz.
        window_ms: Window length in milliseconds.
    
    Returns:
        DataFrame: each row is a window with RMS, VAR, MAV per channel + Label.
    """
    if filtered_data.ndim == 1:
        filtered_data = filtered_data[:, np.newaxis]
    n_samples, n_channels = filtered_data.shape
    window_samples = int(window_ms / 1000 * fs)
    if window_samples < 1:
        window_samples = 1

    rows = []
    for start in range(0, n_samples - window_samples + 1, window_samples):
        end = start + window_samples
        window = filtered_data[start:end]
        window_labels = labels[start:end]
        labels_rounded = np.round(np.asarray(window_labels)).astype(int)
        label = int(scipy_mode(labels_rounded, keepdims=False).mode)

        feats = {}
        for c in range(n_channels):
            x = window[:, c]
            feats[f"CH{c+1}_RMS"] = np.sqrt(np.mean(x**2))
            feats[f"CH{c+1}_VAR"] = np.var(x)
            feats[f"CH{c+1}_MAV"] = np.mean(np.abs(x))
        feats["Target"] = label
        rows.append(feats)

    df = pd.DataFrame(rows)
    df["Target"] = df["Target"].astype(int)
    return df


def train_and_evaluate(
    features_df: pd.DataFrame,
    label_col: str = "Target",
    test_size: float = 0.2,
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

    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
    print("\nConfusion Matrix (rows = true, cols = predicted):")
    print(cm_df)

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
        print(f"Loaded {len(df)} samples, {len(df.columns)} columns: {list(df.columns)}")
    plot_channel_1_first_n_seconds(df, seconds=5)

    # Preprocess and compare raw vs filtered (first 1000 samples)
    raw_ch1 = df["CH1"].values
    filtered_ch1 = preprocess_signals(raw_ch1)
    plot_raw_vs_filtered(raw_ch1, filtered_ch1, n_samples=1000)

    # Full pipeline: preprocess all channels -> extract features -> train
    channel_cols = [f"CH{i}" for i in range(1, 9)]
    filtered = preprocess_signals(df[channel_cols].values)
    features_df = extract_features(
        filtered, df["Target"].values, fs=MINDROVE_SAMPLING_RATE, window_ms=200
    )
    train_and_evaluate(features_df)
