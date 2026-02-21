import numpy as np
import pandas as pd


"""
Extracts the four filtered channels from the output csv, downsample it with a given window size
This program pads 0 if there's insufficient data.

"""
def extract_and_downsample_emg(
        csv_file_path: str,
        window_ms: int = 30,
        sample_rate_hz: int = 1000,
        target_channels: list = [8, 9, 10, 11],
        output_rows: int = 167  # Default: 5000 samples / 30ms windows ≈ 167 windows
) -> np.ndarray:
    # Auto-detect delimiter
    with open(csv_file_path, 'r') as f:
        first_line = f.readline()
        delimiter = '\t' if '\t' in first_line[:100] else ','

    # Read CSV with proper delimiter handling
    try:
        df = pd.read_csv(csv_file_path, delimiter=delimiter, skipinitialspace=True)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")

    # Extract target columns by position
    data = df.iloc[:, target_channels].values.astype(np.float32)
    original_samples = data.shape[0]

    # Calculate window parameters
    samples_per_window = int(sample_rate_hz * window_ms / 1000)
    if samples_per_window <= 0:
        raise ValueError("Window size too small for given sample rate")

    total_required_samples = output_rows * samples_per_window

    # Adjust data length to exactly match required samples
    if original_samples < total_required_samples:
        # Pad with zeros if insufficient data
        pad_rows = total_required_samples - original_samples
        data = np.pad(
            data,
            ((0, pad_rows), (0, 0)),
            mode='constant',
            constant_values=0
        )
        padded_samples = pad_rows
        truncated_samples = 0
    else:
        # Truncate excess data to exact required length
        data = data[:total_required_samples, :]
        padded_samples = 0
        truncated_samples = original_samples - total_required_samples

    # Reshape and average across each window
    reshaped = data.reshape(output_rows, samples_per_window, -1)
    downsampled = reshaped.mean(axis=1)

    # Verification output
    print(f"Original samples: {original_samples} ({original_samples / sample_rate_hz:.2f}s)")
    print(f"Samples per window: {samples_per_window} ({window_ms}ms)")
    print(f"Padded samples: {padded_samples}")
    print(f"Truncated samples: {truncated_samples}")
    print(f"Target output rows: {output_rows}")
    print(f"Output shape: {downsampled.shape} (windows × channels)")

    return downsampled


# Example usage
if __name__ == "__main__":
    file_path = "CSV/Test-Ricardo_Open-Hand.csv"

    try:
        emg_data = extract_and_downsample_emg(
            file_path,
            window_ms=30,
            sample_rate_hz=1000,
            target_channels=[8, 9, 10, 11],
            output_rows=169 # optimized size for 5 seconds -- easier architecture for neural network input
        )

        print(f"\nFirst 5 downsampled samples (FilteredChannel1-4):\n{emg_data[:5]}")
        print(f"\nLast 5 downsampled samples:\n{emg_data[-5:]}")
        np.savetxt('downsampled_emg.csv', emg_data, delimiter=',', fmt='%.6f',
                   header='FilteredChannel1,FilteredChannel2,FilteredChannel3,FilteredChannel4', comments='')

    except Exception as e:
        print(f"Error processing file: {e}")