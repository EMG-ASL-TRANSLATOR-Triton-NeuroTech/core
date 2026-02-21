import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os


def plot_emg_data(filename='downsampled_emg.csv'):
    """
    Plot four channels of downsampled EMG data with shared time axis.
    """
    # Validate file existence
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        print("Usage: python emg_plotter.py [filename.csv]")
        sys.exit(1)

    try:
        # Read EMG data
        df = pd.read_csv(filename)

        # Verify required columns exist
        required_cols = ['FilteredChannel1', 'FilteredChannel2', 'FilteredChannel3', 'FilteredChannel4']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {', '.join(required_cols)}")

        # Calculate time vector based on downsampling parameters
        # Original sampling rate: 1000 Hz (1 sample/ms)
        # Downsampling window: 30 ms → 1 output sample per 30 ms window
        # Effective sampling period after downsampling: 30 ms = 0.03 seconds
        window_duration = 0.03  # seconds
        n_samples = len(df)
        time = np.arange(n_samples) * window_duration  # Time in seconds

        # Create figure with 4 vertically stacked subplots sharing x-axis
        fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True, sharey=True)

        # Plot each channel
        channels = [
            ('FilteredChannel1', 'Channel 1'),
            ('FilteredChannel2', 'Channel 2'),
            ('FilteredChannel3', 'Channel 3'),
            ('FilteredChannel4', 'Channel 4')
        ]

        for idx, (col, label) in enumerate(channels):
            axes[idx].plot(time, df[col], color=f'C{idx}', linewidth=0.8)
            axes[idx].set_ylabel(label, fontsize=10)
            axes[idx].grid(True, alpha=0.3, linestyle='--')
            axes[idx].set_title(f'EMG Signal - {label}', fontsize=9, loc='left', pad=2)

            # Add amplitude annotation in corner
            max_amp = df[col].abs().max()
            axes[idx].text(0.98, 0.93, f'Max: {max_amp:.2f} µV',
                           transform=axes[idx].transAxes,
                           fontsize=8,
                           verticalalignment='top',
                           horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        # Format shared x-axis
        axes[-1].set_xlabel('Time (seconds)', fontsize=11)
        axes[-1].set_xlim(0, time[-1] if n_samples > 0 else 1)

        # Overall figure title and layout
        fig.suptitle('Multi-Channel EMG Data (30ms Downsampled from 1kHz Source)',
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0.01, 1, 0.97])  # Make room for suptitle

        # Add sampling info as footnote
        fig.text(0.5, 0.005,
                 f'Sampling: 1 kHz source → {1 / window_duration:.1f} Hz effective rate (30 ms windows) | Samples: {n_samples}',
                 ha='center', fontsize=8, style='italic', color='gray')

        plt.show()

        print(f"Plotted {n_samples} samples ({time[-1]:.2f} seconds of data)")

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Allow filename override via command line argument
    if len(sys.argv) > 1:
        plot_emg_data(sys.argv[1])
    else:
        plot_emg_data()