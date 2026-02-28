# EMG Gesture Recognition CNN

This project contains a pipeline for processing raw Electromyography (EMG) data and training a Convolutional Neural Network (CNN) to recognize hand gestures.

## Project Structure

- **`extract_data.py`**: Preprocesses raw EMG CSV files. It extracts specific channels, downsamples the data using a sliding window, and formats it into a fixed size suitable for the CNN.
- **`training.py`**: Defines the dataset loader, CNN architecture, and training loop. It loads processed CSV files, trains the model, and saves the weights.


## Usage Instructions

### Step 1: Data Preprocessing

Run `extract_data.py` to convert raw EMG recordings into the format required for training.

```bash
python extract_data.py
```

**Configuration:**
- **Input:** Update `file_path` in the `if __name__ == "__main__":` block to point to your raw CSV file.
- **Output:** By default, processed files are saved to `emg_data/`. You should move these files to the `Processed_CSV` directory expected by the training script, or update the `DATA_DIR` in `training.py`.
- **Parameters:**
  - `window_ms`: Downsampling window (default: 30ms).
  - `target_channels`: Columns to extract from raw data (default: `[8, 9, 10, 11]`).
  - `output_rows`: Must be **169** to match the CNN input reshape logic (13x13).

### Step 2: Model Training

Organize your processed CSV files into a folder named `Processed_CSV`. Ensure the filenames follow the naming convention described below.

Run the training script:

```bash
python training.py
```

**Configuration (`training.py`):**
- `DATA_DIR`: Directory containing processed CSV files (default: `"Processed_CSV"`).
- `NUM_CLASSES`: Number of gesture classes to predict (default: 5).
- `NUM_EPOCHS`: Training iterations (default: 100).
- `BATCH_SIZE`: Batch size for DataLoader (default: 8).

## File Naming Convention

The training script automatically assigns labels based on the filename. Files **must** follow this pattern:

```text
gesture{N}_sample{N}.csv
```

**Examples:**
- `gesture1_sample1.csv` → Class 0 (Gesture 1)
- `gesture2_sample1.csv` → Class 1 (Gesture 2)
- `gesture5_sample3.csv` → Class 4 (Gesture 5)

The script extracts the integer after `gesture` to determine the class label.

## Model Architecture

The CNN expects an input tensor of shape **(4, 13, 13)** (Channels × Height × Width).

1.  **Input:** 4 Channels, 13x13 spatial dimensions (derived from 169 time steps).
2.  **Feature Extractor:**
    - 3 Convolutional Blocks (Conv2d + BatchNorm + ReLU + MaxPool).
    - Channel progression: 4 → 16 → 32 → 64.
3.  **Classifier:**
    - Fully Connected layers (64*3*3 → 128 → Num Classes).
    - Includes Dropout (0.5) for regularization.

## Output Models

Upon completion, `training.py` saves two models in the parent directory:
- **`best_emg_model.pth`**: Weights from the epoch with the highest validation accuracy.
- **`emg_gesture_model_final.pth`**: Weights from the final epoch.
