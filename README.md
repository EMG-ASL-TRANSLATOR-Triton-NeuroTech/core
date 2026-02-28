# TritonNeuroTech - EMG ASL Translator

A comprehensive project for EMG (Electromyography) signal analysis and gesture recognition using dimensionality reduction techniques.

## Project Overview

This project analyzes EMG data from various hand gestures and finger movements using PCA (Principal Component Analysis) and UMAP (Uniform Manifold Approximation and Projection) for visualization and clustering analysis.

## Features

- **EMG Data Analysis**: Process and visualize EMG signals from multiple subjects
- **Gesture Recognition**: Analyze individual hand gestures (Open Hand, Closed Hand)
- **Finger Movement Detection**: Classify individual finger movements (Index, Ring, Pinky)
- **Multi-Subject Support**: Compare EMG patterns across different subjects
- **Flexible Configuration System**: Easy switching between datasets using presets
- **Batch Testing**: Automated testing of multiple configurations
- **Visualization**: Side-by-side PCA and UMAP visualizations

## Quick Start

### Prerequisites

- Python 3.13+
- Virtual environment (`.venv`)

### Installation

1. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Run UMAP analysis with default configuration:**
   ```bash
   python run_umap.py default
   ```

## Available Datasets

### 1. **Default Configuration** - Ricardo's Open/Close Hand
- Open Hand vs Closed Hand gestures
- **Run:** `python run_umap.py default`

### 2. **Individual Finger Tests** (NEW)
- Index Finger, Ring Finger, Pinky Finger
- **Run:** `python run_umap.py fingers`

### 3. **Hand Position Variants** (NEW)
- Closed Hand + Index, Ring, and Pinky Fingers
- **Run:** `python run_umap.py hand`

### 4. **Complete Gesture Set** (NEW)
- Open Hand, Closed Hand + All Finger Tests
- **Run:** `python run_umap.py all_gestures`

### 5. **Closed Hand Subject Comparison** (NEW)
- Ricardo Closed Hand vs Mohak Closed Hand
- **Run:** `python run_umap.py comparison`

### 6. **Multi-Subject Comparison**
- Ricardo and Sirisha data combined
- **Run:** `python run_umap.py all`

### 7. **EMG_ASL Folder Data**
- Test data from EMG_ASL directory
- **Run:** `python run_umap.py emc`

### 8. **Random Forest Dataset**
- Multiple subjects: AYU, DANIEL, LINTANG
- **Run:** `python run_umap.py random`

## Dataset Files

### File Specifications

- Duration: 5 seconds
- Samples: 5000 samples (1kHz sample rate)
- Content: Filtered channels directly from MindRove hardware
- Structure: Columns at the top only (no metadata or extra information)
- Delimiter: Tab-separated (\t), not comma-separated

#### Naming Convention
- Files must follow the format: gesture_sampleN.csv
- Example: `gesture1_sample1.csv`

### Finger Test Data (CSV-Files/)
- `index_finger.csv` (9.0 MB)
- `ring_finger.csv` (23 MB)
- `pinky_finger.csv` (11 MB)
- `closed_hand.csv` (18 MB)

### Other Test Data (CSV-Files/)
- `Test-Ricardo_Open-Hand.csv`
- `Test-Ricardo_Closed-Hand.csv`
- `Test-Ricardo.csv`
- `Test-Sirisha.csv`

## Usage Guide

### Python Runner (Recommended - Cross-Platform)

```bash
# List all available configurations
python run_umap.py list

# Show quick preset commands
python run_umap.py presets

# Run with specific preset
python run_umap.py default
python run_umap.py fingers
python run_umap.py hand
python run_umap.py all_gestures
python run_umap.py comparison

# Run with custom files and labels
python run_umap.py custom --files file1.csv file2.csv --labels "Label 1" "Label 2"
```

### Shell Script Runner (macOS/Linux)

```bash
# Make executable
chmod +x run_umap.sh

# List configurations
./run_umap.sh list

# Run with preset
./run_umap.sh default
./run_umap.sh fingers
./run_umap.sh hand
./run_umap.sh all_gestures
./run_umap.sh comparison
```

### Direct Script Usage

```bash
# Default configuration
python umap_test.py

# With specific files and labels
python umap_test.py --files file1.csv file2.csv --labels "Label 1" "Label 2"
```

## Configuration Files

### gesture_configs.json
Defines preset configurations for different gesture combinations. Each configuration includes:
- File paths
- Gesture labels
- Description

#### Adding Custom Configurations

Edit `gesture_configs.json`:
```json
{
  "my_config": {
    "name": "My Configuration",
    "description": "Description of gestures",
    "gestures": [
      {"file": "path/to/file.csv", "label": "Gesture Name"}
    ]
  }
}
```

## Main Scripts

### umap_test.py
Main analysis script that:
- Loads EMG data from CSV files
- Applies StandardScaler preprocessing
- Performs PCA dimensionality reduction
- Performs UMAP dimensionality reduction
- Visualizes results in side-by-side plots

**Features:**
- Command-line argument support for file paths and labels
- Configurable PCA/UMAP parameters
- Color-coded gesture clustering
- Grid visualization

### run_umap.py
Python runner for easy configuration management
- Lists all available configurations
- Runs presets with single command
- Supports custom file/label combinations
- Cross-platform compatible

### run_umap.sh
Shell script runner for macOS/Linux
- Quick preset execution
- Custom file specification
- Configuration listing

## Data Format

All CSV files must be:
- **Tab-separated** (delimiter: `\t`)
- **Numeric columns** for EMG channels (Channel1-Channel8, etc.)
- **Optional sensor data**: Gyro, Accelerometer, PPG, HR, Battery, etc.
- **Header row** with column names

## Visualization Output

The script generates two subplots:

1. **PCA Visualization** (Left)
   - 2D principal component projection
   - Shows global structure of data
   - First two principal components

2. **UMAP Visualization** (Right)
   - 2D UMAP projection
   - Reveals local cluster structure
   - Non-linear dimensionality reduction

**Color coding:** Each gesture type gets a unique color for easy identification

## Batch Testing Examples

### Test All Finger Configurations

```bash
# Using Python runner (recommended)
python run_umap.py fingers
python run_umap.py hand
python run_umap.py all_gestures
python run_umap.py comparison

# Or create a test script
for config in fingers hand all_gestures comparison; do
    echo "Testing $config..."
    python run_umap.py $config
done
```

### Custom Multi-File Analysis

```bash
python run_umap.py custom \
  --files CSV-Files/index_finger.csv CSV-Files/ring_finger.csv \
  --labels "Index" "Ring"
```

## Project Structure

```
TritonNeuroTech/
├── umap_test.py              # Main analysis script
├── run_umap.py               # Python runner
├── run_umap.sh               # Shell script runner
├── gesture_configs.json      # Configuration presets
├── UMAP_USAGE_GUIDE.md       # Detailed usage guide
├── README.md                 # This file
│
├── CSV-Files/                # EMG test data
│   ├── Test-Ricardo_Open-Hand.csv
│   ├── Test-Ricardo_Closed-Hand.csv
│   ├── index_finger.csv      # NEW
│   ├── ring_finger.csv       # NEW
│   ├── pinky_finger.csv      # NEW
│   └── closed_hand.csv       # NEW
│
├── EMG_ASL/                  # EMG_ASL dataset
│   ├── extract_data.py
│   ├── training.py
│   ├── visualization.py
│   └── CSV/
│
├── RandomForest/             # Random Forest dataset
│   └── DATASET EMG MINDROVE/
│
└── Workshop1_DataCollection/ # Data collection code
```

## Troubleshooting

### Error: "No data files were loaded"
- Verify file paths are correct relative to project root
- Check files exist: `ls CSV-Files/filename.csv`

### Error: "Number of files must match number of labels"
- Ensure `--files` and `--labels` have same count
- Example: 2 files = 2 labels

### Error: "Could not convert string to float"
- Verify CSV is tab-separated, not comma-separated
- Check: `head -1 file.csv | od -c`

### No visualization window appears
- On headless systems, add to `umap_test.py`:
  ```python
  plt.savefig('output.png', dpi=300)
  ```

## Advanced Configuration

### Modify Analysis Parameters

Edit parameters in `umap_test.py`:

```python
# PCA components
pca = PCA(n_components=3)  # Change to 3, 5, 10, etc.

# UMAP parameters
reducer = UMAP(
    n_neighbors=15,        # Higher = global structure
    min_dist=0.1,          # Lower = tighter clusters
    random_state=42        # For reproducibility
)
```

### Save Results

Add to `umap_test.py` after visualization:

```python
# Save figure
plt.savefig('umap_analysis.png', dpi=300, bbox_inches='tight')

# Save PCA results
pca_df = pd.DataFrame(pca_results, columns=['PC1', 'PC2'])
pca_df['Gesture'] = y
pca_df.to_csv('pca_results.csv', index=False)

# Save UMAP results
umap_df = pd.DataFrame(umap_results, columns=['UMAP1', 'UMAP2'])
umap_df['Gesture'] = y
umap_df.to_csv('umap_results.csv', index=False)
```

## Dependencies

- pandas
- numpy
- scikit-learn
- umap-learn
- matplotlib

All dependencies are in the virtual environment.

## Documentation

For detailed usage instructions, see [UMAP_USAGE_GUIDE.md](UMAP_USAGE_GUIDE.md)

## Project Team

- Triton NeuroTech

## Notes

- EMG data requires proper preprocessing (filtering, normalization)
- TAB separator is critical for correct CSV parsing
- Recommend 100+ samples per gesture for meaningful clustering
- UMAP is non-deterministic; set `random_state` for reproducibility

## Latest Updates (Feb 21, 2026)

✅ Fixed file path issues in UMAP test script
✅ Added command-line argument support for file paths and labels
✅ Created gesture_configs.json for preset configurations
✅ Implemented Python and shell script runners
✅ Added individual finger test configurations (Index, Ring, Pinky)
✅ Added hand position variants configuration
✅ Added complete gesture set configuration
✅ Added Ricardo vs Mohak closed-hand comparison command
✅ Created comprehensive UMAP_USAGE_GUIDE.md
✅ Created this README.md with full documentation

## Future Enhancements

- Machine learning classification models
- Real-time EMG processing
- Additional gesture types
- Multi-electrode analysis
- Statistical significance testing
- 3D visualization options

---

For questions or contributions, please refer to the project repository.
