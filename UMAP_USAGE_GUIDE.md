# UMAP Test Configuration Guide

This guide explains how to use the updated EMG UMAP visualization script with command-line arguments and configuration files.

## Overview

The `umap_test.py` script now supports three ways to specify input data:

1. **Default Configuration** - Hard-coded paths (for quick testing)
2. **Configuration File** - JSON-based preset configurations
3. **Command-Line Arguments** - Direct file paths and labels via CLI

## Quick Start

### Using the Python Runner (Recommended for all platforms)

```bash
# List all available configurations
python run_umap.py list

# Show quick preset commands
python run_umap.py presets

# Run with a preset configuration
python run_umap.py default          # Default: Ricardo's Open/Close Hand
python run_umap.py ricardo          # Ricardo only mixed data
python run_umap.py all              # Ricardo + Sirisha
python run_umap.py emc              # EMG_ASL folder data
python run_umap.py random           # Random Forest dataset
python run_umap.py fingers          # Individual finger tests
python run_umap.py hand             # Closed Hand + individual fingers
python run_umap.py all_gestures     # Open/Closed Hand + all finger tests
python run_umap.py comparison       # Ricardo vs Mohak (Closed Hand)

# Run with custom files and labels
python run_umap.py custom --files file1.csv file2.csv --labels "Label1" "Label2"
```

### Using the Shell Script (macOS/Linux)

```bash
# Make the script executable
chmod +x run_umap.sh

# List all available configurations
./run_umap.sh list

# Run with presets
./run_umap.sh default
./run_umap.sh ricardo
./run_umap.sh all
./run_umap.sh emc
./run_umap.sh random
./run_umap.sh fingers
./run_umap.sh hand
./run_umap.sh all_gestures
./run_umap.sh comparison

# Run with custom configuration
./run_umap.sh custom file1.csv "Label 1" file2.csv "Label 2"
```

### Direct Python Script Usage

```bash
# Default configuration (same as repository defaults)
python umap_test.py

# With specific files and labels
python umap_test.py --files "CSV-Files/Test-Ricardo_Open-Hand.csv" "CSV-Files/Test-Ricardo_Closed-Hand.csv" --labels "Open Hand" "Close Hand"

# With custom configuration file
python umap_test.py --config my_custom_config.json
```

## Configuration File Format

Configuration files use JSON format with the following structure:

```json
{
  "configuration_name": {
    "name": "Human-readable configuration name",
    "description": "Description of what this configuration tests",
    "gestures": [
      {
        "file": "path/to/csv/file.csv",
        "label": "Gesture Label"
      },
      {
        "file": "path/to/another/file.csv",
        "label": "Another Gesture Label"
      }
    ]
  }
}
```

### Example: Adding a New Configuration

Edit `gesture_configs.json` and add:

```json
{
  "my_custom_config": {
    "name": "My Custom EMG Analysis",
    "description": "Testing specific gesture patterns",
    "gestures": [
      {
        "file": "CSV-Files/Test-Ricardo_Open-Hand.csv",
        "label": "Open Hand"
      },
      {
        "file": "CustomData/gesture_a.csv",
        "label": "Gesture A"
      },
      {
        "file": "CustomData/gesture_b.csv",
        "label": "Gesture B"
      }
    ]
  }
}
```

Then run with:
```bash
python run_umap.py my_custom_config
```

## Available Preset Configurations

### 1. **Default** - Ricardo's Open/Close Hand
- Files:
  - `CSV-Files/Test-Ricardo_Open-Hand.csv`
  - `CSV-Files/Test-Ricardo_Closed-Hand.csv`
- Visualizes the separation between open and closed hand gestures

### 2. **Ricardo Only** - Mixed Test Data
- Files:
  - `CSV-Files/Test-Ricardo.csv`
- Single file test for Ricardo's mixed gesture data

### 3. **All Subjects** - Multi-Subject Comparison
- Files:
  - `CSV-Files/Test-Ricardo.csv`
  - `CSV-Files/Test-Sirisha.csv`
- Compares EMG patterns between two different subjects

### 4. **EMG_ASL** - EMG_ASL Folder Data
- Files:
  - `EMG_ASL/CSV/Test-Ricardo_Open-Hand.csv`
  - `EMG_ASL/CSV/Test-Ricardo_Closed-Hand.csv`
- Alternative data source from EMG_ASL directory

### 5. **Random Forest** - Multi-Subject Random Forest Dataset
- Files:
  - `RandomForest/DATASET EMG MINDROVE/AYU/subjek_AYU1.csv`
  - `RandomForest/DATASET EMG MINDROVE/DANIEL/subjek_DANIEL1.csv`
  - `RandomForest/DATASET EMG MINDROVE/LINTANG/subjek_Lintang1.csv`
- Compares EMG patterns across multiple subjects from the Random Forest dataset

### 6. **Finger Tests** - Individual Finger Movements
- Files:
  - `CSV-Files/index_finger.csv`
  - `CSV-Files/ring_finger.csv`
  - `CSV-Files/pinky_finger.csv`
- Command: `python run_umap.py fingers`

### 7. **Hand Variants** - Closed Hand + Finger Positions
- Files:
  - `CSV-Files/closed_hand.csv`
  - `CSV-Files/index_finger.csv`
  - `CSV-Files/ring_finger.csv`
  - `CSV-Files/pinky_finger.csv`
- Command: `python run_umap.py hand`

### 8. **All Gestures** - Combined Gesture Set
- Files:
  - `CSV-Files/Test-Ricardo_Open-Hand.csv`
  - `CSV-Files/closed_hand.csv`
  - `CSV-Files/index_finger.csv`
  - `CSV-Files/ring_finger.csv`
  - `CSV-Files/pinky_finger.csv`
- Command: `python run_umap.py all_gestures`

### 9. **Subject Comparison (Closed Hand)** - Ricardo vs Mohak
- Files:
  - `CSV-Files/Test-Ricardo_Closed-Hand.csv`
  - `CSV-Files/Test-Mohak_Closed-Hand.csv`
- Command: `python run_umap.py comparison`

## Batch Testing Examples

### Example 1: Test All Presets Sequentially

Create a script (`test_all.sh`):
```bash
#!/bin/bash
cd /path/to/TritonNeuroTech
source .venv/bin/activate

for config in default ricardo all emc random fingers hand all_gestures comparison; do
    echo "Testing $config..."
    python run_umap.py $config
    echo "Completed $config\n"
done
```

### Example 2: Test Custom Configuration Set

Create a Python script (`batch_test.py`):
```python
import subprocess
import json

# Test configurations
test_cases = [
    {
        'files': ['CSV-Files/Test-Ricardo_Open-Hand.csv'],
        'labels': ['Open Hand']
    },
    {
        'files': ['CSV-Files/Test-Ricardo_Closed-Hand.csv'],
        'labels': ['Close Hand']
    },
    {
        'files': ['CSV-Files/Test-Ricardo_Open-Hand.csv', 'CSV-Files/Test-Ricardo_Closed-Hand.csv'],
        'labels': ['Open Hand', 'Close Hand']
    }
]

for i, test in enumerate(test_cases):
    print(f"\nRunning test case {i+1}...")
    cmd = ['python', 'run_umap.py', 'custom', '--files'] + test['files'] + ['--labels'] + test['labels']
    subprocess.run(cmd)
```

Run with:
```bash
python batch_test.py
```

## Data Format Requirements

All CSV files must be tab-separated (`.csv` with `\t` delimiter) with:
- EMG channel data in numeric columns
- Optional other sensor data (Gyro, Accelerometer, PPG, HR, etc.)
- No special characters or formatting in column names

## Troubleshooting

### Error: "No data files were loaded"
- Check that file paths are correct relative to the current working directory
- Verify files exist: `ls CSV-Files/Test-Ricardo_Open-Hand.csv`

### Error: "Number of files must match number of labels"
- When using `--files` and `--labels`, ensure equal number of arguments
- Example: 2 files must have 2 labels

### Error: "could not convert string to float"
- Verify the CSV file is tab-separated, not comma-separated
- Use text editor to check the delimiter: `od -c file.csv | head -5`

### Matplotlib shows no window
- On headless systems, add this to the script before `plt.show()`:
  ```python
  plt.savefig('output.png')  # Save to file instead
  ```

## Tips & Best Practices

1. **Always activate virtual environment first:**
   ```bash
   source .venv/bin/activate
   ```

2. **Use descriptive labels** for clarity:
   ```bash
   python run_umap.py custom --files data1.csv data2.csv --labels "Left Hand" "Right Hand"
   ```

3. **Create custom configurations** for frequently-used test combinations in `gesture_configs.json`

4. **Test single files** first to verify data before combining multiple files:
   ```bash
   python run_umap.py custom --files "CSV-Files/Test-Ricardo_Open-Hand.csv" --labels "Test Data"
   ```

5. **Check data sizes** to ensure sufficient samples for meaningful clustering:
   - Look for warnings about sample counts in output
   - May need 100+ samples per gesture for good visualization

## Advanced Usage

### Modify PCA/UMAP Parameters

Edit `umap_test.py` to adjust:
```python
# PCA: Change number of components
pca = PCA(n_components=3)  # or 5, 10, etc.

# UMAP: Adjust parameters
reducer = UMAP(
    n_neighbors=15,        # Higher = more global structure
    min_dist=0.1,          # Lower = tighter clusters
    random_state=42        # For reproducibility
)
```

### Save Visualizations

Add to the end of `umap_test.py`:
```python
plt.savefig('umap_comparison.png', dpi=300, bbox_inches='tight')
```

### Export Results Data

Add after UMAP computation:
```python
# Save PCA results
import pandas as pd
pca_df = pd.DataFrame(pca_results, columns=['PC1', 'PC2'])
pca_df['Gesture'] = y
pca_df.to_csv('pca_results.csv', index=False)

# Save UMAP results
umap_df = pd.DataFrame(umap_results, columns=['UMAP1', 'UMAP2'])
umap_df['Gesture'] = y
umap_df.to_csv('umap_results.csv', index=False)
```

## Files Reference

- **umap_test.py** - Main analysis script with CLI argument support
- **run_umap.py** - Python runner for easy configuration management
- **run_umap.sh** - Shell script runner (macOS/Linux)
- **gesture_configs.json** - Preset configuration definitions

---

For questions or modifications, edit the JSON configuration files or command arguments as needed!
