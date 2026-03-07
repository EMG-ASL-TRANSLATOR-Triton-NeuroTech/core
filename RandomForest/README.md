# RandomForest EMG Pipeline

This folder contains the RandomForest training workflow for EMG gesture classification.

## Files

- `prepare_emg_datasets.py`
  - Formats raw EMG CSV files into RandomForest-ready datasets.
  - Reads from `../CSV-Files` (including nested gesture folders).
  - Writes:
    - `DATASET EMG MINDROVE/FORMATTED/subjek_FORMATTED.csv`
    - `DATASET EMG MINDROVE/FORMATTED/by_gesture/*.csv`

- `emg_classifyier.py`
  - Loads the formatted `by_gesture` dataset.
  - Filters/signals -> extracts windowed features -> trains RandomForest (with GridSearchCV).
  - Prints accuracy + confusion matrix and saves `feature_importance.png`.

- `old_emg_classifier`
  - Older classifier version kept for comparison/reference.

## Gesture Labels

- `1`: `closed_hand`
- `2`: `open_hand`
- `3`: `index_finger`
- `4`: `ring_finger`
- `5`: `pinky_finger`
- `6`: `spider_man`
- `7`: `peace`
- `8`: `hang_loose`

## How To Run

From repository root:

```bash
python3 RandomForest/prepare_emg_datasets.py
python3 RandomForest/emg_classifyier.py
```

## Current Result (Latest Local Run)

- Dataset loaded: `327,092` rows
- Classes used in training (after feature extraction): `1..8`
- Final test accuracy: `0.9667` (96.67%)

Confusion matrix from the same run:

```text
     1    2    3    4    5   6   7    8
1  938    0    2   18   10   0   0    0
2    0  140    0    0    1   0   0    1
3    5    0  184   15    0   0   0    0
4   18    0    3  515    0   0   0    0
5   13    1    0    0  247   0   0    0
6    0    0    0    0    0  92   0    0
7    0    0    0    0    0   0  81    0
8    0    0    0    0    0   0   0  325
```

## Notes

- Accuracy depends on dataset balance and feature windowing.
- If you add new gesture files under `CSV-Files`, rerun `prepare_emg_datasets.py` before training again.
