import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler # Added for best practice
from umap import UMAP
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION: Define your files here
# ==========================================
# Format: ('path_to_csv', 'Gesture_Name')
gesture_files = [
    ('CSV-Files/Test-Ricardo_Open-Hand.csv', 'Open Hand'),
    ('CSV-Files/Test-Ricardo_Closed-Hand.csv', 'Close Hand'),
]

# ==========================================
# 2. Load and Combine Data
# ==========================================
dataframes = []

print("Loading data...")
for filepath, gesture_name in gesture_files:
    try:
        df = pd.read_csv(filepath, sep='\t')
        # Add a column to identify which gesture this data belongs to
        df['gesture'] = gesture_name
        dataframes.append(df)
        print(f"  Loaded {filepath} ({len(df)} samples)")
    except FileNotFoundError:
        print(f"  Warning: File not found - {filepath}")

if not dataframes:
    raise ValueError("No data files were loaded. Check your file paths.")

# Combine all CSVs into one master DataFrame
combined_data = pd.concat(dataframes, ignore_index=True)

# Separate Features (EMG channels) from Labels (Gesture names)
# We assume the 'gesture' column is the last one added.
# If your CSVs have other non-EMG columns, adjust this selection.
feature_columns = [col for col in combined_data.columns if col != 'gesture']
X = combined_data[feature_columns].values
y = combined_data['gesture'].values

# ==========================================
# 3. Preprocessing (Optional but Recommended)
# ==========================================
# EMG data often benefits from scaling before PCA/UMAP
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 4. Dimensionality Reduction
# ==========================================
print("Running PCA...")
pca = PCA(n_components=2)
pca_results = pca.fit_transform(X_scaled)

print("Running UMAP...")
# UMAP is non-deterministic; set random_state for reproducibility
reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
umap_results = reducer.fit_transform(X_scaled)

# ==========================================
# 5. Visualization
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Get unique gestures to plot them with different colors
unique_gestures = np.unique(y)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_gestures)))

# Plot PCA results
for i, gesture in enumerate(unique_gestures):
    mask = y == gesture
    axes[0].scatter(
        pca_results[mask, 0],
        pca_results[mask, 1],
        c=[colors[i]],
        label=gesture,
        alpha=0.7,
        edgecolors='w',
        s=50
    )

axes[0].set_title("Gesture Cluster Analysis (PCA)")
axes[0].set_xlabel("PCA Component 1")
axes[0].set_ylabel("PCA Component 2")
axes[0].legend(title="Gesture")
axes[0].grid(True, linestyle='--', alpha=0.3)

# Plot UMAP results
for i, gesture in enumerate(unique_gestures):
    mask = y == gesture
    axes[1].scatter(
        umap_results[mask, 0],
        umap_results[mask, 1],
        c=[colors[i]],
        label=gesture,
        alpha=0.7,
        edgecolors='w',
        s=50
    )

axes[1].set_title("Gesture Cluster Analysis (UMAP)")
axes[1].set_xlabel("UMAP Component 1")
axes[1].set_ylabel("UMAP Component 2")
axes[1].legend(title="Gesture")
axes[1].grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()