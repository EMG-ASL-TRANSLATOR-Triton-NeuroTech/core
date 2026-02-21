import pandas as pd
from sklearn.decomposition import PCA
from umap import UMAP
import matplotlib.pyplot as plt

# 1. Load your preprocessed CSV data
data = pd.read_csv('Test-Ricardo_Closed-Hand.csv')

# 2. Linear Reduction (PCA)
# Helps you see the 'broad strokes' of your EMG signal variance
pca = PCA(n_components=2)
pca_results = pca.fit_transform(data)

# 3. Non-Linear Reduction (UMAP)
# Helps you see if 'Closed-Hand' is physically distinct from 'Open-Hand'
reducer = UMAP(n_neighbors=15, min_dist=0.1)
umap_results = reducer.fit_transform(data)

# 4. Visualization
plt.scatter(umap_results[:, 0], umap_results[:, 1], c=labels)
plt.title("Gesture Cluster Analysis")
plt.show()