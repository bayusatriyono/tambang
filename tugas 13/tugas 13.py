import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# Dataset dari soal
data = np.array([
    [1, 1],  # Data 1
    [4, 1],  # Data 2
    [1, 2],  # Data 3
    [3, 4],  # Data 4
    [5, 4]   # Data 5
])

# Fungsi untuk menampilkan dendrogram
def plot_dendrogram(linkage_method):
    linked = linkage(data, method=linkage_method, metric='euclidean')
    
    plt.figure(figsize=(8, 4))
    dendrogram(linked,
               labels=[1, 2, 3, 4, 5],
               distance_sort='ascending',
               show_leaf_counts=True)
    plt.title(f'Dendrogram - {linkage_method.capitalize()} Linkage')
    plt.xlabel('Data')
    plt.ylabel('Jarak (Euclidean)')
    plt.grid(True)
    plt.show()

# Plot dendrogram untuk ketiga metode linkage
for method in ['single', 'complete', 'average']:
    plot_dendrogram(method)
