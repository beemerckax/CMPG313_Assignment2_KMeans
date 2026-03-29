import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D

# 1. Generate Synthetic Dataset
# We create 7 distinct clusters to start with
X, y = make_blobs(n_samples=500, centers=7, cluster_std=0.60, random_random=42)

def plot_clusters(k_value, title, filename):
    kmeans = KMeans(n_clusters=k_value, init='k-means++', max_iter=300, n_init=10, random_state=0)
    y_kmeans = kmeans.fit_predict(X)
    
    # Create 2D Topology Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5, marker='X')
    plt.title(f'Network Topology ({title})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(filename)
    plt.show()
    return kmeans

# 2. Run for k=7 (Initial State)
kmeans_7 = plot_clusters(7, "k=7", "01_initial_topology_k7.png")

# 3. Run for k=3 (Merged/Optimal State)
kmeans_3 = plot_clusters(3, "k=3", "04_merged_topology_k3.png")

# 4. Generate 3D Energy Surface (Cluster Density)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Create a meshgrid for the surface
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Calculate "Energy" (Distance to nearest centroid)
Z = kmeans_3.transform(np.c_[xx.ravel(), yy.ravel()]).min(axis=1)
Z = Z.reshape(xx.shape)

surf = ax.plot_surface(xx, yy, Z, cmap='magma', edgecolor='none')
ax.set_title('3D Energy Surface (k=3)')
plt.colorbar(surf)
plt.savefig("05_merged_3d_surface.png")
plt.show()

print("Execution Complete. Images saved for GitHub upload.")
