import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from copy import deepcopy
from sklearn.datasets import make_moons

class KMeans:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        #randomly select n clusters for inital centroids
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        for _ in range(self.max_iter):
            self.labels = self._assign_labels(X)
            new_centroids = self._calculate_centroids(X)
            if np.all(np.linalg.norm(self.centroids - new_centroids, axis=1) < self.tol):
                break
            self.centroids = new_centroids

    def _assign_labels(self, X):
        #calculate eaclidean distance to each centroid
        return np.argmin(np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2), axis=1)

    def _calculate_centroids(self, X):
        #calculate mean position of all points in a cluster
        return np.array([X[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])

    def predict(self, X):
        return self._assign_labels(X)
    
class DBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None

    def fit(self, data):
        n_points = data.shape[0]
        #init all labels as -1 (invalid)
        self.labels = np.full(n_points, -1)
        visited = np.zeros(n_points, dtype=bool)
        cluster_id = 0

        for point_idx in range(n_points):
            if not visited[point_idx]:
                visited[point_idx] = True
                neighbors = self._check_neighbors(data, point_idx)
                if len(neighbors) >= self.min_samples:
                    self._expand_cluster(data, point_idx, neighbors, cluster_id, visited)
                    cluster_id += 1

        return self

    def predict(self):
        return self.labels

    def _check_neighbors(self, data, point_idx):
        distances = np.linalg.norm(data - data[point_idx], axis=1)
        return np.where(distances < self.eps)[0]

    def _expand_cluster(self, data, point_idx, neighbors, cluster_id, visited):
        self.labels[point_idx] = cluster_id
        queue = list(neighbors)

        while queue:
            current_idx = queue.pop(0)
            if not visited[current_idx]:
                visited[current_idx] = True
                current_neighbors = self._check_neighbors(data, current_idx)
                if len(current_neighbors) >= self.min_samples:
                    queue.extend(current_neighbors)
            if self.labels[current_idx] == -1:
                self.labels[current_idx] = cluster_id



if __name__ == "__main__":
    X, y = make_moons(n_samples=1000, noise=0.07)

    X_kmeans = deepcopy(X)
    kmeans = KMeans(n_clusters=2, max_iter=50)
    kmeans.fit(X_kmeans)
    kmeans_labels = kmeans.predict(X_kmeans)

    X_dbscan = deepcopy(X)
    dbscan = DBSCAN(eps=0.1, min_samples=10)
    dbscan.fit(X_dbscan)
    dbscan_labels = dbscan.predict()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='Accent', s=20)
    axes[0].set_title("KMeans")
    
    axes[1].scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='Accent', s=20)
    axes[1].set_title("DBSCAN")
    
    plt.show()
