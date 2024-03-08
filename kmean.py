import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class KMeansCluster:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.k_values = range(2, 11)
        self.silhouette_scores = []
        self.best_k = None
        self.best_kmeans = None
        self.cluster_labels = None

    def find_best_k(self):
        for k in self.k_values:
            kmeans = KMeans(n_clusters=k)
            labels = kmeans.fit_predict(self.data)
            silhouette_avg = silhouette_score(self.data, labels)
            self.silhouette_scores.append(silhouette_avg)

        self.best_k = self.k_values[np.argmax(self.silhouette_scores)]

    def train_best_kmeans(self):
        self.best_kmeans = KMeans(n_clusters=self.best_k)
        self.cluster_labels = self.best_kmeans.fit_predict(self.data)

    def add_cluster_labels(self):
        self.data['Cluster'] = self.cluster_labels

    def print_cluster_sizes(self):
        cluster_sizes = self.data['Cluster'].value_counts()
        print("Cluster sizes:")
        print(cluster_sizes)

    def print_cluster_centroids(self):
        centroids = self.best_kmeans.cluster_centers_
        print("Cluster centroids:")
        print(centroids)

    def plot_silhouette_scores(self):
        plt.plot(self.k_values, self.silhouette_scores)
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score vs. Number of Clusters')
        plt.show()

# Example usage
kmeans_cluster = KMeansCluster('Wholesale customers data.csv')
kmeans_cluster.find_best_k()
kmeans_cluster.train_best_kmeans()
kmeans_cluster.add_cluster_labels()
kmeans_cluster.print_cluster_sizes()
kmeans_cluster.print_cluster_centroids()
kmeans_cluster.plot_silhouette_scores()
