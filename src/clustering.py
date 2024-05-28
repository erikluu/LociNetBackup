import torch
from sklearn.cluster import DBSCAN, KMeans, Birch
from sklearn.mixture import GaussianMixture


def kmeans(embeddings: torch.Tensor, n_clusters: int = 5):
    if embeddings.shape[0] < 2:
        return None
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(embeddings)
    return kmeans.labels_

def dbscan(embeddings: torch.Tensor, eps: float = 0.5, min_samples: int = 5, metric: str = "cosine"):
    if embeddings.shape[0] < 2:
        return None
    cluster_labels = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit_predict(embeddings)
    return cluster_labels

def gmm(embeddings, n_components=3, random_state=42):
    if embeddings.shape[0] < 2:
        return None
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    cluster_labels = gmm.fit_predict(embeddings)
    return cluster_labels

def birch(embeddings, n_clusters=3):
    if embeddings.shape[0] < 2:
        return None
    birch = Birch(n_clusters=n_clusters)
    cluster_labels = birch.fit_predict(embeddings)
    return cluster_labels