import torch
import random
import networkx as nx
from src.utils import sort_matrix_values
from sklearn.cluster import SpectralClustering

def random_edges(sim_mat: torch.Tensor, document_ids: list[int], num_edges_per_node: int = 5):
    """
    Returns a list of randomly assigned neighbors in the form of (node0, node1, weight) tuples.
    Each node will have a specified number of random connections.
    """
    num_docs = sim_mat.shape[0]
    edges = []

    for i in range(num_docs):
        node0 = document_ids[i]
        possible_nodes = list(range(num_docs))
        possible_nodes.remove(i)  # Exclude self-connection
        random.shuffle(possible_nodes)
        selected_nodes = possible_nodes[:num_edges_per_node]

        for j in selected_nodes:
            node1 = document_ids[j]
            weight = sim_mat[i, j].item()  # Use similarity as weight, can also be random
            edges.append((node0, node1, weight))

    return edges


def knn(sim_mat: torch.Tensor, document_ids: list[int], k: int = 10):
    """
    Returns a list of neighbors in the form of (node0, node1, weight) tuples.
    """
    indices, values = sort_matrix_values(sim_mat, k=k+1) # not including itself
    indices = indices[:, 1:]
    weights = values[:, 1:]

    edges = []
    for i in range(len(indices)):
        node0 = document_ids[i]
        for j, weight in zip(indices[i], weights[i]):
            node1 = document_ids[j]
            edges.append((node0, node1, weight.item()))

    return edges


def knn_mst(sim_mat: torch.Tensor, document_ids: list[int], k: int = 10):
    """
    Combines KNN and MST for edge assignment.
    """
    # Step 1: KNN for initial connections
    knn_edges = knn(sim_mat, document_ids, k)

    # Create a graph from KNN edges
    G_knn = nx.Graph()
    G_knn.add_weighted_edges_from(knn_edges)

    # Step 2: Apply MST to ensure all nodes are connected
    mst_edges = list(nx.minimum_spanning_edges(G_knn, data=True))
    
    # Combine KNN and MST edges, avoiding duplicates
    combined_edges = set(knn_edges + [(u, v, d['weight']) for u, v, d in mst_edges])

    return list(combined_edges)


def threshold_based_edge_assignment(sim_mat: torch.Tensor, document_ids: list[int], threshold: float = 0.5):
    """
    Returns a list of neighbors in the form of (node0, node1, weight) tuples
    for edges with similarity above the given threshold.
    """
    num_docs = sim_mat.shape[0]
    edges = []
    for i in range(num_docs):
        node0 = document_ids[i]
        for j in range(num_docs):
            if i != j:  # exclude self-similarity
                node1 = document_ids[j]
                weight = sim_mat[i, j].item()
                if weight >= threshold:
                    edges.append((node0, node1, weight))
    return edges


def mutual_knn_edge_assignment(sim_mat: torch.Tensor, document_ids: list[int], k: int = 10):
    """
    Returns a list of neighbors in the form of (node0, node1, weight) tuples
    for mutual k-nearest neighbors.
    """
    indices, values = sort_matrix_values(sim_mat, k=k+1)  # not including itself
    indices = indices[:, 1:]
    weights = values[:, 1:]

    mutual_edges = set()
    for i in range(len(indices)):
        node0 = document_ids[i]
        for j, weight in zip(indices[i], weights[i]):
            node1 = document_ids[j]
            if i in indices[j]:
                mutual_edges.add((min(node0, node1), max(node0, node1), weight.item()))

    return list(mutual_edges)


def spectral_clustering_edge_assignment(sim_mat: torch.Tensor, document_ids: list[int], n_clusters: int = 5):
    """
    Returns a list of neighbors in the form of (node0, node1, weight) tuples
    based on spectral clustering.
    """
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=0)
    labels = clustering.fit_predict(sim_mat.cpu().numpy())
    edges = []
    num_docs = sim_mat.shape[0]
    for i in range(num_docs):
        node0 = document_ids[i]
        for j in range(num_docs):
            if i != j and labels[i] == labels[j]:  # same cluster, exclude self-similarity
                node1 = document_ids[j]
                weight = sim_mat[i, j].item()
                edges.append((node0, node1, weight))
    return edges