import json
import torch
import pickle
import psutil
import hashlib
import numpy as np
from sklearn.decomposition import PCA


def hash(string):
    return hashlib.sha1(string.encode()).hexdigest()[:10]


def softmax(x):
    """Compute softmax values for each row of a 2D array."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def save_graph_to_json_and_pickle(G, save_path: str, encoding_f):
    """
    data/graph_medium_1k_tags_all-MiniLM-L6-v2_knn10_kmeans5
    """
    save_graph_to_json(G, f'{save_path}.json')
    save_graph_to_pickle(G, f'{save_path}.pickle')


def save_graph_to_json(G, save_path: str):
    graph_data = {
        "nodes": [],
        "links": []
    }

    for node_id, data in G.nodes(data=True):
        if "encodings" not in data:
            raise ValueError(f"Node {node_id} missing reduced coordinates.")
        if "titles" not in data:
            raise ValueError(f"Node {node_id} missing title.")
        
        node_data = {"id": node_id, "pos": data["encodings"][:2], "color": normalize_and_convert_to_hex(data["encodings"][2:5]), "title": data["titles"], "tags": data["tags"]}
        if "simplified_tags" in node_data.keys():
            node_data["simplified_tags"] = data["simplified_tags"]
        graph_data["nodes"].append(node_data)

    for source, target, data in G.edges(data=True):
        graph_data["links"].append({"source": source, "target": target, "weight": data["weight"]})

    with open(save_path, 'w') as json_file:
        json.dump(graph_data, json_file)


def pickle_to_json(filename, encoding_f):
    assert encoding_f is not None, "Must pass encoding function for dimensionality reduction."
    with open(filename, 'rb') as f:
        G = pickle.load(f)
    
    embeddings = []
    for node_id in G.nodes():
        embedding = G.nodes[node_id]['embeddings']
        embeddings.append(embedding)
    
    encodings = encoding_f(embeddings)
    for i, node_id in enumerate(G.nodes()):
        G.nodes[node_id]['encodings'] = encodings[i].tolist()
    
    save_graph_to_json(G, filename[:-len('.pickle')] + '.json')


def save_graph_to_pickle(G, filename: str):
    assert filename[-7:] == ".pickle", "Filename must end with .pickle extension."
    pickle.dump(G, open(filename, 'wb'))


def save_to_pickle(object, filename: str):
    assert filename[-7:] == ".pickle", "Filename must end with .pickle extension."
    pickle.dump(object, open(filename, 'wb'))


def load_from_pickle(filename: str):
    assert filename[-7:] == ".pickle", "Provided file must have be a .pickle file."
    return pickle.load(open(filename, 'rb'))


def aggregate_embeddings(embeddings, method='mean', weights=None):
    """
    Aggregate embeddings using the specified method.

    Args:
        embeddings (torch.Tensor): Matrix of embeddings (shape: num_nodes x embedding_dim).
        method (str): Aggregation method ('mean', 'sum', 'weighted_mean', 'max_pooling', 'min_pooling').
        weights (torch.Tensor): Optional weights for weighted aggregation (shape: num_nodes).

    Returns:
        torch.Tensor: Aggregated embedding vector.
    """
    if method == 'mean':
        aggregated_embedding = torch.mean(embeddings, dim=0)
    elif method == 'sum':
        aggregated_embedding = torch.sum(embeddings, dim=0)
    elif method == 'weighted_mean':
        if weights is None:
            raise ValueError("Weights must be provided for weighted aggregation.")
        aggregated_embedding = torch.sum(embeddings * weights[:, None], dim=0) / torch.sum(weights)
    elif method == 'max_pooling':
        aggregated_embedding, _ = torch.max(embeddings, dim=0)
    elif method == 'min_pooling':
        aggregated_embedding, _ = torch.min(embeddings, dim=0)
    else:
        raise ValueError("Invalid aggregation method.")

    return aggregated_embedding


def group_into_clusters(labels, data):
    unique_labels = np.unique(labels)
    groups = {label: [] for label in unique_labels}
    for label, item in zip(labels, data):
        groups[label].append(item)
    return groups


def normalize_and_convert_to_hex(rgb):
    normalized_rgb = [int((value + 1) / 2 * 256) for value in rgb]
    hex_color = '#{:x}{:x}{:x}'.format(*normalized_rgb)
    return hex_color


def pca(matrix: torch.Tensor, n_components: int = 5) -> list[tuple]:
    pca = PCA(n_components=n_components)
    pca.fit(matrix) # pyright: ignore
    return pca.transform(matrix) # pyright: ignore


def argsort(matrix: torch.Tensor, k: int = 0) -> torch.Tensor:
    rankings = torch.argsort(matrix, dim=1, descending=True)
    return rankings[:, :k] if k else rankings


def sort_matrix_values(matrix: torch.Tensor, k: int = 0, descending: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    sorted_indices = torch.argsort(matrix, dim=1, descending=True)
    sorted_values = torch.gather(matrix, 1, sorted_indices)
    if descending:
        sorted_indices = sorted_indices[:, :k] if k else sorted_indices
        sorted_values = sorted_values[:, :k] if k else sorted_values
    else:
        sorted_indices = sorted_indices[:, -k:] if k else sorted_indices
        sorted_values = sorted_values[:, -k:] if k else sorted_values
    return sorted_indices, sorted_values


def batch_size_estimate(em_size, num_em_sets=1):
    mem_req_per_embedding_bytes = em_size.size(-1) * 4 * num_em_sets # req memory in bytes
    mem_req_per_embedding_gbytes = mem_req_per_embedding_bytes / (1024**3) # req memory in GB

    mem = psutil.virtual_memory()
    available_mem = mem.available / (1024**3) # available memory in GB
    max_embeddings_in_memory = available_mem / mem_req_per_embedding_gbytes

    batch_size = max_embeddings_in_memory // 100 # set to a quarter of max
    return int(batch_size)
