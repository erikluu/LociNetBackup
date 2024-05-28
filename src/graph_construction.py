import json
# import torch
import random
import numpy as np
import networkx as nx
from tqdm import tqdm


def insert_nodes(G, document_ids: list, attributes: dict):
    nodes = []
    for i, id in enumerate(document_ids):
        attr = {k: v[i] for k, v in attributes.items()}
        nodes.append((id, attr))

    G.add_nodes_from(nodes)


def construct_graph(edges: list[tuple], document_ids: list, attributes: dict) -> nx.Graph:
    """
    Generate a graph based on the given neighbors and edge weights.

    Args:
        edges (list[tuple]): List of edges in the form of (node0, node1, weight) tuples.
        document_ids (set[int]): Set of document ids.
        **kwargs: Additional attributes to be passed to the graph nodes.
    Returns:
        nx.Graph: A networkx graph representing the generated graph.
    """
    G = nx.Graph()

    insert_nodes(G, document_ids, attributes)
    G.add_weighted_edges_from(edges)

    return G


"""
More central nodes are more likely to be chosen over a span of lots of new edges...
"""
def watts_strogatz(G, similarity_scores, p, weighted_selection=False, seed=None):
    """
    Transform the original graph into a Watts-Strogatz-like graph.
    https://chih-ling-hsu.github.io/2020/05/15/watts-strogatz

    Parameters:
        original_graph (networkx.Graph): The original graph.
        p (float): The probability of rewiring each edge. No disconnection tho?

    Returns:
        networkx.Graph: The transformed Watts-Strogatz graph.
    """
    if seed:
        np.random.seed(seed)

    new_edges = []

    nodes_to_exclude = {node for node in G.nodes if str(node).startswith("ClusterNode_") or str(node) == "TOP_NODE"}
    subgraph_nodes = [node for node in G.nodes if node not in nodes_to_exclude]
    H = G.subgraph(subgraph_nodes).copy()

    for i, u in (t := tqdm(enumerate(H.nodes()))):
        t.set_description("Watts-Strogatz")
        if str(u).startswith("ClusterNode_") or str(u) == "TOP_NODE":
            continue
        neighbors = list(H.neighbors(u))
        probabilities = similarity_scores[i]
        probabilities[i] = 0 # no loops
        # if a node has n neighbors, it can gain, at most, n more
        for _ in neighbors:
            if random.random() < p:
                if weighted_selection:
                    # probabilities = probabilities / probabilities.sum()  # normalize
                    w = np.random.choice(H.nodes(), p=probabilities)
                    while str(w).startswith("ClusterNode_") or w == u or H.has_edge(u, w):
                        w = np.random.choice(H.nodes(), p=probabilities)    
                else:
                    w = random.choice(list(H.nodes()))
                    while str(w).startswith("ClusterNode_") or w == u or H.has_edge(u, w):
                        w = random.choice(list(H.nodes()))
                new_edges.append((u, w, similarity_scores[u, w].item()))

    G.add_weighted_edges_from(new_edges)
    return G


def barabasi_albert(original_graph, m, n):
    """
    Transform the original graph into a Barabási-Albert graph.

    Parameters:
        original_graph (networkx.Graph): The original graph.
        m (int): The number of edges to attach from a new node to existing nodes.
        n (int): The number of new nodes to add.

    Returns:
        networkx.Graph: The transformed Barabási-Albert graph.
    """
    # Create a new graph with the same nodes as the original
    G_ba = nx.Graph(original_graph)

    # Generate Barabási-Albert graph by adding new nodes with preferential attachment
    for i in range(original_graph.number_of_nodes(), original_graph.number_of_nodes() + n):
        # Select m existing nodes to connect to
        targets = list(G_ba.nodes())
        for _ in range(m):
            target = random.choice(targets)
            G_ba.add_edge(i, target)

    return G_ba


def erdos_renyi(original_graph, p):
    """
    Transform the original graph into an Erdős-Rényi graph.

    Parameters:
        original_graph (networkx.Graph): The original graph.
        p (float): The edge probability for the Erdős-Rényi graph.

    Returns:
        networkx.Graph: The transformed Erdős-Rényi graph.
    """
    # Create a new graph with the same nodes as the original
    G_er = nx.Graph(original_graph)

    # Iterate over all possible edges
    for u, v in original_graph.edges():
        # Decide whether to keep the edge based on probability p
        if random.random() > p:
            G_er.remove_edge(u, v)
        else:
            if not G_er.has_edge(u, v):
                G_er.add_edge(u, v)

    return G_er


if __name__ == "__main__":
    pass
