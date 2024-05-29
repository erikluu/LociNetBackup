import torch
import itertools
import networkx as nx
import numpy as np

import src.utils as utils
import src.similarity as ss
import src.graph_construction as gc


def add_cluster_node(G, similarity_metric, aggregator_f, **kwargs):
    embeddings = kwargs["embeddings"]
    del kwargs["embeddings"]
    tags = set(itertools.chain(*kwargs["tags"]))
    tags.add("CLUSTER")

    cluster_node_id = f"ClusterNode_{utils.hash(str(embeddings))}"
    cluster_node = {
        "titles": cluster_node_id,
        "tags": list(tags),
        "embeddings": aggregator_f(embeddings),
    }

    if "cluster_assignment" in kwargs.keys():
        cluster_node["cluster_id"] = kwargs["cluster_assignment"][0]
    else:
        cluster_node["cluster_id"] = 0

    G.add_node(cluster_node_id, **cluster_node)

    similarity_scores = ss.batch_similarity_scores(embeddings, similarity_metric)[-1] # pyright: ignore 

    weighted_edges = []
    for i, node_id in enumerate(G.nodes):
        if node_id != cluster_node_id:
            weighted_edges.append((node_id, cluster_node_id, similarity_scores[i].item()))
    
    G.add_weighted_edges_from(weighted_edges)

    return G


def connect_directly(embeddings: torch.Tensor, similarity_scores, document_ids: list, similarity_metric: str, edge_constructor_f, aggregator_f, **kwargs):
    assert embeddings.size()[0] == similarity_scores.size()[0] == len(document_ids)

    edges = edge_constructor_f(similarity_scores, document_ids)
    
    kwargs["embeddings"] = embeddings
    G = gc.construct_graph(edges, document_ids, kwargs)
    G = add_cluster_node(G, similarity_metric, aggregator_f, **kwargs)

    return G


def connect_cluster_nodes(G, similarity_metric, edge_constructor_f, aggregator_f):
    """
    Gonna connect everything together for now just to guarentee a complete graph.
    Or just make an extra level to make sure the graph is complete.
    """

    cluster_node_ids, embeddings = zip(*[
        (node[0], node[1]["embeddings"]) 
        for node in G.nodes(data=True) 
        if str(node[0]).startswith("ClusterNode_") and "embeddings" in node[1]
    ])
    embeddings = torch.stack(embeddings)

    top_node_id = "TOP_NODE"
    top_node_embedding = aggregator_f(embeddings)
    top_node = {
        "titles": "TOP NODE",
        "tags": ["TOP_NODE"],
        "embeddings": top_node_embedding,
        "number_of_clusters": len(cluster_node_ids)
    }
    G.add_node(top_node_id, **top_node)

    embeddings = torch.cat((embeddings, top_node_embedding.unsqueeze(0)), dim=0)
    similarity_scores = ss.batch_similarity_scores(embeddings, similarity_metric)

    top_node_scores = similarity_scores[-1] # pyright: ignore 
    remaining_scores = similarity_scores[:-1, :-1] # pyright: ignore 

    edges = edge_constructor_f(remaining_scores, cluster_node_ids)

    G.add_weighted_edges_from(edges)

    cluster_to_top_edges = []
    for i, cluster_node_id in enumerate(cluster_node_ids):
        cluster_to_top_edges.append((top_node_id, cluster_node_id, top_node_scores[i].item()))
    
    G.add_weighted_edges_from(cluster_to_top_edges)
    
    return G


def cluster_and_connect(embeddings: torch.Tensor, similarity_scores, document_ids: list, similarity_metric, edge_constructor_f, clusterer_f, aggregator_f, **kwargs):
    cluster_labels = clusterer_f(embeddings)
    clusters = utils.group_into_clusters(cluster_labels, zip(embeddings, document_ids))

    combined_graph = nx.Graph()
    for i, cluster in enumerate(clusters.values()):
        cluster_embeddings, ids_in_cluster = zip(*cluster)
        cluster_embeddings = torch.stack(cluster_embeddings)

        indices_to_keep = torch.tensor(ids_in_cluster) # should never have cluster_id
        cluster_similarity_scores = similarity_scores[indices_to_keep][:, indices_to_keep]

        filtered_attributes = {attr: [values[i] for i in ids_in_cluster] for attr, values in kwargs.items()}
        G = connect_directly(cluster_embeddings, cluster_similarity_scores, ids_in_cluster, similarity_metric, edge_constructor_f, aggregator_f, **filtered_attributes)
        for node_id in G.nodes():
            G.nodes[node_id]["cluster_assignment"] = i
        combined_graph = nx.compose(combined_graph, G) # type: ignore - nx.compose, nx.union, or nx.disjoint_union 

    combined_graph = connect_cluster_nodes(combined_graph, similarity_metric, edge_constructor_f, aggregator_f)
    return combined_graph