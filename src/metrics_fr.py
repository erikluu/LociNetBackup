import gc
import random
import itertools
from collections import defaultdict, Counter, deque

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.cluster import homogeneity_score, completeness_score

import src.similarity as sim
import src.aggregation as agg
import src.pipeline as pipe
import src.utils as utils


# --------------------------------------------------------
# ---------------------- Embeddings ----------------------
# --------------------------------------------------------

def group_by_tags(tags):
    tags_to_rows = defaultdict(list)
    for i, tags in enumerate(tags):
        for tag in tags:
            tags_to_rows[tag].append(i)
    return tags_to_rows


def get_node_pairs(tag_dict: dict) -> set:
    node_pairs = set()
    for doc_ids in tag_dict.values():
        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                node_pairs.add((doc_ids[i], doc_ids[j]))
    return node_pairs


def calculate_metrics(similarity_matrix: np.ndarray, tag_dict: dict, metric_name: str, embedding_model, data_source, agg_method) -> pd.DataFrame:
    num_nodes = similarity_matrix.shape[0]
    shared_tag_pairs = get_node_pairs(tag_dict)
    upper_tri_indices = np.triu_indices(num_nodes, k=1)

    all_distances = similarity_matrix[upper_tri_indices]
    shared_tag_distances = [similarity_matrix[i, j] for i, j in shared_tag_pairs]
    
    all_distances = np.array(all_distances)
    shared_tag_distances = np.array(shared_tag_distances)

    metrics = {
        'data_source': data_source,
        'embedding_model': embedding_model,
        'agg_method': agg_method,
        'metric_name': metric_name,
        'metric': ['mean', 'median', 'std_dev'],
        'between_all_nodes': [
            np.mean(all_distances),
            np.median(all_distances),
            np.std(all_distances)
        ],
        'between_shared_tags': [
            np.mean(shared_tag_distances),
            np.median(shared_tag_distances),
            np.std(shared_tag_distances)
        ]
    }
    
    metrics_df = pd.DataFrame(metrics)
    
    return metrics_df


def calculate_embedding_metrics_for_all(cosine_sim, soft_cosine_sim, euclidean_sim, tags, embedding_model: str, data_source: str, agg_method):
    tag_dict = group_by_tags(tags)

    cosine_metrics_df = calculate_metrics(cosine_sim, tag_dict, 'cosine', embedding_model, data_source, agg_method)
    soft_cosine_metrics_df = calculate_metrics(soft_cosine_sim, tag_dict, 'soft_cosine', embedding_model, data_source, agg_method)
    euclidean_metrics_df = calculate_metrics(euclidean_sim, tag_dict, 'euclidean', embedding_model, data_source, agg_method)

    all_metrics_df = pd.concat([cosine_metrics_df, soft_cosine_metrics_df, euclidean_metrics_df], ignore_index=True)

    return all_metrics_df


def get_embedding_similarity_metrics_per_dataset(dataset_name, dataset_tags, model_names, agg_methods, n):
    dataframes = []

    for model_name in model_names:
        for agg_method in agg_methods:
            embeddings = utils.load_from_pickle(f"embeddings/{dataset_name.split('_')[0]}_{model_name}_{agg_method}_n10000.pickle")[:n]
            cosine_sim, soft_cosine_sim, euclidean_sim = sim.get_all_similarities(embeddings)
            dataframes.append(calculate_embedding_metrics_for_all(cosine_sim, soft_cosine_sim, euclidean_sim,
                                            dataset_tags, model_name, dataset_name, agg_method))
    
    return pd.concat(dataframes)


# --------------------------------------------------------
# ---------------------- Clusters-------------------------
# --------------------------------------------------------

def get_tags_count(tags):
    flattened_tags = [tag for sublist in tags for tag in sublist]
    tag_counts = Counter(flattened_tags)
    return tag_counts


def group_into_clusters(labels, data):
    unique_labels = np.unique(labels)
    groups = {label: [] for label in unique_labels}
    for label, item in zip(labels, data):
        groups[label].append(item)
    return groups


def ids_by_clusters(cluster_labels, ids):
    ids_by_cluster = group_into_clusters(cluster_labels, ids)
    return ids_by_cluster


def tags_by_clusters(ids_by_cluster, tags):
    tags_by_cluster = {}
    for cluster_id, doc_ids in ids_by_cluster.items():
        tags_by_cluster[cluster_id] = [tags[id] for id in doc_ids]
    
    return tags_by_cluster


def calculate_homogeneity(tags_by_cluster):
    true_labels = []
    predicted_clusters = []

    for cluster_id, tags in tags_by_cluster.items():
        for tag in itertools.chain.from_iterable(tags):
            true_labels.append(tag)
            predicted_clusters.append(cluster_id)

    return homogeneity_score(true_labels, predicted_clusters)


def calculate_completeness(tags_by_cluster):
    true_labels = []
    predicted_clusters = []

    for cluster_id, tags in tags_by_cluster.items():
        for tag in itertools.chain.from_iterable(tags):
            true_labels.append(tag)
            predicted_clusters.append(cluster_id)

    return completeness_score(true_labels, predicted_clusters)


def calculate_tag_concentration_purity(docs_by_cluster, tag_counts, k):
    """
    Calculate the Tag Concentration Purity.
    
    For each of the K most popular tags, find the cluster that contains the highest
    number of documents with that tag. Calculate the percentage of all documents 
    with that tag contained within this cluster. If there are an even amount of tags in multiple clusters,
    it will average them.
    
    Parameters:
    docs_by_cluster (dict): A dictionary where keys are cluster IDs and values are lists of documents (each document is a set of tags) in that cluster.
    tag_counts (dict): A dictionary where keys are tags and values are the total count of documents with that tag.
    k (int): The number of most popular tags to consider.
    
    Returns:
    dict: A dictionary where keys are tags and values are the purity scores for each tag.
    """
    most_popular_tags = [tag for tag, _ in Counter(tag_counts).most_common(k)]
    
    purity_scores = {}
    
    for tag in most_popular_tags:
        max_cluster_count = 0
        max_cluster_ids = []
        for cluster_id, docs in docs_by_cluster.items():
            # Count occurrences of the tag in the current cluster
            tag_count_in_cluster = sum(1 for doc in docs if tag in doc)
            if tag_count_in_cluster > max_cluster_count:
                max_cluster_count = tag_count_in_cluster
                max_cluster_ids = [cluster_id]
            elif tag_count_in_cluster == max_cluster_count:
                max_cluster_ids.append(cluster_id)
        
        # Calculate purity score for the tag
        scores = []
        for cluster_id in max_cluster_ids:
            if tag in tag_counts and tag_counts[tag] > 0:
                scores.append(max_cluster_count / len(docs_by_cluster[cluster_id]))
        purity_scores[tag] = round(np.mean(scores), 3)
    
    return purity_scores


def calculate_cluster_tag_purity(docs_by_cluster, tag_counts, total_docs, k):
    """
    Calculate the Cluster Tag Purity.
    
    For each of the K most popular tags, find the cluster that contains the highest
    number of documents with that tag. Calculate the percentage of documents in this
    cluster that have the tag, over all documents in the dataset.
    
    Parameters:
    docs_by_cluster (dict): A dictionary where keys are cluster IDs and values are lists of documents (each document is a set of tags) in that cluster.
    tag_counts (dict): A dictionary where keys are tags and values are the total count of documents with that tag.
    total_docs (int): The total number of documents in the dataset.
    k (int): The number of most popular tags to consider.
    
    Returns:
    dict: A dictionary where keys are tags and values are the purity scores for each tag.
    """
    # Find the K most popular tags
    most_popular_tags = [tag for tag, _ in Counter(tag_counts).most_common(k)]
    
    purity_scores = {}
    
    for tag in most_popular_tags:
        max_cluster_count = 0
        
        for _, docs in docs_by_cluster.items():
            # Count occurrences of the tag in the current cluster
            tag_count_in_cluster = sum(1 for doc in docs if tag in doc)
            if tag_count_in_cluster > max_cluster_count:
                max_cluster_count = tag_count_in_cluster
        
        # Calculate purity score for the tag
        if total_docs > 0:
            purity_score = max_cluster_count / total_docs
        else:
            purity_score = 0  # If there are no documents, purity is 0
        
        purity_scores[tag] = round(purity_score, 3)
    
    return purity_scores


def compare_cluster_metrics(dataset_name, embedding_models, agg_methods, clusterer_functions, n, ids, tags, k):
    results = []

    tag_counts = get_tags_count(tags)
    total_docs = sum(tag_counts.values())

    for embedding_model in embedding_models:
        for agg_method in agg_methods:
            embeddings = utils.load_from_pickle(f"embeddings/{dataset_name.split('_')[0]}_{embedding_model}_{agg_method}_n10000.pickle")[:n]
            for clusterer_name, clusterer_f in clusterer_functions.items():
                try:
                    cluster_labels = clusterer_f(embeddings)
                except Exception as e:
                    print(f"Error: {e}")
                    cluster_labels = None

                if cluster_labels is None:
                    print(f"Skipping {embedding_model}, {agg_method}, {clusterer_name} due to insufficient samples")
                    results.append({
                        "dataset": dataset_name,
                        'embedding_model': embedding_model,
                        'agg_method': agg_method,
                        'clusterer': clusterer_name,
                        'homogeneity': None,
                        'completeness': None,
                        'tag_concentration_purity': None,
                        'cluster_tag_purity': None
                    })
                    continue
                
                ids_by_cluster = ids_by_clusters(cluster_labels, ids)
                tags_by_cluster = tags_by_clusters(ids_by_cluster, tags)
                
                homogeneity = round(calculate_homogeneity(tags_by_cluster), 3)
                completeness = round(calculate_completeness(tags_by_cluster), 3)
                tag_concentration_purity = calculate_tag_concentration_purity(tags_by_cluster, tag_counts, k)
                cluster_tag_purity = calculate_cluster_tag_purity(tags_by_cluster, tag_counts, total_docs, k)

                results.append({
                    "dataset": dataset_name,
                    'embedding_model': embedding_model,
                    'agg_method': agg_method,
                    'clusterer': clusterer_name,
                    'homogeneity': homogeneity,
                    'completeness': completeness,
                    'tag_concentration_purity': tag_concentration_purity,
                    'cluster_tag_purity': cluster_tag_purity
                })

                # Free up memory
                del cluster_labels, ids_by_cluster, tags_by_cluster, homogeneity, completeness, tag_concentration_purity, cluster_tag_purity
                gc.collect()

    results_df = pd.DataFrame(results)
    return results_df




# --------------------------------------------------------
# -------------------- Graph Separation ------------------
# --------------------------------------------------------

def bfs_tag_connectivity(G, max_depth=3):
    connectivity = {depth: 0 for depth in range(1, max_depth + 1)}
    tag_counts = {depth: Counter() for depth in range(1, max_depth + 1)}
    visited = set()

    for node in G.nodes():
        if node in visited:
            continue
        visited.add(node)
        queue = deque([(node, 0)])
        level_nodes = defaultdict(set)

        while queue:
            current_node, depth = queue.popleft()
            if depth >= max_depth:
                continue

            for neighbor in G.neighbors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
                    shared_tags = set(G.nodes[current_node]['tags']) & set(G.nodes[neighbor]['tags'])
                    if shared_tags: # share at least two tags (only for ArXiv)
                        level_nodes[depth + 1].add(neighbor)
                        tag_counts[depth + 1].update(shared_tags)

        for depth in level_nodes:
            connectivity[depth] += len(level_nodes[depth])

    # Calculate percentages
    total_nodes = G.number_of_nodes()
    connectivity_percentages = {depth: (count, count / total_nodes) for depth, count in connectivity.items() if count > 0}
    
    # Filter out empty tag counts
    tag_counts = {depth: dict(tags) for depth, tags in tag_counts.items() if tags}

    return connectivity_percentages, tag_counts


def degree_of_separation(G):
    """
    Calculate the degree of separation between nodes that share one or more tags.
    
    Parameters:
    G (Graph): The networkx graph with nodes having a 'tags' attribute.
    
    Returns:
    float: The average degree of separation between nodes that share one or more tags.
    """
    path_lengths = []
    selected_nodes_sets = []
    # Select 3 sets of 20 nodes to test
    if len(G.nodes()) > 20:
        selected_nodes_sets.append(random.sample(list(G.nodes()), 20))
        selected_nodes_sets.append(random.sample(list(G.nodes()), 20))
        selected_nodes_sets.append(random.sample(list(G.nodes()), 20))
    else:
        selected_nodes_sets.append(list(G.nodes()))

    for selected_nodes in selected_nodes_sets:
        for node in selected_nodes:
            for neighbor in selected_nodes:
                if node != neighbor and set(G.nodes[node]['tags']) & set(G.nodes[neighbor]['tags']):
                    try:
                        length = nx.shortest_path_length(G, source=node, target=neighbor, weight='weight')
                        path_lengths.append(length)
                    except nx.NetworkXNoPath:
                        continue

    if path_lengths:
        return np.mean(path_lengths)
    else:
        return float('inf')  # If no paths are found


def calculate_edge_assignment_metrics(G, max_depth=3):
    """
    Calculate edge assignment metrics.
    
    Parameters:
    G (Graph): The networkx graph with nodes having a 'tags' attribute.
    max_depth (int): The maximum depth for BFS.
    
    Returns:
    dict: A dictionary containing the edge assignment metrics.
    """
    tag_connectivity, tag_counts = bfs_tag_connectivity(G, max_depth)
    # separation_degree = degree_of_separation(G)
    separation_degree = None

    metrics = {
        'tag_connectivity': tag_connectivity,
        'tag_counts': tag_counts,
        'degree_of_separation': separation_degree
    }

    return metrics


def compare_edge_assignment_metrics(dataset_name, embedding_models, agg_methods, clusterer_functions, edge_constructor_functions, n, ids, tags, titles, max_depth=3):
    results = []

    for embedding_model in embedding_models:
        for agg_method in agg_methods:
            embeddings = utils.load_from_pickle(f"embeddings/{dataset_name.split('_')[0]}_{embedding_model}_{agg_method}_n10000.pickle")[:n]
            cosine_sim, soft_cosine_sim, euclidean_sim = sim.get_all_similarities(embeddings)
            for sim_mat, sim_metric in zip([cosine_sim, soft_cosine_sim, euclidean_sim], ["cosine", "soft_cosine", "euclidean"]):
                for edge_name, edge_f in edge_constructor_functions.items():
                    for clusterer_name, clusterer_f in clusterer_functions.items():
                        print(f"graphs/{dataset_name}_{embedding_model}_{sim_metric}_{agg_method}_{edge_name}_{clusterer_name}")
                        try:
                            G = pipe.cluster_and_connect(embeddings, sim_mat, ids, sim_metric, edge_f, clusterer_f, agg.mean_pooling, tags=tags, titles=titles)
                            metrics = calculate_edge_assignment_metrics(G, max_depth)
                        except Exception as e:
                            print(f"Error: {e}")
                            results.append({
                                "dataset": dataset_name,
                                'embedding_model': embedding_model,
                                'agg_method': agg_method,
                                'similarity': sim_metric,
                                'edge constructor': edge_name,
                                'clusterer': clusterer_name,
                                'depth': None,
                                'connected_nodes': None,
                                'percentage_connected': None,
                                'tag_counts': None,
                                'degree_of_separation': None
                            })
                            continue

                        # utils.save_graph_to_pickle(G, f"graphs/{dataset_name}_{embedding_model}_{sim_metric}_{agg_method}_{edge_name}_{clusterer_name}.pickle")
                        
                        tag_connectivity = metrics['tag_connectivity']
                        tag_counts = metrics['tag_counts']
                        degree_of_separation = metrics['degree_of_separation']

                        for depth, (connected_nodes, percentage) in tag_connectivity.items():
                            results.append({
                                "dataset": dataset_name,
                                'embedding_model': embedding_model,
                                'agg_method': agg_method,
                                'similarity': sim_metric,
                                'edge constructor': edge_name,
                                'clusterer': clusterer_name,
                                'depth': depth,
                                'connected_nodes': connected_nodes,
                                'percentage_connected': round(percentage, 3),
                                'tag_counts': tag_counts,
                                'degree_of_separation': degree_of_separation
                            })

                        # Free up memory
                        del G, metrics, tag_connectivity, degree_of_separation
                        gc.collect()

    results_df = pd.DataFrame(results)
    return results_df
