## Clustering

### Overview

This section covers various clustering techniques and methods for analyzing network data.

### Nodes belonging to multiple clusters

- **NetworkX:**
  - Explore NetworkX functionalities such as `compose`, `union`, and `disjoint_union`.
- **Train GNN (Graph Neural Network):**
  - Utilize GNN to learn representations from data, predict clusters, etc.
- **Graph Models:**
  - Explore graph models like Erdős-Rényi, Barabási-Albert, and Watts-Strogatz.
- **Hierarchical Clustering:**
  - Understand hierarchical clustering techniques that form clusters and then cluster those clusters.

## Improvements

### Aggregator Nodes
- Using this to make sure graph is connected.

### Manual Calculation of Clustering

- Implement manual calculation of clustering to enable the use of similarity matrices without recalculation.
- **compare both orginal and simplified tags**

## Ideas

Small world graph: add new edges based on probability p but weight it by cosine similarity.

Cosine similarity convert to angular distance

## To Test

Spectral Analysis - if things are frequencies and stuff

### Feature Representation

- Explore various methods for feature representation:
  - tfidf vectorizer
  - word2vec
  - doc2vec
  - BERT
  - count vectorization
  - Latent Semantic Analysis (LSA)
  - Latent Dirichlet Allocation (LDA)
  - Non-Negative Matrix Factorization (NMF)
  - GloVe
- **Critical Point and Scale Invariance:**
  - Investigate critical points and scale invariance.
  - Explore methods to identify and utilize scale invariance.
- **Understanding Scale Invariance:**
  - Examine how to prove scale invariance and its significance.

1. Degree Distribution: Analyze the distribution of node degrees in your graph. A power-law distribution indicates scale-free behavior, while a bell-shaped distribution may suggest a more random or regular structure.
2. Clustering Coefficient: Measure the extent of clustering or transitivity in your graph. A higher clustering coefficient suggests a greater tendency for nodes to form tightly interconnected clusters.
3. Average Path Length: Calculate the average shortest path length between all pairs of nodes in your graph. A shorter average path length indicates greater overall connectivity and a higher likelihood of information flow between nodes.
4. Degree Centrality: Assess the centrality of nodes based on their degree, i.e., the number of connections they have. Nodes with higher degree centrality are more central to the network structure.
5. Betweenness Centrality: Measure the extent to which nodes act as intermediaries in connecting other nodes. Nodes with higher betweenness centrality lie on a large number of shortest paths between other nodes.
6. Closeness Centrality: Evaluate how close a node is to all other nodes in the graph. Nodes with higher closeness centrality can quickly interact with other nodes in the network.
7. Community Structure: Identify cohesive groups or communities within your graph using community detection algorithms such as Louvain or Girvan-Newman. Assess the modularity score to quantify the strength of community structure.
8. Assortativity Coefficient: Measure the degree correlation between connected nodes. Positive assortativity indicates that nodes tend to connect to other nodes with similar degrees, while negative assortativity suggests the opposite.
9. Network Efficiency: Assess the efficiency of information transfer in your graph using measures such as global efficiency or local efficiency. These metrics quantify how efficiently information can be exchanged between nodes in the network.
10. Network Resilience: Evaluate the robustness of your graph to node or edge removal. Measure metrics such as the size of the largest connected component or the average node degree under targeted attacks or random failures.
