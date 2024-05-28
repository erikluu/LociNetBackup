# LociNet

This is my thesis project for my Master's of Science in Computer Science at California Polytechnic State University - San Luis Obispo.

It is based on the method of loci is a strategy for memory enhancement, which uses visualizations of familiar spatial environments in order to enhance the recall of information.

[The Method of Loci](https://en.wikipedia.org/wiki/Method_of_loci)

"memory journey, memory palace, journey method, memory spaces, or mind palace technique"

# Code Base

## Main modules

`src/clustering.py` Clusters embeddings.

`src/edge_constructors.py` Constructs edges between embeddings.

`src/embeddings.py` Gets embeddings from corpus using designated model.

`src/graph_construction.py` Builds graph from embedding similarities and clustering information.

`src/similarity.py` Gets similarity between embeddings.

`src/pipeline.py` Composes modules into a pipeline.

`src/utils.py` General utilities.

# Thoughts

## The Work

The goal of this project is to develop a dynamic knowledge graph system to enhance user and machine recall and content exploration. This system will utilize machine learning techniques, including semantic latent space models, dimensionality reduction, and embeddings to analyze relationships between user-generated media chunks and external databases to provide content and structural recommendations.

This repo is specifically exploring the groundwork and bootstrapping of knowledge graphs based on provided textual documents. The goal is to create a system that can take in a corpus of text and output a graph that can be used for further analysis and exploration.

Prior work: [LociMaps](https://github.com/loci-maps/mini-map).

## Ideas

### General Pipeline

1. Embed data
2. Cluster data (hierarchical or not) + generate labels (summaries)
3. Connect into graph

<img src="assets/graph_knn3.png" alt="network" width="563" height="392"/>
<img src="assets/clusters.png" alt="clusters" width="250" height="250"/>

### Validation and Analysis

I decided to simplify the human-labeled tags using ChatGPT. It helps to create a more streamlined, consistent, and accurate validation process for your knowledge graph. It provides a solid foundation for comparing and evaluating the relationships between different concepts within your dataset. Reducing Ambiguity; Improving Comparability; Enhancing Accuracy.

Metric Thoughts:

- How well a human or AI agent can find a document based on a goal query
- Cooccurence of human vs AI tags: "Group X and Y overlap by this much in B subbranches"
- Compare title + text vs. text vs. title
- Average distance between all nodes
- Average distance between tags
- How long to get from X to Y

### Rando Notes

Some words that may or may not be used later for my thesis 🤷‍♂️🤷‍♂️:
We are trying to mimic, why not make something new. The greatest works aren’t people trying to recreate something they’ve read about. The creators of those things didn’t go about it that way. They didn’t have the hindsight.
# LociNetBackup
