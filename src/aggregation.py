import torch
from src.utils import softmax

def mean_pooling(embedding_vectors):
    return torch.mean(embedding_vectors, dim=0)


def sum_pooling(embedding_vectors):
    return torch.sum(embedding_vectors, dim=0)


def max_pooling(embedding_vectors):
    return torch.max(embedding_vectors, dim=0)


def min_pooling(embedding_vectors):
    return torch.min(embedding_vectors, dim=0)


def self_attention(embedding_vectors):
    attention_scores = torch.dot(embedding_vectors, embedding_vectors.T)
    
    attention_weights = softmax(attention_scores)
    
    aggregated_embedding = torch.dot(attention_weights, embedding_vectors)
    
    return aggregated_embedding


def global_attention(embedding_vectors):
    query_vector = torch.mean(embedding_vectors, dim=0)
    
    attention_scores = torch.dot(embedding_vectors, query_vector)
    
    attention_weights = softmax(attention_scores)
    
    aggregated_embedding = torch.dot(attention_weights, embedding_vectors)
    
    return aggregated_embedding


def content_based_attention(embedding_vectors):
    dot_products = torch.dot(embedding_vectors, embedding_vectors.T)
    norms = torch.linalg.norm(embedding_vectors, axis=1, keepdims=True)
    similarity_matrix = dot_products / (torch.dot(norms, norms.T) + 1e-10)
    
    attention_weights = softmax(similarity_matrix)
    
    aggregated_embedding = torch.dot(attention_weights, embedding_vectors)
    
    return aggregated_embedding


def learned_self_attention(embedding_vectors):
    query_vector = torch.randn(embedding_vectors.shape[1])
    
    attention_scores = torch.dot(embedding_vectors, query_vector)
    
    attention_weights = softmax(attention_scores)
    
    aggregated_embedding = torch.dot(attention_weights, embedding_vectors)
    
    return aggregated_embedding



