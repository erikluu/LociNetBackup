import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, logging
from tqdm import tqdm
from typing import Any, List, Tuple
import gensim.downloader as api
from gensim.models import Word2Vec
import numpy as np

logging.set_verbosity_error()

# -----------------
# Initializers
# -----------------

def initialize_sentence_transformer_model(model_id: str) -> Tuple[Any, Any, int]:
    print(f"Initializing {model_id} Model")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, return_dict=True)
    max_length = tokenizer.model_max_length
    return tokenizer, model, max_length

def initialize_nomic_model() -> Tuple[Any, Any, int]:
    print("Initializing Nomic Model")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, safe_serialization=True)
    model.eval()
    max_length = tokenizer.model_max_length
    return tokenizer, model, max_length

def initialize_google_bert_model() -> Tuple[Any, Any, int]:
    print("Initializing Google BERT Model")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    model.eval()
    max_length = tokenizer.model_max_length
    return tokenizer, model, max_length

def initialize_specter_base_model() -> Tuple[Any, Any, int]:
    print("Initializing AllenAI Specter Model")
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
    model = AutoModel.from_pretrained("allenai/specter")
    model.eval()
    # max_length = tokenizer.model_max_length # returns 1000000000000000019884624838656 lmao
    max_length = 512
    return tokenizer, model, max_length

# def initialize_llama_model() -> Tuple[Any, Any, int]:
#     print("Initializing Llama Model")
#     tokenizer = AutoTokenizer.from_pretrained("deerslab/llama-7b-embeddings", trust_remote_code=True)
#     model = AutoModel.from_pretrained("deerslab/llama-7b-embeddings", trust_remote_code=True)
#     model.eval()
#     max_length = tokenizer.model_max_length
#     return tokenizer, model, max_length 

def initialize_word2vec_model() -> Word2Vec:
    print("Initializing Word2Vec Model")
    model = api.load("word2vec-google-news-300")
    return model # pyright: ignore

# -----------------
# Pooling Functions
# -----------------

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def sum_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1)

def max_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.max(token_embeddings * input_mask_expanded, 1)[0]

def min_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.min(token_embeddings * input_mask_expanded, 1)[0]

def self_attention(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    attention_scores = torch.matmul(token_embeddings, token_embeddings.transpose(-1, -2))
    attention_weights = F.softmax(attention_scores, dim=-1)
    aggregated_embedding = torch.matmul(attention_weights, token_embeddings)
    return torch.mean(aggregated_embedding, dim=1)

def global_attention(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    query_vector = torch.mean(token_embeddings, dim=1, keepdim=True)
    attention_scores = torch.matmul(token_embeddings, query_vector.transpose(-1, -2))
    attention_weights = F.softmax(attention_scores, dim=1)
    aggregated_embedding = torch.matmul(attention_weights.transpose(-1, -2), token_embeddings)
    return aggregated_embedding[:, 0, :]

def content_based_attention(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    dot_products = torch.matmul(token_embeddings, token_embeddings.transpose(-1, -2))
    norms = torch.linalg.norm(token_embeddings, dim=2, keepdims=True)
    similarity_matrix = dot_products / (torch.matmul(norms, norms.transpose(-1, -2)) + 1e-10)
    attention_weights = F.softmax(similarity_matrix, dim=-1)
    aggregated_embedding = torch.matmul(attention_weights, token_embeddings)
    return torch.mean(aggregated_embedding, dim=1)

def learned_self_attention(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    query_vector = torch.randn(token_embeddings.shape[-1])
    attention_scores = torch.matmul(token_embeddings, query_vector)
    attention_weights = F.softmax(attention_scores, dim=1)
    aggregated_embedding = torch.matmul(attention_weights.transpose(-1, -2), token_embeddings)
    return aggregated_embedding[:, 0, :]


# -----------------
# Utility Functions
# -----------------

def split_into_chunks(text: str, tokenizer, max_length: int) -> List[str]:
    """Split text into chunks that fit within the model's max context length."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_length - 2):
        chunk = tokens[i:i + max_length - 2]
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        chunk_tokens = tokenizer.encode(chunk_text, add_special_tokens=True)
        if len(chunk_tokens) <= max_length:
            chunks.append(chunk_text)
        else:
            sub_chunks = [chunk[j:j + max_length - 2] for j in range(0, len(chunk), max_length - 2)]
            for sub_chunk in sub_chunks:
                sub_chunk_text = tokenizer.decode(sub_chunk, skip_special_tokens=True)
                chunks.append(sub_chunk_text)
    return chunks

def aggregate_embeddings(embeddings: List[torch.Tensor]) -> torch.Tensor:
    """Aggregate embeddings using mean pooling."""
    stacked_embeddings = torch.stack(embeddings, dim=0)
    aggregated_embedding = torch.mean(stacked_embeddings, dim=0)
    return F.normalize(aggregated_embedding, p=2, dim=0)

def document_to_vector(model, doc):
    """Convert a document into a vector by averaging its word vectors."""
    doc = doc.split()
    word_vectors = [model[word] for word in doc if word in model]
    if not word_vectors:  # Handle the case where none of the words are in the model
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)

# -----------------
# Embedding Functions
# -----------------

def get_embeddings(input: List[str], tokenizer: Any, model: Any, max_length: int, pooling_function: Any) -> torch.Tensor:
    """Get embeddings for a list of documents."""
    document_embeddings = []
    for document in input:
        chunks = split_into_chunks(document, tokenizer, max_length)
        chunk_embeddings = []
        for chunk in chunks:
            encoded_input = tokenizer(chunk, padding=True, truncation=True, return_tensors='pt', max_length=max_length)
            with torch.no_grad():
                model_output = model(**encoded_input)
            chunk_embedding = pooling_function(model_output, encoded_input['attention_mask']).squeeze()
            chunk_embeddings.append(chunk_embedding)
        document_embedding = aggregate_embeddings(chunk_embeddings)
        document_embeddings.append(document_embedding)
    return torch.stack(document_embeddings)

def batch_embeddings(input: List[str], tokenizer: Any, model: Any, pooling_function: Any, batch_size: int = 32, save_path: str = "", max_length: int = 512) -> torch.Tensor:
    """Get embeddings for a list of documents in batches."""
    embeddings = []
    pbar = tqdm(range(0, len(input), batch_size))
    for i in pbar:
        batch = input[i: i + batch_size]
        batch_embeddings = get_embeddings(batch, tokenizer, model, max_length, pooling_function)
        pbar.set_description(f"Processing batch: {batch[0][:20]}...")
        embeddings.append(batch_embeddings)
        if save_path:
            torch.save(torch.cat(embeddings, dim=0), save_path)
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings


def get_word2vec_embeddings(input: List[List[str]], model: Word2Vec) -> torch.Tensor:
    """Get Word2Vec embeddings for a list of documents."""
    document_embeddings = [document_to_vector(model, doc) for doc in input]
    document_embeddings = np.array(document_embeddings)
    return torch.tensor(document_embeddings)


def batch_word2vec_embeddings(input: List[List[str]], model: Word2Vec, batch_size: int = 32, save_path: str = "") -> torch.Tensor:
    """Get Word2Vec embeddings for a list of documents in batches."""
    embeddings = []
    pbar = tqdm(range(0, len(input), batch_size))
    for i in pbar:
        batch = input[i: i + batch_size]
        batch_embeddings = get_word2vec_embeddings(batch, model)
        pbar.set_description(f"Processing batch: {' '.join(batch[0][:3])}...")
        embeddings.append(batch_embeddings)
        if save_path:
            torch.save(torch.cat(embeddings, dim=0), save_path)
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings

# -----------------
# Wrapper Function
# -----------------

def get_model_and_pooling_func(model_name: str) -> Tuple[Any, Any, int]:
    """Return the appropriate tokenizer, model, mean pooling function, and max length based on the model name."""
    if model_name == "nomic":
        tokenizer, model, max_length = initialize_nomic_model()
    elif model_name == "minilm":
        model_id = 'sentence-transformers/all-MiniLM-L6-v2'
        tokenizer, model, max_length = initialize_sentence_transformer_model(model_id)
    elif model_name == "mpnet":
        model_id = 'sentence-transformers/all-mpnet-base-v2'
        tokenizer, model, max_length = initialize_sentence_transformer_model(model_id) 
    elif model_name == "bert":
        tokenizer, model, max_length = initialize_google_bert_model()
    elif model_name == "specter":
        tokenizer, model, max_length = initialize_specter_base_model()
    # elif model_name == "llama":
    #     tokenizer, model, max_length = initialize_llama_model()
    elif model_name == "word2vec":
        model = initialize_word2vec_model()
        return None, model, None  # pyright: ignore Word2Vec doesn't use tokenizer and max_length
    else:
        raise ValueError(f"Model name {model_name} is not supported.")
    
    return tokenizer, model, max_length

def process_embeddings(input: List[Any], model_name: str, pooling_strategy: str = "mean_pooling", batch_size: int = 32, save_path: str = "") -> torch.Tensor:
    """Process embeddings based on the specified model name and pooling strategy."""
    tokenizer, model, max_length = get_model_and_pooling_func(model_name)
    pooling_strategies = {
        'mean_pooling': mean_pooling,
        'sum_pooling': sum_pooling,
        'max_pooling': max_pooling,
        'min_pooling': min_pooling,
        'self_attention': self_attention,
        'global_attention': global_attention,
        'content_based_attention': content_based_attention,
        'learned_self_attention': learned_self_attention
    }
    pooling_function = pooling_strategies[pooling_strategy]

    if model_name == "word2vec":
        embeddings = batch_word2vec_embeddings(input, model, batch_size, save_path)
    else:
        embeddings = batch_embeddings(input, tokenizer, model, pooling_function, batch_size, save_path, max_length)
    return embeddings


if __name__ == "__main__":
    # strings = ["Hello, my name is Erik.", "What is that song called?", "Tell me the name of that song.", "What year was that song made?"]
    # tokenized_strings = [string.lower().split() for string in strings]

    # # Process embeddings with different pooling strategies
    # nomic_embeddings = process_embeddings(strings, "nomic", "mean_pooling")
    # print("Nomic Model with Mean Pooling Embeddings:")
    # print(nomic_embeddings.size())
    # print(nomic_embeddings)

    # st_embeddings = process_embeddings(strings, "minilm", "self_attention")
    # print("Sentence Transformer (MiniLM) with Self-Attention Embeddings:")
    # print(st_embeddings.size())
    # print(st_embeddings)

    # bert_embeddings = process_embeddings(strings, "bert", "global_attention")
    # print("Google BERT Model with Global Attention Embeddings:")
    # print(bert_embeddings.size())
    # print(bert_embeddings)

    # specter_embeddings = process_embeddings(strings, "specter", "content_based_attention")
    # print("AllenAI Specter Model with Content-Based Attention Embeddings:")
    # print(specter_embeddings.size())
    # print(specter_embeddings)

    # word2vec_embeddings = process_embeddings(tokenized_strings, "word2vec", "mean_pooling")
    # print("Word2Vec Model Embeddings:")
    # print(word2vec_embeddings.size())
    # print(word2vec_embeddings)

    import ast
    import pickle
    import pandas as pd

    def save_to_pickle(object, filename: str):
        assert filename[-7:] == ".pickle", "Filename must end with .pickle extension."
        pickle.dump(object, open(filename, 'wb'))

    def load_data(filepath, n=None):
        assert filepath[-4:] == ".csv", "Must be a .csv file"
        data = pd.read_csv(filepath)
        if n:
            data = data.head(n)

        attrs = {
            "titles": data["title"].tolist(),
            "text": data["text"].tolist(),
            "tags": data["tags"].apply(ast.literal_eval).tolist(),
            "ids": data.index.tolist()
        }

        if "simplified_tags" in data.columns:
            attrs["simplified_tags"] = data["simplified_tags"].apply(ast.literal_eval).tolist()

        return attrs

    def run(dataset_name: str, input_texts: List[str]):
        # embedding_models = ["nomic", "minilm", "mpnet", "bert", "specter", "word2vec"]
        embedding_models = ["word2vec", "nomic", "minilm", "mpnet", "bert", "specter"]
        pooling_strategies = [
            'mean_pooling', 'global_attention', 'content_based_attention'
        ]
        # pooling_strategies = [
        #     'mean_pooling', 'sum_pooling', 'max_pooling', 'min_pooling',
        #     'global_attention', 'content_based_attention'
        # ]
        # pooling_strategies = [
        #     'mean_pooling', 'sum_pooling', 'max_pooling', 'min_pooling',
        #     'self_attention', 'global_attention', 'content_based_attention', 'learned_self_attention'
        # ]

        for model_name in embedding_models:
            for pooling_strategy in pooling_strategies:
                print(f"Processing {model_name} with {pooling_strategy} pooling")
                embeddings = process_embeddings(input_texts, model_name, pooling_strategy)

                save_to_pickle(embeddings, f"embeddings/{dataset_name}_{model_name}_{pooling_strategy}_n5000.pickle")

    data = load_data("data/medium_1k_tags_simplified.csv", n=5000)
    run("medium1k", data["text"])
