import pandas as pd
import src.embeddings as em


def combine_columns(df, columns):
    combined_texts = df[columns].apply(lambda row: ' '.join(row), axis=1).tolist()
    return combined_texts


def embeddings_from_csv(model_id, path_to_csv, csv_columns, save_path, num_obs=5000):
    df = pd.read_csv(path_to_csv)
    if num_obs != None:
        df = df.loc[:num_obs-1]
    texts = combine_columns(df, csv_columns)
    tokenizer, model = em.initialize_embedding_model(model_id)
    em.batch_embeddings(texts, tokenizer, model, batch_size=50, save_path=save_path)

