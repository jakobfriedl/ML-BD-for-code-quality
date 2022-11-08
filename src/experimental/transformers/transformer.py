from sentence_transformers import SentenceTransformer


def transformer(dataframe):
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    # Encode all sentences
    embedding = model.encode(dataframe)
    return embedding
