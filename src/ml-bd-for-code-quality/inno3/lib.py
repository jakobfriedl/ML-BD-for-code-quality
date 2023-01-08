import re
from src.experimental.supervised.classifiers import rfc, svm, mlp
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from experimental.transformers.special_characters import characters
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer


# Preprocessing
def replace_special_characters(words):
    return [characters[w] if w in characters else w for w in words]


def clean_text(words):
    # Split words at punctuation
    r = re.compile(r'[\s{}]+'.format(re.escape(punctuation)))
    cleaned = []
    for w in words:
        strings = r.split(w)
        for s in strings:
            cleaned.append(s)

    return cleaned


def filter_text(words):
    # Filter out unnecessary characters
    stop_words = set(stopwords.words('english'))

    return [w.lower() for w in words            # Convert to Lowercase
            if w not in stop_words              # Remove Stop words
            and w not in set(punctuation)       # Remove Special Characters
            and not w.isdigit()                 # Remove Numbers
            and w != '']                        # Remove Whitespaces


def stem_text(words, mode):
    if mode == 'l':
        l = WordNetLemmatizer()
        lemm = [l.lemmatize(w) for w in words]
        return ' '.join(x for x in lemm if x != '\n')
    elif mode == 's':
        s = PorterStemmer()
        stem = [s.stem(w) for w in words]
        return ' '.join(x for x in stem if x != '\n')
    else:
        print("Invalid mode.")
        return


def process_stack_trace_column(dataframe, data_col, replace_special_chars=False, stem_mode='l'):
    print("Preprocessing started")
    dataframe.dropna(inplace=True)
    tokenized = dataframe[data_col].apply(word_tokenize)
    if replace_special_chars:
        replaced = tokenized.apply(replace_special_characters)
        cleaned = replaced.apply(clean_text)
    else:
        cleaned = tokenized.apply(clean_text)
    filtered = cleaned.apply(filter_text)

    dataframe[data_col] = filtered.apply(stem_text, mode=stem_mode)
    print("Preprocessing finished")
    return dataframe

## Word Embedding
def word_embedding(df, data_col):
    v = TfidfVectorizer(use_idf=True)
    return v.fit_transform(df[data_col])

## Transformer
def transformer(dataframe):
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')
    # Encode all sentences
    embedding = model.encode(dataframe)
    return embedding

## KMeans
def k_means_gui(tf_idf, n_clusters, random_state):
    km = KMeans(n_clusters, random_state=random_state)
    model = km.fit_predict(tf_idf)
    score = silhouette_score(tf_idf, model)
    return score

## Random Forest Classifier 
def rfc_gui(features, labels, test_size, estimators, max_depth):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size)
    rfc = RandomForestClassifier(n_estimators=estimators, max_depth=max_depth)
    rfc.fit(X_train, y_train)  # Train model using training sets
    acc_score = accuracy_score(y_test, rfc.predict(X_test))
    rfc_score = rfc.score(X_test, y_test)
    return(acc_score, rfc_score)


def rfc_transformer_gui(df_processed, data_col, labels, test_size, estimators, max_depth):
    transformer_result = transformer(df_processed[data_col].tolist())
    X_train, X_test, y_train, y_test = train_test_split(transformer_result, labels, test_size=test_size)
    rfc = RandomForestClassifier(n_estimators=estimators, max_depth=max_depth)
    rfc.fit(X_train, y_train)  # Train model using training sets
    acc_score = accuracy_score(y_test, rfc.predict(X_test))
    rfc_score = rfc.score(X_test, y_test)
    return (acc_score, rfc_score)

## Support Vector Machine
def svm_gui(features, labels, test_size, kernel):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size)
    svm = SVC(kernel=kernel)
    svm.fit(X_train, y_train)
    acc_score = accuracy_score(y_test, svm.predict(X_test))
    svm_score = svm.score(X_test, y_test)
    return(acc_score, svm_score)


def svm_transformer_gui(df_processed, data_col, labels, test_size, kernel):
    transformer_result = transformer(df_processed[data_col].tolist())
    X_train, X_test, y_train, y_test = train_test_split(transformer_result, labels, test_size=test_size)
    svm = SVC(kernel=kernel)
    svm.fit(X_train, y_train)
    acc_score = accuracy_score(y_test, svm.predict(X_test))
    svm_score = svm.score(X_test, y_test)
    return (acc_score, svm_score)

## Neural Network# rfc(start, X_train, X_test, y_train, y_test)
def mlp_gui(features, labels, test_size, pca_components, neurons, hidden_layer):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size)

    # Data scale to have 0 mean and 1 variance
    scaler = MaxAbsScaler()
    X_train = scaler.fit_transform(X_train)

    pca = PCA(n_components=pca_components)

    # Applying the dimensionality reduction
    # toarray() to convert to a dense numpy array
    pca.fit(X_train.toarray())
    X_train = pca.fit_transform(X_train.toarray())
    pca.fit(X_test.toarray())
    X_test = pca.transform(X_test.toarray())

    mlp = (MLPClassifier(hidden_layer_sizes=(neurons, hidden_layer), max_iter=1000,
                         learning_rate_init=0.001))
    mlp.fit(X_train, y_train)

    X_pred = mlp.predict(X_test)

    acc_score = accuracy_score(X_pred, y_test)
    return (acc_score, X_pred)


def mlp_transformer_gui(df_processed, data_col, labels, test_size, pca_components, neurons, hidden_layer):
    transformer_result = transformer(df_processed[data_col].tolist())

    pca_components = 200
    X_train, X_test, y_train, y_test = train_test_split(transformer_result, labels, test_size=test_size)

    scaler = MaxAbsScaler()
    X_train = scaler.fit_transform(X_train)

    pca = PCA(n_components=pca_components)

    # Applying the dimensionality reduction
    # toarray() to convert to a dense numpy array
    pca.fit(X_train)
    X_train = pca.fit_transform(X_train)
    pca.fit(X_test)
    X_test = pca.transform(X_test)

    mlp = (MLPClassifier(hidden_layer_sizes=(neurons, hidden_layer), max_iter=1000,
                         learning_rate_init=0.001))
    mlp.fit(X_train, y_train)

    X_pred = mlp.predict(X_test)

    # Calculate and display accuracy
    acc_score = accuracy_score(X_pred, y_test)
    return (acc_score, X_pred)