import pandas as pd
import re
from string import punctuation

# Pre Processing
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

# Word Embedding
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Clustering
from sklearn.cluster import KMeans
from sklearn import metrics

# Others
import time


def clean_text(words):
    r = re.compile(r'[\s{}]+'.format(re.escape(punctuation)))
    cleaned = []
    for w in words:
        strings = r.split(w)
        for s in strings:
            cleaned.append(s)

    return cleaned


def filter_text(words):
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
        return lemm
    elif mode == 's':
        s = PorterStemmer()
        stem = [s.stem(w) for w in words]
        return stem


def process_stack_trace_row(stack_trace, mode):
    if pd.isnull(stack_trace):
        return []

    # Tokenize Stack Trace
    words = word_tokenize(stack_trace)
    # Split words at punctuation
    cleaned = clean_text(words)
    # Filter out unnecessary characters
    filtered = filter_text(cleaned)

    return stem_text(filtered, mode)


def process_stack_trace_column(dataframe, mode):
    print("Pre Processing...")
    dataframe.dropna(inplace=True)
    tokenized = dataframe.iloc[:, -1].apply(word_tokenize)
    cleaned = tokenized.apply(clean_text)
    filtered = cleaned.apply(filter_text)
    dfa = []

    dataframe = filtered.apply(stem_text, mode=mode)
    dataframe.apply(lambda row: dfa.append(' '.join(row)))
    dfx = pd.DataFrame(dfa, columns=['StackTrace'])

    return dfx


def process_stack_trace(dataframe, stem_mode, process_mode, vector_mode):
    start = time.time()

    if process_mode == 'c':
        to_vec_dataframe = process_stack_trace_column(dataframe, stem_mode)
        dataframe = sklearn_vector(to_vec_dataframe, vector_mode)
    else:
        for cols, item in dataframe.iterrows():
            print(process_stack_trace_row(item.iloc[-1], stem_mode))  # Process Stack Trace
            # print(process_stack_trace_row(item['Stack trace'], stem_mode))  # Process Stack Trace
            # sklearn_vector(process_stack_trace_row(item.iloc[-1], stem_mode))

    kmeans_function(dataframe)

    print("Completed:", time.time() - start)


def sklearn_vector(dataframe, vector_mode):
    print("Word Embedding...")
    # call function to vectorize
    if vector_mode == 'wv':
        sklearn_dataframe = spacy_word_2_vec(dataframe)
        return sklearn_dataframe
    elif vector_mode == 'sv':
        sklearn_dataframe = sklearn_vector_vectorizer(dataframe)
        return sklearn_dataframe
    else:
        print("Wrong vector_mode!\nUse:\n  'sv'\n  'wv'")


def spacy_word_2_vec(dataframe):
    # print(dataframe)
    nlp = spacy.load('en_core_web_md')
    docs = dataframe['StackTrace'].apply(lambda x: nlp(x))
    pdv = []
    for index, value in docs.iteritems():
        pdv.append(value.vector)
        # print(value)
        # print(value.vector)

    # pdv = pd.DataFrame(pdv)

    return pdv


def sklearn_vector_vectorizer(dataframe):
    v = TfidfVectorizer()
    x = v.fit_transform(dataframe['StackTrace'])
    df = pd.DataFrame(x.toarray(), columns=v.get_feature_names_out())
    # print(df)

    return df


def kmeans_function(dataframe):
    print("Kmeans...")
    for num_clusters in range(2, 10):
        print("\nClusters:")
        print(num_clusters)

        km = KMeans(n_clusters=num_clusters, n_init=10, random_state=1)
        km.fit_predict(dataframe)

        print("Labels:")
        print(km.labels_)
        labels = km.labels_
        print("Centroids:")
        print(km.cluster_centers_)
        print("Score:")
        print(km.score(dataframe))
        print("Silhouette:")
        print(metrics.silhouette_score(dataframe, km.labels_, metric='euclidean'))




if __name__ == "__main__":

    df_comp = pd.read_csv('../../data/monkey_data_stack_trace_entry.csv')
    df_monkey = pd.read_csv('../../data/monkey_data_stack_trace.csv')
    df_github = pd.read_csv('../../data/github_issues_stack_trace.csv', encoding='utf-8')
    df_w3c = pd.read_csv('../../data/w3c_test_results_failed.csv')

    process_stack_trace(df_comp, stem_mode='l', process_mode='c', vector_mode='wv')
    # Modes:
    # stem_mode = 'l' || 's'
    # process_mode = 'c' || 'r' //'r' is not suitable for vector_mode
    # vector_mode = 'sv' || 'wv'
