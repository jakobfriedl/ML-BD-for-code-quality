from src.experimental.unsupervised import preprocessing
import pandas as pd
import numpy as np
import time
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

df_monkey = pd.read_csv('../../data/unsupervised/monkey_data_stack_trace.csv')
df_github = pd.read_csv('../../data/unsupervised/github_issues_stack_trace.csv')


def add_values_in_dict(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = list()
    dictionary[key].append(value)
    return dictionary


def spacy_word_2_vec(dataframe):
    nlp = spacy.load('en_core_web_md')
    docs = dataframe[:].apply(lambda x: nlp(x))
    pdv = []
    for index, value in docs.iteritems():
        pdv.append(value.vector)
    return pdv

print("Started Clustering Comparison!\n")
start = time.time()

# Preprocessing
data = preprocessing.process_stack_trace_column(df_monkey, 'l')
y = data.iloc[:].values

# Word Embedding
v = TfidfVectorizer(use_idf=True)
tf_idf_x = v.fit_transform(data)

w2v_x = spacy_word_2_vec(data)

km = KMeans(n_clusters=5, random_state=1)

# TF-IDF
tf_idf_model = km.fit(tf_idf_x)
tf_idf_result = tf_idf_model.predict(tf_idf_x)
tf_idf_dict = {}

# Word-to-Vec
w2v_model = km.fit(w2v_x)
w2v_result = w2v_model.predict(w2v_x)
w2v_dict = {}

for i in range(20):
    container = np.random.randint(low=0, high=3944, size=1)
    j = container[0]

    add_values_in_dict(tf_idf_dict, tf_idf_result[j], y[j]+"\n")
    add_values_in_dict(w2v_dict, w2v_result[j], y[j]+"\n")

for key in range(0, 5):
    print("Cluster:", key, "\n")
    print("TF-IDF cluster\n")
    if key in tf_idf_dict:
        print(*tf_idf_dict[key], sep="\n")
    print("W2V cluster\n")
    if key in w2v_dict:
        print(*w2v_dict[key], sep="\n")

print("elapsed time:", time.time() - start)