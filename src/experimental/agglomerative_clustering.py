from preprocessing import process_stack_trace_column
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

df_monkey = pd.read_csv('../../data/monkey_data_stack_trace.csv')
df_github = pd.read_csv('../../data/github_issues_stack_trace.csv')

start = time.time()
# Preprocessing
data = process_stack_trace_column(df_monkey, 'l')

# Word Embedding
v = TfidfVectorizer(use_idf=True)
vectors = v.fit_transform(data)
tfidf_matrix = pd.DataFrame(vectors.todense(), columns=v.vocabulary_)


