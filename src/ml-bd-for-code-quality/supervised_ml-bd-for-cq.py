import pandas as pd
import spacy
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from src.experimental.supervised.preprocessing import process_stack_trace_column, word2vec

df_monkey_labeled = pd.read_csv('../../data/supervised/monkey_selection.csv')
df_github_labeled = pd.read_csv('../../data/supervised/github_selection.csv')

start = time.time()

# Preprocessing
df = process_stack_trace_column(df_monkey_labeled, mode="l")
print('preprocessing completed:', time.time() - start)

# Word-Embedding
v = TfidfVectorizer(use_idf=True)
tf_idf = v.fit_transform(df['Stack trace'])
# w2v = word2vec(df['Stack trace'])
print('word-embedding completed:', time.time() - start)

print('completed:', time.time() - start)




