from src.experimental.supervised.preprocessing import process_stack_trace_column, word2vec
from src.experimental.supervised.classifiers import rfc, svm, mlp
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

df_monkey_labeled = pd.read_csv('../../data/supervised/monkey_selection.csv')
df_github_labeled = pd.read_csv('../../data/supervised/github_selection.csv')
df = pd.read_csv('../../data/supervised/monkey_processed.csv')

start = time.time()
print('started')

# Preprocessing
# df = process_stack_trace_column(df_monkey_labeled, mode="l")
# df.to_csv('../../data/supervised/monkey_processed.csv')
print('preprocessing completed:', time.time() - start)

# Word-Embedding
v = TfidfVectorizer(use_idf=True)
tf_idf = v.fit_transform(df['Stack trace'])
# w2v = word2vec(df['Stack trace'])
print('word-embedding completed:', time.time() - start)

test_size = 0.3  # 70:30 split
features = tf_idf
# features = w2v
labels = df['Exception name']
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size)
print('dataset splitting completed:', time.time() - start)

# Random Forest Classifier
# rfc(start, X_train, X_test, y_train, y_test)

# Support Vector Machine
# svm(start, X_train, X_test, y_train, y_test)

# Nerual Network
mlp(start, X_train, X_test, y_train, y_test, 500, 1000, 1)

print('completed:', time.time() - start)
