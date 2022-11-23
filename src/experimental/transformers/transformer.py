from src.experimental.supervised.preprocessing import process_stack_trace_column, word2vec
from src.experimental.supervised.classifiers import rfc, svm, mlp, mlp_transf
from src.experimental.supervised.classifiers_hyper_parameter import rfc_hyper_parameter_grid, svm_hyper_parameter_grid, \
    mlp_hyper_parameter_grid, rfc_hyper_parameter_random, svm_hyper_parameter_random, mlp_hyper_parameter_random
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import numpy as np

df_monkey_labeled = pd.read_csv('../../../data/supervised/monkey_selection.csv')
df_github_labeled = pd.read_csv('../../../data/supervised/github_selection.csv')
df = pd.read_csv('../../../data/supervised/monkey_processed.csv')


def transformer(dataframe):
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device='cpu')
    # Encode all sentences
    embedding = model.encode(dataframe)
    return embedding


start = time.time()
print('started')

# Preprocessing
print('preprocessing completed:', time.time() - start)

# Word-Embedding
v = TfidfVectorizer(use_idf=True)
tf_idf = v.fit_transform(df['Stack trace'])
# w2v = word2vec(df['Stack trace'])

# save to csv
transformer_result = transformer(df['Stack trace'].tolist())
# dataframe = pd.DataFrame(transformer_result)
# dataframe.to_csv('transformer_result.csv')

np.savetxt("GFG.csv" ,transformer_result ,delimiter =" ", fmt ='% s')

# read from csv
# transformer_result = pd.read_csv('transformer_result.csv')
# transformer_result = transformer_result.values.tolist()
# transformer_list = [list(row) for row in transformer_result.values]
# transformer_list[:,1:]

# print(transformer_list)

print('word-embedding completed:', time.time() - start)

test_size = 0.3  # 70:30 split
features = transformer_result
# features = w2v
labels = df['Exception name']
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size)
print('dataset splitting completed:', time.time() - start)

# Random Forest Classifier
# rfc(start, X_train, X_test, y_train, y_test)

# Support Vector Machine
# svm(start, X_train, X_test, y_train, y_test)

# Neural Network
# mlp(start, X_train, X_test, y_train, y_test, 500, 1000, 1)

#Neural Network for Transformer
mlp_transf(start, X_train, X_test, y_train, y_test, 500, 1000, 1)

print('completed:', time.time() - start)
