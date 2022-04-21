from src.experimental.supervised.preprocessing import process_stack_trace_column, word2vec
import pandas as pd
import spacy
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

# Random Forest Classifier
split = 0.3 # 70:30 split
features = tf_idf
# features = w2v
labels = df['Exception name']
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=split)

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train) # Train model using training sets
print('training completed: ', time.time() - start)

print('\n---------- Test results ------------')
acc_score = accuracy_score(y_test, rfc.predict(X_test))
print("accuracy_score: ", round(acc_score*100, 5), "%", sep="")
rfc_score = rfc.score(X_test, y_test)
print("RandomForestClassifier.score(): ", round(rfc_score*100, 5), "%", sep="")
print('------------------------------------\n')

# Plotting
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

fig = plt.figure(figsize=(50, 50))
plot_tree(rfc.estimators_[0],
          class_names=labels,
          filled=True, impurity=True, rounded=True)

# from dtreeviz.trees import dtreeviz
# from sklearn import preprocessing
# label_encoder = preprocessing.LabelEncoder()
# label_encoder.fit(list(df["Exception name"].unique()))
#
# viz = dtreeviz(rfc.estimators_[0],
#                X_train,
#                y_train,
#                class_names=list(label_encoder.classes_),
#                title="1. estimator visualization")

print('completed:', time.time() - start)

