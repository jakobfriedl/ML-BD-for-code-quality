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
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier



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

## KMeans
def k_means_gui(tf_idf, n_clusters, random_state):

    print(n_clusters, random_state)

    km = KMeans(n_clusters, random_state=random_state)
    model = km.fit(tf_idf)
    return model.predict(tf_idf)

## Random Forest Classifier 
def rfc_gui(features, labels, test_size, estimators, max_depth):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size)
    rfc = RandomForestClassifier(n_estimators=estimators, max_depth=max_depth)
    rfc.fit(X_train, y_train)  # Train model using training sets
    acc_score = accuracy_score(y_test, rfc.predict(X_test))
    rfc_score = rfc.score(X_test, y_test)
    return(acc_score, rfc_score)


def rfc_gui_transformer(features, labels, test_size, estimators, max_depth):
    print("TODO")

## Support Vector Machine
def svm_gui(features, labels, test_size, kernel):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size)
    svm = SVC(kernel=kernel)
    svm.fit(X_train, y_train)
    acc_score = accuracy_score(y_test, svm.predict(X_test))
    svm_score = svm.score(X_test, y_test)
    return(acc_score, svm_score)


def svm_gui_transformer(features, labels, test_size, kernel):
    print("TODO")

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

    # PCA Plots
    # 2D Plot
    if pca.n_components == 2:
        Xax = X_train[:, 0]
        Yax = X_train[:, 1]
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor('white')
        for l in np.unique(y_train):
            ix = np.where(y_train == l)
            ax.scatter(Xax[ix], Yax[ix])

        plt.xlabel("First Principal Component", fontsize=14)
        plt.ylabel("Second Principal Component", fontsize=14)
        plt.legend()
        plt.show()

    # 3D plot
    if pca.n_components == 3:
        Xax = X_train[:, 0]
        Yax = X_train[:, 1]
        Zax = X_train[:, 2]
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection='3d')

        fig.patch.set_facecolor('white')
        for l in np.unique(y_train):
            ix = np.where(y_train == l)
            ax.scatter(Xax[ix], Yax[ix], Zax[ix])

        ax.set_xlabel("First Principal Component", fontsize=14)
        ax.set_ylabel("Second Principal Component", fontsize=14)
        ax.set_zlabel("Third Principal Component", fontsize=14)
        ax.legend()
        plt.show()

    # Plotting the evaluation of the number of components
    plt.rcParams["figure.figsize"] = (18, 6)
    fig, ax = plt.subplots()
    arange_limit = pca.n_components + 1
    xi = np.arange(1, arange_limit, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)
    plt.ylim(0.0, 1.1)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')

    plt.xlabel('Number of Components')
    plt.xticks(np.arange(0, pca.n_components + 5, step=pca.n_components / 10))
    plt.ylabel('Cumulative variance (%)')
    plt.title('The number of components needed to explain variance')

    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.text(0.5, 0.85, '95% cut-off threshold', color='red', fontsize=16)

    ax.grid(axis='x')
    plt.show()

    mlp = (MLPClassifier(hidden_layer_sizes=(neurons, hidden_layer), max_iter=1000,
                         learning_rate_init=0.001))
    mlp.fit(X_train, y_train)

    X_pred = mlp.predict(X_test)

    acc_score = accuracy_score(X_pred, y_test)
    return (acc_score, X_pred)


def mlp_gui_transformer(features, labels, test_size, pca_components, neurons, hidden_layer):
    print("TODO")


