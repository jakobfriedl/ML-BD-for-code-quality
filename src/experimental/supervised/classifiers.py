from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import MaxAbsScaler
import numpy as np


def rfc(start_time, X_train, X_test, y_train, y_test, estimators=100, max_depth=None):
    rfc = RandomForestClassifier(n_estimators=estimators, max_depth=max_depth)
    rfc.fit(X_train, y_train)  # Train model using training sets
    print('rfc training completed: ', time.time() - start_time)

    print('\n---------- Test results ------------')
    acc_score = accuracy_score(y_test, rfc.predict(X_test))
    print("accuracy_score: ", round(acc_score * 100, 5), "%", sep="")
    rfc_score = rfc.score(X_test, y_test)
    print("RandomForestClassifier.score(): ", round(rfc_score * 100, 5), "%", sep="")
    print('------------------------------------\n')

    # Plotting
    fig = plt.figure(figsize=(50, 50))
    plot_tree(rfc.estimators_[0],
              # class_names=labels,
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


def svm(start_time, X_train, X_test, y_train, y_test):
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    print('svm training completed: ', time.time() - start_time)

    print('\n---------- Test results ------------')
    acc_score = accuracy_score(y_test, svm.predict(X_test))
    print("accuracy_score: ", round(acc_score * 100, 5), "%", sep="")
    svm_score = svm.score(X_test, y_test)
    print("SVC.score(): ", round(svm_score * 100, 5), "%", sep="")
    print('------------------------------------\n')


def mlp(start, X_train, X_test, y_train, y_test, pca_components, neurons, hidden_layer):
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

    print('PCA completed: ', time.time() - start)
    print('\n---------- Test results ------------')
    if pca.n_components >= 450:
        print("As the plot presents, to get 95% of variance explained 450 principal components are needed")

    mlp = (MLPClassifier(hidden_layer_sizes=(neurons, hidden_layer), max_iter=1000,
                         learning_rate_init=0.001))
    mlp.fit(X_train, y_train)

    X_pred = mlp.predict(X_test)

    # Calculate and display accuracy
    acc = accuracy_score(X_pred, y_test)
    print("Accuracy score: ", round(acc, 3))
    print('------------------------------------\n')

def mlp_transf(start, X_train, X_test, y_train, y_test, pca_components, neurons, hidden_layer):
    # Data scale to have 0 mean and 1 variance
    scaler = MaxAbsScaler()
    X_train = scaler.fit_transform(X_train)

    pca = PCA(n_components=pca_components)

    # Applying the dimensionality reduction
    # toarray() to convert to a dense numpy array
    pca.fit(X_train)
    X_train = pca.fit_transform(X_train)
    pca.fit(X_test)
    X_test = pca.transform(X_test)

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

    print('PCA completed: ', time.time() - start)
    print('\n---------- Test results ------------')
    if pca.n_components >= 450:
        print("As the plot presents, to get 95% of variance explained 450 principal components are needed")

    mlp = (MLPClassifier(hidden_layer_sizes=(neurons, hidden_layer), max_iter=1000,
                         learning_rate_init=0.001))
    mlp.fit(X_train, y_train)

    X_pred = mlp.predict(X_test)

    # Calculate and display accuracy
    acc = accuracy_score(X_pred, y_test)
    print("Accuracy score: ", round(acc, 3))
    print('------------------------------------\n')