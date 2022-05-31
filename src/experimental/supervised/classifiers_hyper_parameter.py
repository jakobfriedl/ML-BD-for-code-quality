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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np


def rfc_hyper_parameter(start_time, X_train, X_test, y_train, y_test):
    print()
    print('======================= RANDOM FOREST CLASSIFIER =======================')
    print()

    # tuned parameters
    param_grid = [
        {'n_estimators': [50, 75, 100, 200], 'max_depth': [80, 100, 120, None]},
    ]

    rfc_hyper = GridSearchCV(RandomForestClassifier(), param_grid)
    rfc_hyper.fit(X_train, y_train)  # Train model using training sets
    print('rfc training completed: ', time.time() - start_time)

    print("Best parameters set found on:")
    print()
    print(rfc_hyper.best_params_)
    print()
    print("Grid scores:")
    print()
    means = rfc_hyper.cv_results_["mean_test_score"]
    stds = rfc_hyper.cv_results_["std_test_score"]
    for mean, std, params in zip(means, stds, rfc_hyper.cv_results_["params"]):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

    print('----------------------- Test results -----------------------')
    acc_score = accuracy_score(y_test, rfc_hyper.predict(X_test))
    print("accuracy_score: ", round(acc_score * 100, 5), "%", sep="")

    print()
    print('---------------------- Detailed Report ----------------------')
    y_true, y_pred = y_test, rfc_hyper.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()


def svm_hyper_parameter(start_time, X_train, X_test, y_train, y_test):
    print()
    print('======================= SUPPORT VECTOR MACHINE =======================')
    print()

    # tuned parameters
    param_grid = [
        {'C': [1, 10, 50, 100, 500], 'kernel': ['linear']},
        {'C': [1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]

    svm_hyper = GridSearchCV(SVC(), param_grid)
    svm_hyper.fit(X_train, y_train)
    print('svm training completed: ', time.time() - start_time)

    print("Best parameters set found on:")
    print()
    print(svm_hyper.best_params_)
    print()
    print("Grid scores:")
    print()
    means = svm_hyper.cv_results_["mean_test_score"]
    stds = svm_hyper.cv_results_["std_test_score"]
    for mean, std, params in zip(means, stds, svm_hyper.cv_results_["params"]):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

    print('----------------------- Test results -----------------------')
    acc_score = accuracy_score(y_test, svm_hyper.predict(X_test))
    print("accuracy_score: ", round(acc_score * 100, 5), "%", sep="")

    print()
    print('---------------------- Detailed Report ----------------------')
    y_true, y_pred = y_test, svm_hyper.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()


def mlp_hyper_parameter(start_time, X_train, X_test, y_train, y_test):
    print()
    print('======================= NEURAL NETWORK =======================')
    print()

    # tuned parameters
    param_grid = [
        {'hidden_layer_sizes': [10, 100, 1000], 'max_iter': [1000, 1500], 'learning_rate_init': [0.001]},
    ]

    # Data scale to have 0 mean and 1 variance
    # scaler = MaxAbsScaler()
    # X_train = scaler.fit_transform(X_train)
    #
    # pca = PCA(n_components=500)
    #
    # # Applying the dimensionality reduction
    # # toarray() to convert to a dense numpy array
    # pca.fit(X_train.toarray())
    # X_train = pca.fit_transform(X_train.toarray())
    # pca.fit(X_test.toarray())
    # X_test = pca.transform(X_test.toarray())

    mlp_hyper = GridSearchCV(MLPClassifier(), param_grid)

    mlp_hyper.fit(X_train, y_train)

    print('mlp training completed: ', time.time() - start_time)

    print("Best parameters set found on:")
    print()
    print(mlp_hyper.best_params_)
    print()
    print("Grid scores:")
    print()
    means = mlp_hyper.cv_results_["mean_test_score"]
    stds = mlp_hyper.cv_results_["std_test_score"]
    for mean, std, params in zip(means, stds, mlp_hyper.cv_results_["params"]):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

    print('----------------------- Test results -----------------------')
    acc_score = accuracy_score(y_test, mlp_hyper.predict(X_test))
    print("accuracy_score: ", round(acc_score * 100, 5), "%", sep="")

    print()
    print('---------------------- Detailed Report ----------------------')
    y_true, y_pred = y_test, mlp_hyper.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

