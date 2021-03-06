from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report
import numpy as np


def rfc_hyper_parameter_grid(start_time, X_train, X_test, y_train, y_test):
    printHeading('RANDOM FOREST CLASSIFIER');

    # tuned parameters
    param_grid = [
        {'n_estimators': [50, 75, 100, 200], 'max_depth': [80, 100, 120, None]},
    ]

    rfc_hyper = GridSearchCV(RandomForestClassifier(), param_grid)
    rfc_hyper.fit(X_train, y_train)  # Train model using training sets
    print('rfc training completed: ', time.time() - start_time)

    printBestParams(rfc_hyper)
    printTestResult(y_test, rfc_hyper, X_test)
    printDetailedResult(y_test, rfc_hyper, X_test)


def rfc_hyper_parameter_random(start_time, X_train, X_test, y_train, y_test):
    printHeading('RANDOM FOREST CLASSIFIER');

    # tuned parameters
    n_estimators = [int(x) for x in np.linspace(start=10, stop=200, num=20)]
    max_depth = [80, 100, 120, None]

    param_grid = [
        {'n_estimators': n_estimators, 'max_depth': max_depth},
    ]

    rfc_hyper = RandomizedSearchCV(RandomForestClassifier(), param_grid, n_iter=50)
    rfc_hyper.fit(X_train, y_train)  # Train model using training sets
    print('rfc training completed: ', time.time() - start_time)

    printBestParams(rfc_hyper)
    printTestResult(y_test, rfc_hyper, X_test)
    printDetailedResult(y_test, rfc_hyper, X_test)

def svm_hyper_parameter_grid(start_time, X_train, X_test, y_train, y_test):
    printHeading('SUPPORT VECTOR MACHINE')

    # tuned parameters
    param_grid = [
        {'C': [1, 10, 50, 100, 500], 'kernel': ['linear']},
        {'C': [1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]

    svm_hyper = GridSearchCV(SVC(), param_grid)
    svm_hyper.fit(X_train, y_train)
    print('svm training completed: ', time.time() - start_time)

    printBestParams(svm_hyper)
    printTestResult(y_test, svm_hyper, X_test)
    printDetailedResult(y_test, svm_hyper, X_test)


def svm_hyper_parameter_random(start_time, X_train, X_test, y_train, y_test):
    printHeading('SUPPORT VECTOR MACHINE')

    C = [int(x) for x in np.linspace(start=10, stop=500, num=50)]
    kernel = ['linear']

    # tuned parameters
    param_grid = [
        {'C': C, 'kernel': kernel},
    ]

    svm_hyper = RandomizedSearchCV(SVC(), param_grid, n_iter=20)
    svm_hyper.fit(X_train, y_train)
    print('svm training completed: ', time.time() - start_time)

    printBestParams(svm_hyper)
    printTestResult(y_test, svm_hyper, X_test)
    printDetailedResult(y_test, svm_hyper, X_test)


def mlp_hyper_parameter_grid(start_time, X_train, X_test, y_train, y_test):
    printHeading('NEURAL NETWORK')

    # tuned parameters
    param_grid = [
        {'hidden_layer_sizes': [(100, 1), (100, 2)], 'max_iter': [500, 1000, 1500], 'learning_rate_init': [0.001]},
    ]

    # Data scale to have 0 mean and 1 variance
    scaler = MaxAbsScaler()
    X_train = scaler.fit_transform(X_train)

    pca = PCA(n_components=500)

    # Applying the dimensionality reduction
    # toarray() to convert to a dense numpy array
    pca.fit(X_train.toarray())
    X_train = pca.fit_transform(X_train.toarray())
    pca.fit(X_test.toarray())
    X_test = pca.transform(X_test.toarray())

    mlp_hyper = GridSearchCV(MLPClassifier(), param_grid)

    mlp_hyper.fit(X_train, y_train)

    print('mlp training completed: ', time.time() - start_time)

    printBestParams(mlp_hyper)
    printTestResult(y_test, mlp_hyper, X_test)
    printDetailedResult(y_test, mlp_hyper, X_test)


def mlp_hyper_parameter_random(start_time, X_train, X_test, y_train, y_test):
    printHeading('NEURAL NETWORK');

    # tuned parameters
    hidden_layer_sizes = [(100, 1)]
    max_iter = [int(x) for x in np.linspace(start=1000, stop=5000, num=1000)]
    learning_rate_init = [0.001, 0.0001, 0.00001]

    param_grid = [
        {'hidden_layer_sizes': hidden_layer_sizes, 'max_iter': max_iter, 'learning_rate_init': learning_rate_init},
    ]

    # Data scale to have 0 mean and 1 variance
    scaler = MaxAbsScaler()
    X_train = scaler.fit_transform(X_train)

    pca = PCA(n_components=500)

    # Applying the dimensionality reduction
    # toarray() to convert to a dense numpy array
    pca.fit(X_train.toarray())
    X_train = pca.fit_transform(X_train.toarray())
    pca.fit(X_test.toarray())
    X_test = pca.transform(X_test.toarray())

    mlp_hyper = RandomizedSearchCV(MLPClassifier(), param_grid, n_iter=2)

    mlp_hyper.fit(X_train, y_train)

    print('mlp training completed: ', time.time() - start_time)

    printBestParams(mlp_hyper)
    printTestResult(y_test, mlp_hyper, X_test)
    printDetailedResult(y_test, mlp_hyper, X_test)


def printHeading(heading):
    print('\n======================= ', heading, ' =======================\n')


def printBestParams(result):
    print("\nBest parameters set found on:\n")
    print(result.best_params_, '\n')

    print("Grid scores:\n")
    means = result.cv_results_["mean_test_score"]
    stds = result.cv_results_["std_test_score"]
    for mean, std, params in zip(means, stds, result.cv_results_["params"]):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()


def printTestResult(y_test, result, X_test):
    print('\n----------------------- Test results -----------------------')
    acc_score = accuracy_score(y_test, result.predict(X_test))
    print("accuracy_score: ", round(acc_score * 100, 5), "%\n", sep="")


def printDetailedResult(y_test, result, X_test):
    print('\n---------------------- Detailed Report ----------------------')
    y_true, y_pred = y_test, result.predict(X_test)
    print(classification_report(y_true, y_pred), '\n')
