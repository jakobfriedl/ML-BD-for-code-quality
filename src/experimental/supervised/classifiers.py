from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def rfc(start_time, X_train, X_test, y_train, y_test, estimators=100, max_depth=None):
    rfc = RandomForestClassifier(n_estimators=estimators, max_depth=max_depth)
    rfc.fit(X_train, y_train) # Train model using training sets
    print('rfc training completed: ', time.time() - start_time)

    print('\n---------- Test results ------------')
    acc_score = accuracy_score(y_test, rfc.predict(X_test))
    print("accuracy_score: ", round(acc_score*100, 5), "%", sep="")
    rfc_score = rfc.score(X_test, y_test)
    print("RandomForestClassifier.score(): ", round(rfc_score*100, 5), "%", sep="")
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
    print("accuracy_score: ", round(acc_score*100, 5), "%", sep="")
    svm_score = svm.score(X_test, y_test)
    print("SVC.score(): ", round(svm_score*100, 5), "%", sep="")
    print('------------------------------------\n')