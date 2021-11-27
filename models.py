from numpy.core.fromnumeric import mean
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import numpy as np
import matplotlib.pyplot as plt

def knn_neighbors(features, dlevel):
    error = []

    feat_train, feat_test, dlevel_train, dlevel_test = train_test_split(features, dlevel, test_size = 0.2)

    # calculate the error for k-values (1-100)
    for neighbors in range(1, 100):
        knn = KNeighborsClassifier(n_neighbors = neighbors)
        knn.fit(feat_train, dlevel_train)
        dlevel_predict = knn.predict(feat_test)
        error.append(np.mean(dlevel_predict != dlevel_test))

    plt.figure()
    plt.plot(range(1, 100), error)
    plt.title("Error Rate of Each K-Value")
    plt.xlabel('K-Value')
    plt.ylabel('Mean Error')
    plt.show()

    return error

def knn_fun(features, dlevel, k_val):
    num_correct = 0

    # split the data into training and testing data
    feat_train, feat_test, dlevel_train, dlevel_test = train_test_split(features, dlevel, test_size = 0.2)

    # train the classifier
    knn = KNeighborsClassifier(n_neighbors = 100)
    knn.fit(feat_train, dlevel_train)

    # run the model to determine the accuracy
    dlevel_predict = knn.predict(feat_test)
    acc_array = dlevel_test.__eq__(dlevel_predict)

    for item in acc_array:
        if item == True:
            num_correct += 1
    
    accuracy = num_correct/(acc_array.size)

    return accuracy

def qda_fun(features, dlevel):
    
    # split into 5 groups for k-Fold analysis
    cv = KFold(n_splits = 5, shuffle = False, random_state = None)
    
    # determine the model to be used
    qda = QuadraticDiscriminantAnalysis()

    # run the model, get the accuracy
    accuracy = cross_val_score(qda, features, dlevel, scoring = 'accuracy', cv = cv)

    return accuracy