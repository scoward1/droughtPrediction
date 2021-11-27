from numpy.core.fromnumeric import mean,std
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split, RepeatedStratifiedKFold
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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
    knn = KNeighborsClassifier(n_neighbors = k_val)
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

def linReg_fun(features, dlevel):

    # split the data into training and testing data
    feat_train, feat_test, dlevel_train, dlevel_test = train_test_split(features, dlevel, test_size = 0.2)
    
    # determine the model to be used
    linReg = LinearRegression()
    linReg.fit(feat_train, dlevel_train)

    # run the model
    dlevel_predict = linReg.predict(feat_test)

    # compare results to actual
    results = pd.DataFrame({'Actual': dlevel_test, 'Predicted': dlevel_predict})

    print('Mean Absolute Error:', metrics.mean_absolute_error(dlevel_test, dlevel_predict))
    print('Mean Squared Error:', metrics.mean_squared_error(dlevel_test, dlevel_predict))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(dlevel_test, dlevel_predict)))


def lda(features, dlevel):
    #Reduced Dataset to be trained

        [set,drought] = make_classification(n_samples=dlevel.size,n_features=10,n_classes=4,n_clusters_per_class = 1)

    #Define LDA
        lda = LinearDiscriminantAnalysis()
    
    #Fit LDA
        lda.fit (set,drought)

    #Model Results

    #kfold
        crossval = RepeatedStratifiedKFold(n_splits=5,random_state = None)
        score = cross_val_score(lda, set, drought, scoring = 'accuracy', cv=crossval,n_jobs=1)
    
    #Print the Score

        print('Mean Accuracy(Standard Deviation): %.3f (%.3f)' % (mean(score), std(score)))

