from numpy.core.fromnumeric import mean,std
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn import metrics, svm
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from numpy import sum
import matplotlib.pyplot as plt
import pandas as pd


def knn_neighbors(features, dlevel):
    """
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
    """
    # create new a knn model
    knn = KNeighborsClassifier()
    # create a dictionary of all values we want to test for n_neighbors
    param_grid = {'n_neighbors': np.arange(1, 500)}
    # use gridsearch to test all values for n_neighbors
    knn_grid = GridSearchCV(knn, param_grid, cv=5)
    # fit model to data
    knn_grid.fit(features, dlevel)
    # check top performing n_neighbors value
    print(knn_grid.best_params_)
    # check mean score for the top performing value of n_neighbors
    print(knn_grid.best_score_)


def knn_fun(features, dlevel, k_val):

    # split the data into training and testing data
    feat_train, feat_test, dlevel_train, dlevel_test = train_test_split(features, dlevel, test_size = 0.2)
    
    # determine the model to be used
    knn = KNeighborsClassifier()
    knn.fit(feat_train, dlevel_train)

    dlevel_predict = knn.predict(feat_test)
    compare = dlevel_test.__eq__(dlevel_predict)

    correct = 1
    for item in compare:
        if item == True:
            correct = correct + 1
    
    accuracy = (sum(compare))/(len(compare))
    print('\nKNN accuracy: %.3f' % accuracy)


def knnReg_fun(features, dlevel, k_val):
    
    # split the data into training and testing data
    feat_train, feat_test, dlevel_train, dlevel_test = train_test_split(features, dlevel, test_size = 0.2)
    
    # determine the model to be used
    knnReg = KNeighborsRegressor()
    knnReg.fit(feat_train, dlevel_train)
    
    # run the model
    dlevel_predict = knnReg.predict(feat_test)

    # compare results to actual
    print('\nknn regression')
    print('r2:', metrics.r2_score(dlevel_test, dlevel_predict))
    print('Mean Absolute Error:', metrics.mean_absolute_error(dlevel_test, dlevel_predict))
    print('Mean Squared Error:', metrics.mean_squared_error(dlevel_test, dlevel_predict))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(dlevel_test, dlevel_predict)))


def qda_fun(features, dlevel):
    
    # split into 5 groups for k-Fold analysis
    cv = RepeatedStratifiedKFold(n_splits = 5, random_state = None)
    
    # determine the model to be used
    qda = QuadraticDiscriminantAnalysis()

    # run the model, get the accuracy
    accuracy = cross_val_score(qda, features, dlevel, scoring = 'accuracy', cv = cv)

    print('\nQDA Mean Accuracy(Standard Deviation): %.3f (%.3f)' % (mean(accuracy), std(accuracy)))


def linReg_fun(features, dlevel):

    # split the data into training and testing data
    feat_train, feat_test, dlevel_train, dlevel_test = train_test_split(features, dlevel, test_size = 0.2)
    
    # determine the model to be used
    linReg = LinearRegression()
    linReg.fit(feat_train, dlevel_train)

    # run the model
    dlevel_predict = linReg.predict(feat_test)

    # compare results to actual
    print('\nlinear regression')
    print('r2:', metrics.r2_score(dlevel_test, dlevel_predict))
    print('Mean Absolute Error:', metrics.mean_absolute_error(dlevel_test, dlevel_predict))
    print('Mean Squared Error:', metrics.mean_squared_error(dlevel_test, dlevel_predict))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(dlevel_test, dlevel_predict)))


def lda(features, dlevel):

    #Define LDA
    lda = LinearDiscriminantAnalysis()

    #kfold
    crossval = RepeatedStratifiedKFold(n_splits=5, random_state = None)
    score = cross_val_score(lda, features, dlevel, scoring = 'accuracy', cv=crossval,n_jobs=1)
    
    #Print the Score
    print('\nLDA Mean Accuracy(Standard Deviation): %.3f (%.3f)' % (mean(score), std(score)))


def SvM(features, dlevel):
    sup_vec = svm.SVC(kernel='linear')

    feat_train, feat_test, dlevel_train, dlevel_test = train_test_split(features, dlevel, test_size = 0.2)

    sup_vec.fit(feat_train, dlevel_train)

    predict = sup_vec.predict(feat_test)

    print("\nSVM accuracy: ",metrics.accuracy_score(dlevel_test,predict))