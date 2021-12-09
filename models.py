from numpy.core.fromnumeric import mean,std
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split, RepeatedStratifiedKFold, GridSearchCV, cross_val_predict, KFold
from sklearn import metrics, svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
import numpy as np
from numpy import sum, max
import matplotlib.pyplot as plt
import pandas as pd
import itertools as it


classes = ["D0", "D1", "D2", "D3", "D4"]

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

    # split the data into training and testing data
    feat_train, feat_test, dlevel_train, dlevel_test = train_test_split(features, dlevel, test_size = 0.2)
    
    # determine the model to be used
    knn = KNeighborsClassifier(n_neighbors= k_val)
    knn.fit(feat_train, dlevel_train)

    dlevel_predict = knn.predict(feat_test)
    compare = dlevel_test.__eq__(dlevel_predict)

    correct = 1
    for item in compare:
        if item == True:
            correct = correct + 1
    
    accuracy = (sum(compare))/(len(compare))

    print('\nKNN accuracy: %.3f' % accuracy)
    conf_mat = confusion_matrix(dlevel_test, dlevel_predict)
    plot_confusion_matrix(conf_mat, "KNN Confusion Matrix", classes)


def knnReg_fun(features, dlevel, k_val):
    
    # split the data into training and testing data
    feat_train, feat_test, dlevel_train, dlevel_test = train_test_split(features, dlevel, test_size = 0.2)
    
    # determine the model to be used
    knnReg = KNeighborsRegressor()
    knnReg.fit(feat_train, dlevel_train)
    
    # run the model
    dlevel_predict = knnReg.predict(feat_test)
    dlevel_predict_int = dlevel_predict.astype(np.int64)
    dlevel_test_int = dlevel_test.astype(np.int64)

    compare = dlevel_test.__eq__(dlevel_predict_int)

    correct = 1
    for item in compare:
        if item == True:
            correct = correct + 1
    
    accuracy = (sum(compare))/(len(compare))
    print('\nKNN regression accuracy: %.3f' % accuracy)
    conf_mat = confusion_matrix(dlevel_test_int, dlevel_predict_int)
    plot_confusion_matrix(conf_mat, "KNN Regression Confusion Matrix", classes)

    # compare results to actual
    print('\nKNN regression')
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

    kf = KFold(n_splits = 5)
    
    confusion_matrix_array = []
    for train_index, test_index in kf.split(features):
        feat_train, feat_test = features[train_index], features[test_index]
        dlevel_train, dlevel_test = dlevel[train_index], dlevel[test_index]

        qda.fit(feat_train, dlevel_train)
        conf_mat = confusion_matrix(dlevel_test, qda.predict(feat_test))
        confusion_matrix_array .append(conf_mat)

    confusion_matrix_sum = np.sum(confusion_matrix_array, axis = 0)

    plot_confusion_matrix(confusion_matrix_sum, "QDA Confusion Matrix", classes)

    print('\nQDA Mean Accuracy(Standard Deviation): %.3f (%.3f)' % (mean(accuracy), std(accuracy)))

def linReg_fun(features, dlevel):

    # split the data into training and testing data
    feat_train, feat_test, dlevel_train, dlevel_test = train_test_split(features, dlevel, test_size = 0.2)
    
    # determine the model to be used
    linReg = LinearRegression()
    linReg.fit(feat_train, dlevel_train)

    # run the model
    dlevel_predict = linReg.predict(feat_test)
    dlevel_predict_int = dlevel_predict.astype(np.int64)
    dlevel_test_int = dlevel_test.astype(np.int64)

    compare = dlevel_test.__eq__(dlevel_predict_int)

    correct = 1
    for item in compare:
        if item == True:
            correct = correct + 1
    
    accuracy = (sum(compare))/(len(compare))
    print('\nLinear Regression accuracy: %.3f' % accuracy)
    conf_mat = confusion_matrix(dlevel_test_int, dlevel_predict_int)
    plot_confusion_matrix(conf_mat, "Linear Regression Confusion Matrix", classes)

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
    cv = RepeatedStratifiedKFold(n_splits=5, random_state = None)
    score = cross_val_score(lda, features, dlevel, scoring = 'accuracy', cv=cv,n_jobs=1)
    
    kf = KFold(n_splits = 5)
    confusion_matrix_array = []
    for train_index, test_index in kf.split(features):
        feat_train, feat_test = features[train_index], features[test_index]
        dlevel_train, dlevel_test = dlevel[train_index], dlevel[test_index]

        lda.fit(feat_train, dlevel_train)
        conf_mat = confusion_matrix(dlevel_test, lda.predict(feat_test))
        confusion_matrix_array .append(conf_mat)

    confusion_matrix_sum = np.sum(confusion_matrix_array, axis = 0)

    plot_confusion_matrix(confusion_matrix_sum, "LDA Confusion Matrix", classes)
    
    #Print the Score
    print('\nLDA Mean Accuracy(Standard Deviation): %.3f (%.3f)' % (mean(score), std(score)))

    return mean(score)


def SvM(features, dlevel):
    sup_vec = svm.SVC(kernel='linear')

    feat_train, feat_test, dlevel_train, dlevel_test = train_test_split(features, dlevel, test_size = 0.2)

    sup_vec.fit(feat_train, dlevel_train)

    predict = sup_vec.predict(feat_test)

    conf_mat = confusion_matrix(dlevel_test, predict)
    plot_confusion_matrix(conf_mat, "Linear Regression Confusion Matrix", classes)

    print("\nSVM accuracy: ",metrics.accuracy_score(dlevel_test,predict))


def plot_confusion_matrix(conf_mat, title, classes):
    
    plt.figure()

    plt.imshow(conf_mat, cmap = plt.get_cmap('Blues'), interpolation = 'nearest')
    plt.title(title)
    plt.colorbar

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = conf_mat.max()/2

    for i, j in it.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        plt.text(j, i, format(conf_mat[i, j], fmt), horizontalalignment = "center", color = "white" if conf_mat[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    plt.show()

    return conf_mat