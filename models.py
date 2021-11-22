from numpy.core.fromnumeric import mean
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
from numpy import mean, std

def qda_fun(features, dlevel):
    
    # split into 5 groups for k-Fold analysis
    cv = KFold(n_splits = 5, shuffle = False, random_state = None)
    
    # determine the model to be used
    qda = QuadraticDiscriminantAnalysis()

    # run the model, get the accuracy
    accuracy = cross_val_score(qda, features, dlevel, scoring = 'accuracy', cv = cv)

    print('Accuracy: %.3f (%.3f)' % (mean(accuracy), std(accuracy)))

    return accuracy