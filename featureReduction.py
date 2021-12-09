import pymrmr
from mrmr import mrmr_classif
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from decimal import Decimal
from sklearn.linear_model import LogisticRegression

def mrmr_fun(train_mrmr, train, inter_dlevel, numFeats):
    
    # compare MIQ and MID 
    #selected_features = pymrmr.mRMR(train_mrmr, 'MIQ', numFeats)
    #selected_features = pymrmr.mRMR(train_mrmr, 'MID', numFeats)

    # compare FCQ
    selected_features = mrmr_classif(train, inter_dlevel, 10)
    """
    see: pip install git+https://github.com/smazzanti/mrmr
    """
    print("10 features selected by mRMR: \n")
    print(selected_features)
    return selected_features

def sfs_fun(features, classes):
    #knn= KNeighborsClassifier(n_neighbors=100)
    Lreg = LogisticRegression()
    # clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    
    sfs_set = sfs(Lreg,
        k_features=10,
        forward =True,
        floating = False,
        verbose =2,
        scoring= 'accuracy').fit(features, classes)

    feat_names = list(sfs_set.k_feature_names_)
    print(feat_names)

    print('\nSequential Forward Selection (k=10):')
    print(sfs_set.k_feature_idx_)
    print('CV Score:')
    print(sfs_set.k_score_)
    return feat_names