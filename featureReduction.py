import pymrmr
from mrmr import mrmr_classif
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from decimal import Decimal
from sklearn.linear_model import LogisticRegression

def mrmr_fun(mrmr_df, train, inter_dlevel):
    
    # best 10 features  
    pymrmr.mRMR(mrmr_df, 'MIQ', 10)
    pymrmr.mRMR(mrmr_df, 'MID', 10)

    # use mrmr to select the 10 best features
    selected_features = mrmr_classif(train, inter_dlevel, 10)
    """
    see: pip install git+https://github.com/smazzanti/mrmr
    """
    print("10 features selected by mRMR: \n")
    print(selected_features)

def sfs_fun(features, classes):
    knn= KNeighborsClassifier(n_neighbors=100)
    # Lreg = LogisticRegression()
    # clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    
    sfs_set = sfs(knn,
        k_features=10,
        forward =True,
        floating = False,
        verbose =2,
        scoring= 'accuracy',
        cv=5).fit(features, classes)

    feat_names = list(sfs_set.k_feature_names_)
    print(feat_names)

    print('\nSequential Forward Selection (k=10):')
    print(sfs_set.k_feature_idx_)
    print('CV Score:')
    print(sfs_set.k_score_)