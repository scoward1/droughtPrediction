import numpy as np
import pandas as pd
from pandas.core.indexing import IndexSlice
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier

def pca_fun(train, dlevel, n_components):
    # all feature reduction was done to get 10 features
    # n_components = 10

    pca = PCA(n_components)
    reduced_train = pca.fit_transform(train)

    # check the variance and keep 99%
    tot_var = pca.explained_variance_ratio_.cumsum()
    
    num_feat = 1
    for item in tot_var:
        if item >= 0.99:
           break
        else:
            num_feat += 1

    reduced_train = reduced_train[:, 0:num_feat]

    # plot2PC(reduced_train, dlevel)

    return reduced_train


# plot the results from PCA
def plot2PC(pca, dlevel):

    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.scatter(pca[:, 0], pca[:, 1], pca[:, 3], c = dlevel)

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    ax.set_title("California - 3 PCs")
    plt.show()


# SVD
item = 0
def SVD(train, inter_dlevel):

    rf_red_mat = []

    for item in range (9):
        item +=1
        svd = TruncatedSVD(n_components=item)
        X_reduced = svd.fit_transform(train)
        rf_reduced = RandomForestClassifier(oob_score=True)
        rf_reduced.fit(X_reduced, inter_dlevel)
        #rf_red_mat[item] = rf_reduced

        print("Transformed Eigendecomposition Matrix SVD")
        print(X_reduced)
        print(svd.explained_variance_ratio_.sum())
        print(svd)
        #print(rf_red_mat)
        #return rf_red_mat