import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA

def pca_fun(train):
    # all feature reduction was done to get 10 features
    n_components = 10

    pca = PCA(n_components) #  = 0.90, svd_solver = 'full'
    reduced_train = pca.fit_transform(train)

    # check the variance and keep 95%
    tot_var = pca.explained_variance_ratio_.cumsum()
    print(tot_var)
    num_feat = 1
    for item in tot_var:
        if item >= 0.95:
           break
        else:
            num_feat += 1

    reduced_train = reduced_train[:, 0:num_feat]

    return reduced_train


def ica_fun(train):
        ICA = FastICA(n_components=10)
        reduced_train_ICA = FastICA.fit_transform(train)

        return reduced_train_ICA

