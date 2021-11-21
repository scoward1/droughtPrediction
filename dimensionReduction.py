import numpy as np
from sklearn.decomposition import PCA

def pca_fun(train):
    # all feature reduction was done to get 10 features
    n_components = 10

    pca = PCA(n_components)
    reduced_train = pca.fit_transform(train)

    return reduced_train