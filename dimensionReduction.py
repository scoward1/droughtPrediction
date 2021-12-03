import numpy as np
import pandas as pd
from pandas.core.indexing import IndexSlice
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

classes = ["D0", "D1", "D2", "D3", "D4"]

def pca_fun(train, dlevel):
    # all feature reduction was done to get 10 features
    n_components = 10

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
    
    pca_df = pd.DataFrame(reduced_train)
    pca_df['dlevel'] = dlevel
    pca_df.columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'dlevel']

    # plot2PC(reduced_train, dlevel)

    return reduced_train


def plot2PC(pca, dlevel):

    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.scatter(pca[:, 0], pca[:, 1], pca[:, 3], c = dlevel)

    classes = ["D0", "D1", "D2", "D3", "D4"]

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    ax.set_title("First 3 PCs")
    plt.show()