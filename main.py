# importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from numpy import inner, mean, std
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
import os
from featureReduction import mrmr_fun, sfs_fun
from dimensionReduction import pca_fun, SVD
from models import qda_fun, knn_fun, knn_neighbors, linReg_fun, lda, SvM, knnReg_fun


# function to interpolate the dlevels that are entered as NaN
def linInter_NaN(nanData, pkind='linear'):
    """
    see: https://stackoverflow.com/a/53050216/2167159
    """
    aindexes = np.arange(nanData.shape[0])
    agood_indexes, = np.where(np.isfinite(nanData))
    f = interp1d(agood_indexes
               , nanData[agood_indexes]
               , bounds_error=False
               , copy=False
               , fill_value="extrapolate"
               , kind=pkind)
    return f(aindexes)

os.chdir('D:/uni/year4/ece4553/project/localRepo/droughtPrediction')        # locRepo for sie
# os.chdir('/Users/Acer/Documents/Drought')                                  # locRepo for max

# read the data. train set brought in in two parts because it was too large to
# do at the same time. turned into numpy arrays to allow for normalization of data

train1 = pd.read_csv('washington_trainseries.txt', sep = ",", header = 0)
train2 = pd.read_csv('washington_trainseries2.txt', sep = ",", header = 0)
test = pd.read_csv('Washington_test.txt', sep = ",", header = 0)
validation = pd.read_csv('Washington_validation.txt', sep = ",", header = 0)
frames = [train1, train2, validation, test]
train = pd.concat(frames, ignore_index= True)
"""
train1 = pd.read_csv('california_trainseries.txt', sep = ",", header = 0)
train2 = pd.read_csv('california_trainseries2.txt', sep = ",", header = 0)
test = pd.read_csv('california_testseries.txt', sep = ",", header = 0)
validation = pd.read_csv('california_validationseries.txt', sep = ",", header = 0)
frames = [train1, train2, validation, test]
train = pd.concat(frames, ignore_index= True)
"""
# rearrange the information to work with the MRMR feature selection
dlevel = pd.Series(train["score"].values)                               # creating a class series
train = train.drop(columns = "score")                                   # dropping the class from the dataframe

# change location variable
Location = train.iloc[:,0]

count = 1
for x in range (len(Location)):
    if(x<= 6210*count):
        Location[x] = count
        if (x % 6210 == 0 and x!=0):
              count = count+1

# break apart date
new_date = train
new_date.iloc[:,1] = pd.to_datetime(new_date.iloc[:,1], format = '%Y-%m-%dT', errors = 'coerce')

# split date into year-month-day
new_date['year'] = new_date.iloc[:,1].dt.year
new_date['month'] =  new_date.iloc[:,1].dt.month
new_date['day'] = new_date.iloc[:,1].dt.day

new_date.groupby('year').size()
new_date.groupby('month').size()
new_date.groupby('day').size()

# drop orginal date and location columns
train = train.drop(columns = "date")   
train = train.drop(columns = "fips")

# add new date data to training set
train['day'] = new_date.day
train['month'] = new_date.month
train['year'] = new_date.year
train['Location'] = Location

# normalize the data using the sklearn package (keep the column and index names by using .iloc)
scaler = StandardScaler()
train.iloc[:, 0:22] = scaler.fit_transform(train.iloc[:, 0:22].to_numpy())

# linearly interpolate the missing dlevels
inter_dlevel = linInter_NaN(dlevel)
inter_dlevel_int = inter_dlevel.astype(np.int64)

# built-in pymrmr requries a df where the first col is the class, the rest are features
inter_dlevel_df = pd.DataFrame(inter_dlevel_int)
inter_dlevel_df.columns = ['score']
mrmr_df = inter_dlevel_df.join(train)

# determine best 10 features using mrmr
top_features = mrmr_fun(mrmr_df, train, inter_dlevel, 10)
train_fr = train.filter(top_features, axis = 1)

# use SFS to select 10 best features
#train_sfs = train.iloc[:,0:22]
#level_sfs = inter_dlevel_int
#feats = sfs_fun(train_sfs, level_sfs)
#train_sfs = train.filter(feats, axis = 1)

# use PCA, return new lower dimension training data
train_dr = pca_fun(train_fr, inter_dlevel_int)

# SVD - Singular Value Decomposition
train_SVD = SVD(train_fr,inter_dlevel_int)

# LDA - Linear Discriminent Analysis
lda(train_dr, inter_dlevel_int)

# QDA - Quadratice Disriminent Analysis
qda_fun(train_dr, inter_dlevel_int)

# SVM - Standard Vector Machine
SvM(train_dr, inter_dlevel_int)

# KNN - k-Nearest Neigbors
# knn_neighbors(train_dr, inter_dlevel_int) # determine optimal number of k-values
knn_neighbors = 5
knn_fun(train_dr, inter_dlevel_int, knn_neighbors)
knnReg_fun(train_dr, inter_dlevel, knn_neighbors)

# linear regression
linReg_fun(train_dr, inter_dlevel)

# used to plot the correlation maps for feature selection
correl = train.corr()
mask = np.triu(np.ones_like(correl, dtype=np.bool))
sns.set_style(style = 'white')
sns.set(font_scale=1)
cmap = sns.diverging_palette(10, 250, as_cmap=True)
sns.heatmap(correl, cmap=cmap, mask=mask, cbar_kws={"shrink":.5}, annot=True)
plt.show()