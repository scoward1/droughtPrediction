# importing libraries
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
import os
import math
from featureReduction import mrmr_fun, sfs_fun
from dimensionReduction import pca_fun
from models import qda_fun


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
#os.chdir('/Users/Acer/Documents/Drought')                                  # locRepo for max

# read the data. train set brought in in two parts because it was too large to
# do at the same time. turned into numpy arrays to allow for normalization of data
train1 = pd.read_csv('washington_trainseries.txt', sep = ",", header = 0)
train2 = pd.read_csv('washington_trainseries2.txt', sep = ",", header = 0)
train = pd.concat([train1, train2], ignore_index=True)                       # putting both train sets dataframes together
test = pd.read_csv('Washington_test.txt', sep = ",", header = 0)
validation = pd.read_csv('Washington_validation.txt', sep = ",", header = 0)
train = pd.concat([train, test], ignore_index= True)

# rearrange the information to work with the MRMR feature selection
train_dlevel = train['score']                                                # creating a class series
train_dlevel = train_dlevel.values                                           # changing from series to numpy array
train = train.drop(columns = "score")                                        # dropping the class from the dataframe

# normalize the data using the sklearn package (keep the column and index names by using .iloc)
scaler = StandardScaler()
train.iloc[:, 2:20] = scaler.fit_transform(train.iloc[:, 2:20].to_numpy())
validation.iloc[:, 2:20] = scaler.fit_transform(validation.iloc[:, 2:20].to_numpy())
test.iloc[:, 2:20] = scaler.fit_transform(test.iloc[:, 2:20].to_numpy())

# changing date from object to int, train data from float64 to int64
# train['date'] = (pd.to_datetime(train['date']).dt.strftime("%Y%m%d")).astype(np.int64)

# linearly interpolate the missing dlevels
train_inter_dlevel = linInter_NaN(train_dlevel)
train_inter_dlevel_int = train_inter_dlevel.astype(np.int64)


# built-in pymrmr requries a df where the first col is the class, the rest are features
# take out the date and location because the number mess it up
train_inter_dlevel_df = pd.DataFrame(train_inter_dlevel_int)
train_inter_dlevel_df.columns = ['score']
train_mrmr_df = train_inter_dlevel_df.join(train.iloc[:, 4:20])

# determine best 10 features using mrmr
train_top_features = mrmr_fun(train_mrmr_df, train, train_inter_dlevel_int)

# use SFS to select 10 best features, use 50,000 points to run
# train_sfs = train.iloc[1:50000,:]
# dlevel_sfs = inter_dlevel_int[1:50000]
# sfs_fun(train_sfs, dlevel_sfs)

# only use the top 10 features of the train dataframe
train_fr = train.filter(train_top_features, axis = 1)

# use PCA, return new lower dimension training data
train_dr = pca_fun(train_fr)

# QDA
qda_acc = qda_fun(train_dr, train_inter_dlevel_int)
print(qda_acc)