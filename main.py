# importing libraries
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from featureReduction import mrmr_fun, sfs_fun
import os
import math
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from decimal import Decimal
from sklearn.linear_model import LogisticRegression


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

# rearrange the information to work with the MRMR feature selection
dlevel = train['score']                                                     # creating a class series
dlevel = dlevel.values                                                      # changing from series to numpy array
train = train.drop(columns = "score")                                       # dropping the class from the dataframe

# normalize the data using the sklearn package (keep the column and index names by using .iloc)
scaler = StandardScaler()
train.iloc[:, 2:20] = scaler.fit_transform(train.iloc[:, 2:20].to_numpy())
validation.iloc[:, 2:20] = scaler.fit_transform(validation.iloc[:, 2:20].to_numpy())
test.iloc[:, 2:20] = scaler.fit_transform(test.iloc[:, 2:20].to_numpy())

# changing date from object to int, train data from float64 to int64
# multiply the float val by 100,000 so that none of the information is lost
train['date'] = (pd.to_datetime(train['date']).dt.strftime("%Y%m%d")).astype(np.int64)
# train.iloc[:, 2:20] = (train.iloc[:, 2:20]*100000).astype(np.int64)

# linearly interpolate the missing dlevels
inter_dlevel = linInter_NaN(dlevel)
inter_dlevel_int = inter_dlevel.astype(np.int64)

# built-in pymrmr requries a df where the first col is the class, the rest are features
inter_dlevel_df = pd.DataFrame(inter_dlevel)
inter_dlevel_df.columns = ['score']
mrmr_df = inter_dlevel_df.join(train)

# determine best 10 features using 3 different types of mrmr
# mrmr_fun(mrmr_df, train, inter_dlevel)

# use SFS to select 10 best features, use 50,000 points to run
train_sfs = train.iloc[1:50000,:]
dlevel_sfs = inter_dlevel_int[1:50000]
sfs_fun(train_sfs, dlevel_sfs)

