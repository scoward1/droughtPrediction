# importing libraries
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
from mrmr import mrmr_classif
import os
import math

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

print(os.getcwd())
os.chdir('D:/uni/year4/ece4553/project/localRepo/droughtPrediction')        # locRepo for sie
#os.chdir('/Users/Acer/Desktop/4th Year/Python/Import Data')                # locRepo for max
print(os.getcwd())

# read the data. train set brought in in two parts because it was too large to
# do at the same time. turned into numpy arrays to allow for normalization of data
train1 = pd.read_csv('washington_trainseries.txt', sep = ",", header = 0)
train2 = pd.read_csv('washington_trainseries2.txt', sep = ",", header = 0)
frames = [train1, train2]
train = pd.concat(frames)                       # putting both train sets dataframes together
test = pd.read_csv('Washington_test.txt', sep = ",", header = 0)
validation = pd.read_csv('Washington_validation.txt', sep = ",", header = 0)

# rearrange the information to work with the MRMR feature selection
dlevel = train['score']                         # creating a class series
dlevel = dlevel.values                          # changing from series to numpy array
train = train.drop(columns = "score")           # dropping the class from the dataframe

# normalize the data using the sklearn package (keep the column and index names by using .iloc)
scaler = StandardScaler()
train.iloc[:, 3:-1] = scaler.fit_transform(train.iloc[:,3:-1].to_numpy())
validation.iloc[:, 3:-1] = scaler.fit_transform(validation.iloc[:,3:-1].to_numpy())
test.iloc[:, 3:-1] = scaler.fit_transform(test.iloc[:,3:-1].to_numpy())

# linearly interpolate the missing dlevels
inter_dlevel = linInter_NaN(dlevel)

# use mrmr to select the 10 best features
selected_features = mrmr_classif(train, inter_dlevel, 10)
"""
see: pip install git+https://github.com/smazzanti/mrmr
"""
print("10 features selected by mRMR: \n")
print(selected_features)