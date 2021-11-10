# importing libraries
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
from scipy.interpolate import interp1d
from mrmr import mrmr_classif
import os
import math

def interpolate_nans(padata, pkind='linear'):
    aindexes = np.arange(padata.shape[0])
    agood_indexes, = np.where(np.isfinite(padata))
    f = interp1d(agood_indexes
               , padata[agood_indexes]
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
# do at the same time
train1 = pd.read_csv('washington_trainseries.txt', sep = ",", header = 0)
train2 = pd.read_csv('washington_trainseries2.txt', sep = ",", header = 0)
test = pd.read_csv('Washington_test.txt', sep = ",", header = 0)
validation = pd.read_csv('Washington_validation.txt', sep = ",", header = 0)

# rearrange the information to work with the MRMR feature selection
frames = [train1, train2]
train = pd.concat(frames)                       # putting both train sets dataframes together
dlevel = train['score']                         # creating a class series
dlevel = dlevel.values                          # changing from series to numpy array
train = train.drop(columns = "score")           # dropping the class from the dataframe

inter_dlevel = interpolate_nans(dlevel)

selected_features = mrmr_classif(train, inter_dlevel, 10)
print(selected_features)