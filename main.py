# importing libraries
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
import os
import math
from featureReduction import mrmr_fun, sfs_fun
from dimensionReduction import ica_fun, pca_fun
import sys


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

#os.chdir('D:/uni/year4/ece4553/project/localRepo/droughtPrediction')        # locRepo for sie
os.chdir('/Users/Acer/Documents/Drought')                                  # locRepo for max

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

#Change Location column

Location = train.iloc[0:242190,0]

count=1

for x in range (242190):
    if(x<= 6210*count):
        Location[x] = count
        if (x % 6210 == 0 and x!=0):
              count = count+1

#Break Apart Date

new_date=train
print(new_date.date.head())

new_date.iloc[0:242190,1] = pd.to_datetime(new_date.iloc[0:242190,1], 
format = '%Y-%m-%dT',
errors = 'coerce')

#Date Split in to year-month-day
new_date['year'] = new_date.iloc[0:242190,1].dt.year
new_date['month'] =  new_date.iloc[0:242190,1].dt.month
new_date['day'] = new_date.iloc[0:242190,1].dt.day

new_date.groupby('year').size()
new_date.groupby('month').size()
new_date.groupby('day').size()


print(new_date[['year','month','day']].head())
train = train.drop(columns = "date")   
train = train.drop(columns = "fips")   

#Add Split date to training set
train['day'] = new_date.day
train['month'] = new_date.month
train['year'] = new_date.year

train['Location'] = Location
print(train.head())

# changing date from object to int, train data from float64 to int64
#train['date'] = (pd.to_datetime(train['date']).dt.strftime("%Y%m%d")).astype(np.int64)

# linearly interpolate the missing dlevels
inter_dlevel = linInter_NaN(dlevel)
inter_dlevel_int = inter_dlevel.astype(np.int64)

# built-in pymrmr requries a df where the first col is the class, the rest are features
inter_dlevel_df = pd.DataFrame(inter_dlevel)
inter_dlevel_df.columns = ['score']
#mrmr_df = inter_dlevel_df.join(train)

# determine best 10 features using 3 different types of mrmr
#top_features = mrmr_fun(mrmr_df, train, inter_dlevel)

# use SFS to select 10 best features, use 50,000 points to run
#train_sfs = train.iloc[1:50000,3:20]
#dlevel_sfs = inter_dlevel_int[1:50000]
#sfs_fun(train_sfs, dlevel_sfs)

# only use the top 10 features of the train dataframe
#train_fr = train.filter(top_features, axis = 1)
#train_fr = train[[top_features]].copy()
#print(train_fr)

# use PCA, return new lower dimension training data
#train_dr = pca_fun(train_fr)
#print(train_dr)

#Use ICA, return new lower dimension training data
#train_dr_ica = ica_fun(train_fr)
#print(train_dr_ica)
