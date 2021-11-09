# importing libraries

import pandas as pd
#import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import os

print(os.getcwd())
os.chdir('E:/uni/year4/ece4553/project/localRepo/droughtPrediction')
#os.chdir('/Users/Acer/Desktop/4th Year/Python/Import Data')
print(os.getcwd())

# import the data
train1 = pd.read_csv('washington_trainseries.txt', sep = ",", header = 0)
train2 = pd.read_csv('washington_trainseries2.txt', sep = ",", header = 0)
test = pd.read_csv('Washington_test.txt', sep = ",", header = 0)
validation = pd.read_csv('Washington_validation.txt', sep = ",", header = 0)

print(train1.head())
#print(train2['fips'])

print(test.head())

print(validation.head())