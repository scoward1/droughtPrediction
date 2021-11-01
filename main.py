# importing libraries

import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import os

print(os.getcwd())
os.chdir('/Users/sienn/Documents/uni/engg4553/project/droughtPrediction')
print(os.getcwd())

train1 = pd.read_csv('washington_trainseries.txt', sep = ",", header = 0)
train2 = pd.read_csv('washington_trainseries2.txt', sep = ",", header = 0)

print(train1.head())
print(train2['fips'])