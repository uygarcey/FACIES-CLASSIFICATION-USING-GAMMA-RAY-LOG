

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno


############################################################################################################################
#### #This code checks  completeness of data, plots distribution of layer types and correlation matrix of traning dataa ##
############################################################################################################################



GR_Log = pd.read_csv('CAX_LogFacies_Train_File.csv')


print(GR_Log.head())

 # null object

print(GR_Log.describe())

# checking completeness of data
msno.matrix(GR_Log,figsize=(10,5), fontsize=7)
plt.title(' Data completeness')
plt.ylabel('Sample Numbers')
plt.xlabel('Data')
plt.show()

# plot distribution of layer types

GR_Log['label'].value_counts().plot(kind='bar')
plt.title('Distribution of Coded Layers')
plt.ylabel('Numbers of layers')
plt.xlabel('Layer types')
plt.show()

plt.figure(figsize=(14,8))

GR=GR_Log.drop(['row_id','well_id'],axis = 1)
ax = sns.heatmap(GR.corr(), annot=True, fmt ='.0%')
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
plt.title('Correlation Matrix')
plt.show()

  


  


