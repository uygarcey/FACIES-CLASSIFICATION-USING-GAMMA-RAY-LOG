
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns



#########################################################################################################################
####     This code plots distribution of layer types and correlation matrix of selected well to be used for training ####
#########################################################################################################################

GR_Log = pd.read_csv('CAX_LogFacies_Train_File.csv')
#example wells:288, 500, 521, 706, 862, 1119, 1134, 1566, 1836, 1991, 2229, 2331, 2443, 2472, 2478, 2882, 2995, 3337, 3394, 3565, 3639, 3737
GR1=GR_Log .loc[GR_Log ['well_id'].isin([288, 500, 521, 706, 862, 1119, 1134, 1566, 1836, 1991])]



print(GR1.head())

 # null object

print(GR1.describe())


# plot distribution of layer types

GR1['label'].value_counts().plot(kind='bar')
plt.title('Distribution of Coded Layers')
plt.ylabel('Numbers of layers')
plt.xlabel('Layer types')
plt.show()

plt.figure(figsize=(14,8))

GR=GR1.drop(['row_id','well_id'],axis = 1)

ax = sns.heatmap(GR.corr(), annot=True, fmt ='.0%')
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
plt.title('Correlation Matrix')
plt.show()

  


  

