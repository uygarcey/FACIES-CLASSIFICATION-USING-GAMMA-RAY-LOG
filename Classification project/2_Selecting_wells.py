


import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

#pre-processing
from sklearn import preprocessing


# classifiers



from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV

# model metrics

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold



############################################################################################################################################################
#### The aim of the code is to get an idea about conditions of wells  before traning phase                                                        ##########                                           
#### These wells are evaluated by using accuracy and F1-weighted metric for different classifiers                                                  ##########
#### Running of the code is finished in  a few minute. There is warning since some wells contain just one sample of certain layer                 ##########
############################################################################################################################################################

#read data

GR_log = pd.read_csv('CAX_LogFacies_Train_File.csv')

##### metric selection #####
#metric_select=1, 'accuracy'
#metric_select=2,'f1_weighted'

metric_select= 2

if(metric_select==1):
  scoring='accuracy' 

if (metric_select==2):
 scoring='f1_weighted'

#Model evaluation
def model_evaluation(X, y, model):
  

 #cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
	# evaluate model
 #scores = cross_val_score(model, X, y, scoring=scoring, cv=cv, n_jobs=-1)
 scores = cross_val_score(model, X, y, scoring=scoring, cv=3, n_jobs=-1)
 return scores


##### model selection ######
#selection=1, LogisticRegression()
#selection=2, KNeighborsClassifier


selection=1


index=[]
results = []


# This part takes a few minutes

print(' Running of the code will be finished in  a few minutes ')
for i in range(0,3998):
  GR=GR_log.loc[GR_log['well_id'].isin([i])]
  labels = GR['label'][:] 
 
#a=list(  labels.value_counts())   
#value=0    #   more value means, more samples from layers 1,2,3,4 
#if sum(a[1::])> a[0] +value:    #  This part optinal. Using this part, wells which contain more samples from layers 1,2,3,4 can be selected  
   

  if sum((labels.values == i).any() for i in (0,1,2,3,4))==5 :     #  Selecting wells containing all layer types
      feature_1 =GR.drop(['label','row_id','well_id'],axis = 1)
      scaler = preprocessing.StandardScaler().fit(feature_1)
      features = scaler.transform(feature_1)
      X = features
      y = labels
      
      #X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size=0.33, random_state=42)
      #X=X_train
      #y=y_train
   
                       
      #MODELS


      if(selection==1):
       model=LogisticRegression()
       #model=LogisticRegression(class_weight='balanced') #
       model_name='LogisticRegression'

      if(selection==2):
       model=KNeighborsClassifier()
       model_name='KNeighborsClassifier'

      #METRICS

      cv_results=model_evaluation(X, y, model)
      if(metric_select==1):
        threshold=cv_results.mean()*100     #this for accuracy
        metric='Accuracy'

      if(metric_select==2):
        threshold=np.mean(cv_results)*100  # inding wells giving  f1_weighted threshold
        metric='f1_weighted'

      if threshold > 80: # finding wells giving  accuracy score threshold
       results.append(threshold)
       index.append(i)

print('Model Name:',model_name)
print('*******************************************')
print('Well_id:')
print(index)
print('*******************************************')
print( metric)
print(results)


