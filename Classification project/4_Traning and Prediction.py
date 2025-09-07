
from calendar import c
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

import matplotlib.colors as colors
import numpy as np
import pandas as pd
from scipy.stats import randint
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
#Filter
from sklearn.datasets import make_classification
import scipy
from scipy.signal import butter, filtfilt
from scipy.signal import butter, filtfilt
#models


from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

#pre-processing
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
# Classifier

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression


from warnings import simplefilter

# model metrics
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix,make_scorer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay


                                                    
                                  #######################################################
                                  #### This code performs traning and predicitions ######
                                  #### Running of the code is finished in  a few minutes#
                                  #######################################################

#read data

GR_log = pd.read_csv('CAX_LogFacies_Train_File.csv')
GR1=GR_log.loc[GR_log['well_id'].isin([ 288,500, 521, 706, 862, 1003, 1119, 1134, 1566, 1836, 1991, 2229, 2331, 2443, 2472, 2589, 2882, 2995, 3337, 3394, 3565, 3639, 3737])]


#### model selection ######
#selection=1, DecisionTreeClassifier
#selection=2, LogisticRegression()
#selection=3, KNeighborsClassifier
#selection=4, RandomForestClassifier

selection=4


layer_test = GR1['label'][:] 
features_test1 =GR1.drop(['label','row_id','well_id'],axis = 1)


#Scaling
scaler = preprocessing.StandardScaler().fit(features_test1)
features_test = scaler.transform(features_test1)


# X and y
X = features_test
y = layer_test




#Select one of any well
# Wells:98,598,1897

GR2=GR_log.loc[GR_log['well_id'].isin([98])]

#Layer Distribution of Selected well

GR2['label'].value_counts().plot(kind='bar')
plt.title('Layer Distribution of Selected well')
plt.ylabel('Numbers of layers ')
plt.xlabel('Coded Layers')
plt.show()


features_well1 =GR2.drop(['label','row_id','well_id'],axis = 1)
layer_well = GR2['label'][:] 





#Remove oscillations by butter filter
order=5
fs=250
nyq = 0.5 * fs
normal_cutoff = 35 / nyq
b, a =butter(order, normal_cutoff, btype='low', analog=False)
CC= filtfilt(b, a, GR2.GR.values)
#Scaling
features_well1 = CC.reshape(-1, 1)
scaler = preprocessing.StandardScaler().fit(features_well1)
features_well = scaler.transform(features_well1)



X_well=features_well
y_well=layer_well

#Creating tests traning data sets


X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.3, random_state =1017)

print('*****************************************************')
print(' Running will be finished in  a few minutes ' )
print('******************************************************')
print('Length of X_train, y_train  :',len(X_train),len(y_train))
print('Length of X_test,y_test     :',len(X_test), len(y_test))
print('*******************************************************')








#select best parameters.Hyperprameter tuning.Use it for new wells to be employed for training

def hyper_parameter(classifier,parameters,X_train,y_tarin):
   scoring='accuracy' # scoring='f1_weighted'

   k_fold_method = RepeatedStratifiedKFold(n_splits=5, 
                                    n_repeats=3, 
                                    random_state=8)
   dt_model = GridSearchCV(estimator=classifier,
                    param_grid=parameters,
                    cv= k_fold_method,
                    verbose=1,
                    n_jobs=-2,
                    scoring=scoring,
                    return_train_score=True)

   result=dt_model.fit(X_train,y_train)
   return  result.best_params_



if(selection==1):
      
    parameters = {'criterion':['gini','entropy'],'max_depth':[2,3,4,5]}
    
    classifier = DecisionTreeClassifier()

    print('Hyperparameters:', hyper_parameter(classifier,parameters,X_train,y_train))
    model=DecisionTreeClassifier(criterion = 'entropy', max_depth=4, max_features= 1)


        

if(selection==2):
   

      model=LogisticRegression(C=1)
      
      model=LogisticRegression(class_weight='balanced') 
      
if(selection==3):
     

      parameters  = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7,8,9,10,11,12,13,14,15,16,17,18,19,20], 
              'p': [1, 2, 5], 'weights' : ['uniform', 'distance']}

      classifier = KNeighborsClassifier()

      print('Hyperparameters:', hyper_parameter(classifier,parameters,X_train,y_train))
 

   
      model=KNeighborsClassifier(n_neighbors=16,p=1,weights='uniform')


if(selection==4):
      
     parameters = {'max_depth':[2, 8, 16], 'n_estimators':[64, 128, 256]}
     
     classifier= RandomForestClassifier()

     print('Hyperparameters:', hyper_parameter(classifier,parameters,X_train,y_train))
     model=RandomForestClassifier(max_depth=2, n_estimators=128,class_weight='balanced')
       






#RandomForestClassifier(n_estimators=1000,class_weight='balanced')




#clf=RandomForestClassifier(max_depth=8, n_estimators=20)


#
##RandomForestClassifier(n_estimators=1000,class_weight='balanced')#DecisionTreeClassifier()#LogisticRegression()# LogisticRegression(class_weight='balanced')#
predict_train = model.fit(X_train, y_train).predict(X_train)



clfit = model.fit(X_train, y_train)


y_pred_test = clfit.predict(X_test)
y_pred_proba = clfit.predict_proba(X_test)

##Prediction for test data#####

#Calculate AUC 

auc = metrics.roc_auc_score(y_test, y_pred_proba,multi_class='ovo', average='weighted')

cm=confusion_matrix(y_true=y_test, y_pred=y_pred_test)

#Ploting Confusiom Matrix

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()
plt.title('Confusion Matrix for Test Data')
plt.show()

#Classification reports
print('Prediction for test data')
print('AUC for test data',auc)
print('Accuracy Score for trainin data :' , accuracy_score(y_train, predict_train))
print('Accuracy Score for test data :' , accuracy_score(y_test, y_pred_test))

print(classification_report(y_test, y_pred_test,zero_division=1))
print('****************************************************')



#Prediction for selected well

y_well_pred = clfit.predict(X_well)



print('Prediction for selected well')
print('Accuracy Score:' , accuracy_score(y_well, y_well_pred ))
print(classification_report(y_well, y_well_pred ,zero_division=1))



rows,cols = 1,3

#Make colors consistent for the ture and predicted layers

facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00', '#1B4F72']
cmap_facies = colors.ListedColormap(
            facies_colors[0:len(facies_colors)], 'indexed')

fig,ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12,6), sharey=True)

plt.suptitle('Selected Well', size=15)
for i in range(cols):
  if i < cols-2:
    ax[i].plot(features_well1, GR2.row_id.values, color='b', lw=0.5)
    ax[i].set_title('%s' % 'GR')
    ax[i].minorticks_on()
    ax[i].grid(which='major', linestyle='-', linewidth='0.5', color='lime')
    ax[i].grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    ax[i].set_ylim(max(GR2.row_id.values), min(GR2.row_id.values))
  elif i==cols-2:
    F = np.vstack((y_well,y_well)).T
    ax[i].imshow(F, aspect='auto',  cmap=cmap_facies, extent=[0,1,max(GR2.row_id.values), min(GR2.row_id.values)])
    ax[i].set_title('TRUE FACIES')
  elif i==cols-1:
    F = np.vstack((y_well_pred,y_well_pred)).T
    ax[i].imshow(F, aspect='auto',  cmap=cmap_facies, extent=[0,1,max(GR2.row_id.values), min(GR2.row_id.values)])
    ax[i].set_title('PRED. FACIES') 

plt.show()


