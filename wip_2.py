# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 07:33:02 2019

@author: himas
"""

import pandas as pd
import os
import numpy as np

os.chdir("D:\Deep Sespis\Nov25")
new_df = pd.read_csv("bolus_500_1_1.csv")
new_df.to_csv("new_df.csv")

len(new_df)
new_df = new_df.loc[(new_df.heart_rate*new_df.sysbp*new_df.diasbp*new_df.resprate*new_df.spo2*new_df.meanbp>0)].reset_index()
new_df=new_df.drop(["index"],axis = 1)

#new_df_n.to_csv("new_df_n.csv")

new_df_ = new_df.loc[new_df.reset_index().groupby(['icustay_id'])['volumne_responsiveness'].idxmax()]
new_df_ = new_df_.reset_index()
new_df_ = new_df_.drop(["index"],axis = 1)

#new_df.to_csv("new_df.csv")
#new_df_.to_csv("new_df_.csv")



new_df = new_df[['icustay_id',
                 'hr',
                 'heart_rate',
                 'sysbp',
                 'diasbp',
                 'meanbp',
                 'resprate',
                 'spo2']]
new_df_ = new_df_[['icustay_id',
                 'hr',
                 'volumne_responsiveness',
                 'heart_rate',
                 'sysbp',
                 'diasbp',
                 'meanbp',
                 'resprate',
                 'spo2']]


file1 = pd.read_csv('file1.1.csv')
file2 = pd.read_csv('file2.1.csv')

def descriptive_stats_hr(file1, file2, hr):
    
    if hr == 0:
        df_ = pd.DataFrame(columns=file1.columns)
        
        for index, row in file2.iterrows(): 
            df_ = df_.append(file1[(file1.icustay_id == row['icustay_id']) & (file1.hr <= row['hr'])], ignore_index = True)
            
        df_ = df_.apply(pd.to_numeric, errors='ignore')
        df = df_.groupby("icustay_id")[list(file1.columns[2:])].describe(include='all').reset_index()
        df.columns = ['.'.join(col).strip() for col in df.columns.values]
        df.rename({'icustay_id.': 'icustay_id'}, axis=1, inplace=True)
        result = pd.merge(file2, df, on='icustay_id')
        
        return result
    
    else:
        df_ = pd.DataFrame(columns=file1.columns)
        temp = pd.DataFrame(columns=file1.columns[:2])
    
        for index, row in file2.iterrows():
            df_ = df_.append(file1[(file1.icustay_id == row['icustay_id']) & (file1.hr <= row['hr'])], ignore_index = True)
        
            if len(df_[df_['icustay_id']==row['icustay_id']]) > int(hr):
                df_ = df_[:-int(hr)]
                temp = temp.append(df_.iloc[-1])
            else:
                df_ = df_[df_['icustay_id'] != row['icustay_id']]
    
        df_ = df_.apply(pd.to_numeric, errors='ignore')
        temp = temp.loc[:,['icustay_id','hr']]
        temp.drop_duplicates(subset ="icustay_id", keep = "first", inplace = True)
        df = df_.groupby("icustay_id")[list(file1.columns[2:])].describe(include='all').reset_index()
        df.columns = ['.'.join(col).strip() for col in df.columns.values]
        df.rename({'icustay_id.': 'icustay_id'}, axis=1, inplace=True)
        
        temp2 = file2.loc[:,['icustay_id','volumne_responsiveness']]
        temp = temp.apply(pd.to_numeric, errors='ignore')
        result = pd.merge(temp, df, on='icustay_id')
        result = pd.merge(result, temp2, on='icustay_id')
    
        return result


df3 = descriptive_stats_hr(new_df, new_df_, 0)
df3_1 = descriptive_stats_hr(new_df, new_df_, 1)
df3_2 = descriptive_stats_hr(new_df, new_df_, 2)
df3_3 = descriptive_stats_hr(new_df, new_df_, 3)
df3_4 = descriptive_stats_hr(new_df, new_df_, 4)
df3_5 = descriptive_stats_hr(new_df, new_df_, 5)
df3_6 = descriptive_stats_hr(new_df, new_df_, 6)
    

###############


# Lets get the first>=500 bolus record for each icu stay
#df3 = df2.groupby('icustay_id').first()
#df3 = df3.drop(["tempc","glucose"],axis=1)
# 24K unique icu ids!
#((df3==0).astype(int).sum()/len(df3))*100
# Tempc 36% missing and glucose 56% missing. Remove them
df4 = pd.read_csv("Sepsis Definition.csv")
df4 = df4[["icustay_id","hadm_id", "age","is_male","race_white","race_black","race_hispanic","metastatic_cancer","diabetes","height","weight","sepsis-3"]]

##########sepsi inclusion#######
df4 = df4.loc[df4["sepsis-3"]==1,:]
len(df4)
sum(df4["sepsis-3"]==0)
#################sepsis inclusion###############
df4 = df4.drop(["sepsis-3"],axis=1)
df_5 = pd.merge(df4,df3,on="icustay_id", how= "inner").reset_index()
df_5 = df_5.drop(["index"],axis = 1)

df_5_1 = pd.merge(df4,df3_1,on="icustay_id", how= "inner").reset_index()
df_5_1 = df_5_1.drop(["index"],axis = 1)

df_5_2 = pd.merge(df4,df3_2,on="icustay_id", how= "inner").reset_index()
df_5_2 = df_5_2.drop(["index"],axis = 1)

df_5_3 = pd.merge(df4,df3_3,on="icustay_id", how= "inner").reset_index()
df_5_3 = df_5_3.drop(["index"],axis = 1)

df_5_4 = pd.merge(df4,df3_4,on="icustay_id", how= "inner").reset_index()
df_5_4 = df_5_4.drop(["index"],axis = 1)

df_5_5 = pd.merge(df4,df3_5,on="icustay_id", how= "inner").reset_index()
df_5_5 = df_5_5.drop(["index"],axis = 1)

df_5_6 = pd.merge(df4,df3_6,on="icustay_id", how= "inner").reset_index()
df_5_6= df_5_6.drop(["index"],axis = 1)

df_5_2.to_csv("df_5_2.csv")
len(df_5)
######


df_5.to_csv("df_5.csv")


x_variable_list = ['age',
 'is_male',
 'race_white',
 'race_black',
 'race_hispanic',
 'metastatic_cancer',
 'diabetes',
# 'height',
# 'weight',
 'heart_rate.count',
 'heart_rate.mean',
# 'heart_rate.std',
 'heart_rate.min',
 'heart_rate.25%',
 'heart_rate.50%',
 'heart_rate.75%',
 'heart_rate.max',
 'sysbp.count',
 'sysbp.mean',
# 'sysbp.std',
 'sysbp.min',
 'sysbp.25%',
 'sysbp.50%',
 'sysbp.75%',
 'sysbp.max',
 'diasbp.count',
 'diasbp.mean',
# 'diasbp.std',
 'diasbp.min',
 'diasbp.25%',
 'diasbp.50%',
 'diasbp.75%',
 'diasbp.max',
 'meanbp.count',
 'meanbp.mean',
# 'meanbp.std',
 'meanbp.min',
 'meanbp.25%',
 'meanbp.50%',
 'meanbp.75%',
 'meanbp.max',
 'resprate.count',
 'resprate.mean',
# 'resprate.std',
 'resprate.min',
 'resprate.25%',
 'resprate.50%',
 'resprate.75%',
 'resprate.max',
 'spo2.count',
 'spo2.mean',
# 'spo2.std',
 'spo2.min',
 'spo2.25%',
 'spo2.50%',
 'spo2.75%',
 'spo2.max']

x = df_5[x_variable_list]
y = df_5["volumne_responsiveness"]

#####          Data Prep ####### Begin
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegressionCV

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)


##to get test_ids## begin
#y_test_ = y_test[['icustay_id',"hr"]].reset_index()
#y_test_ = y_test_.drop(["index"],axis = 1)
##to get test_ids## End

### Next lest remove the 'icustay_id',"hr" columns in y##

#y_test = y_test["volumne_responsiveness"]
#y_train = y_train["volumne_responsiveness"]

### We need to build test tests now ##### Begin



x_test_1 = df_5_1[x_variable_list]
x_test_2 = df_5_2[x_variable_list]
x_test_3 = df_5_3[x_variable_list]
x_test_4 = df_5_4[x_variable_list]
x_test_5 = df_5_5[x_variable_list]
x_test_6 = df_5_6[x_variable_list]

y_test_1 = df_5_1["volumne_responsiveness"]
y_test_2 = df_5_2["volumne_responsiveness"]
y_test_3 = df_5_3["volumne_responsiveness"]
y_test_4 = df_5_4["volumne_responsiveness"]
y_test_5 = df_5_5["volumne_responsiveness"]
y_test_6 = df_5_6["volumne_responsiveness"]








#############################################################
# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
x_test_1 = sc.fit_transform(x_test_1)
x_test_2 = sc.fit_transform(x_test_2)
x_test_3 = sc.fit_transform(x_test_3)
x_test_4 = sc.fit_transform(x_test_4)
x_test_5 = sc.fit_transform(x_test_5)
x_test_6 = sc.fit_transform(x_test_6)

#####          Data Prep ####### End

########                *******Logistic Regression,RF***********          #####################
#https://www.edureka.co/blog/logistic-regression-in-python/
clf = LogisticRegressionCV(cv=10, random_state=0).fit(x_train, y_train) # for logistic

#################    for t zero ######################################
predictions = clf.predict(x_test)
probabilities = clf.predict_proba(x_test)[:,1]  

print(classification_report(y_test, predictions))
df_confusion = pd.crosstab(y_test, predictions, rownames=['Actual'], colnames=['Predicted'], margins=False)
print(df_confusion)
import matplotlib.pyplot as plt

print ('accuracy:' + str(round(accuracy_score(y_test, predictions)*100,2)) + "%")
print (clf.coef_)
print ("-feature coefficients-")
for i,j in enumerate(list(x.columns)):
    print(str(j)+" :"+str(round(clf.coef_[:,i],3)))

#https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
# ROC curve
from sklearn import preprocessing,metrics
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, predictions)
fpr, tpr, thresholds = roc_curve(y_test, probabilities)
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
print ( 'AUC:' + str(round(metrics.auc(fpr,tpr),5)))

#################    for t 1 ######################################
predictions = clf.predict(x_test_1)
probabilities = clf.predict_proba(x_test_1)[:,1]  

print(classification_report(y_test_1, predictions))
df_confusion = pd.crosstab(y_test_1, predictions, rownames=['Actual'], colnames=['Predicted'], margins=False)
print(df_confusion)
import matplotlib.pyplot as plt

print ('accuracy:' + str(round(accuracy_score(y_test_1, predictions)*100,2)) + "%")
print (clf.coef_)
print ("-feature coefficients-")
for i,j in enumerate(list(x.columns)):
    print(str(j)+" :"+str(round(clf.coef_[:,i],3)))

#https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
# ROC curve
logit_roc_auc = roc_auc_score(y_test_1, predictions)
fpr, tpr, thresholds = roc_curve(y_test_1, probabilities)
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
print ( 'AUC:' + str(round(metrics.auc(fpr,tpr),5)))

###################### fro t 2 #####################################

predictions = clf.predict(x_test_2)
probabilities = clf.predict_proba(x_test_2)[:,1]  

print(classification_report(y_test_2, predictions))
df_confusion = pd.crosstab(y_test_2, predictions, rownames=['Actual'], colnames=['Predicted'], margins=False)
print(df_confusion)
import matplotlib.pyplot as plt

print ('accuracy:' + str(round(accuracy_score(y_test_2, predictions)*100,2)) + "%")
print (clf.coef_)
print ("-feature coefficients-")
for i,j in enumerate(list(x.columns)):
    print(str(j)+" :"+str(round(clf.coef_[:,i],3)))

#https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
# ROC curve
logit_roc_auc = roc_auc_score(y_test_2, predictions)
fpr, tpr, thresholds = roc_curve(y_test_2, probabilities)
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
print ( 'AUC:' + str(round(metrics.auc(fpr,tpr),5)))

###################### fro t 3 #####################################

predictions = clf.predict(x_test_3)
probabilities = clf.predict_proba(x_test_3)[:,1]  

print(classification_report(y_test_3, predictions))
df_confusion = pd.crosstab(y_test_3, predictions, rownames=['Actual'], colnames=['Predicted'], margins=False)
print(df_confusion)
import matplotlib.pyplot as plt

print ('accuracy:' + str(round(accuracy_score(y_test_3, predictions)*100,2)) + "%")
print (clf.coef_)
print ("-feature coefficients-")
for i,j in enumerate(list(x.columns)):
    print(str(j)+" :"+str(round(clf.coef_[:,i],3)))

#https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
# ROC curve
logit_roc_auc = roc_auc_score(y_test_3, predictions)
fpr, tpr, thresholds = roc_curve(y_test_3, probabilities)
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
print ( 'AUC:' + str(round(metrics.auc(fpr,tpr),5)))

#######################################


###################### fro t 4 #####################################

predictions = clf.predict(x_test_4)
probabilities = clf.predict_proba(x_test_4)[:,1]  

print(classification_report(y_test_4, predictions))
df_confusion = pd.crosstab(y_test_4, predictions, rownames=['Actual'], colnames=['Predicted'], margins=False)
print(df_confusion)
import matplotlib.pyplot as plt

print ('accuracy:' + str(round(accuracy_score(y_test_4, predictions)*100,2)) + "%")
print (clf.coef_)
print ("-feature coefficients-")
for i,j in enumerate(list(x.columns)):
    print(str(j)+" :"+str(round(clf.coef_[:,i],3)))

#https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
# ROC curve
logit_roc_auc = roc_auc_score(y_test_4, predictions)
fpr, tpr, thresholds = roc_curve(y_test_4, probabilities)
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
print ( 'AUC:' + str(round(metrics.auc(fpr,tpr),5)))

###################### fro t 5 #####################################

predictions = clf.predict(x_test_5)
probabilities = clf.predict_proba(x_test_5)[:,1]  

print(classification_report(y_test_5, predictions))
df_confusion = pd.crosstab(y_test_5, predictions, rownames=['Actual'], colnames=['Predicted'], margins=False)
print(df_confusion)
import matplotlib.pyplot as plt

print ('accuracy:' + str(round(accuracy_score(y_test_5, predictions)*100,2)) + "%")
print (clf.coef_)
print ("-feature coefficients-")
for i,j in enumerate(list(x.columns)):
    print(str(j)+" :"+str(round(clf.coef_[:,i],3)))

#https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
# ROC curve
logit_roc_auc = roc_auc_score(y_test_5, predictions)
fpr, tpr, thresholds = roc_curve(y_test_5, probabilities)
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
print ( 'AUC:' + str(round(metrics.auc(fpr,tpr),5)))

########################### fro t 6 #####################
predictions = clf.predict(x_test_6)
probabilities = clf.predict_proba(x_test_6)[:,1]  

print(classification_report(y_test_6, predictions))
df_confusion = pd.crosstab(y_test_6, predictions, rownames=['Actual'], colnames=['Predicted'], margins=False)
print(df_confusion)
import matplotlib.pyplot as plt

print ('accuracy:' + str(round(accuracy_score(y_test_6, predictions)*100,2)) + "%")
print (clf.coef_)
print ("-feature coefficients-")
for i,j in enumerate(list(x.columns)):
    print(str(j)+" :"+str(round(clf.coef_[:,i],3)))

#https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
# ROC curve
logit_roc_auc = roc_auc_score(y_test_6, predictions)
fpr, tpr, thresholds = roc_curve(y_test_6, probabilities)
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
print ( 'AUC:' + str(round(metrics.auc(fpr,tpr),5)))

#########################*************************   RANDOM FOREST *******************################
from sklearn.ensemble import RandomForestClassifier

clf2 = RandomForestClassifier(n_estimators=20, random_state=0)
clf2.fit(x_train, y_train)

########################## t0 ##############################
predictions = clf2.predict(x_test)
probabilities = clf2.predict_proba(x_test)[:,1]

print(classification_report(y_test, predictions))
#print(confusion_matrix(y_test, predictions))
df_confusion = pd.crosstab(y_test, predictions, rownames=['Actual'], colnames=['Predicted'], margins=False)
print(df_confusion)
import matplotlib.pyplot as plt

print ('accuracy:' + str(round(accuracy_score(y_test, predictions)*100,2)) + "%")

importances = clf2.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf2.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print ("-feature importance-")
for i,j in enumerate(list(x.columns)):
    print(str(j)+" :"+str(round(clf2.feature_importances_[i],3)))


logit_roc_auc = roc_auc_score(y_test, predictions)
fpr, tpr, thresholds = roc_curve(y_test, probabilities)
plt.figure()
plt.plot(fpr, tpr, label='Random Forest')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
print ( 'AUC:' + str(round(metrics.auc(fpr,tpr),5)))

########################## t1 ##############################
predictions = clf2.predict(x_test_1)
probabilities = clf2.predict_proba(x_test_1)[:,1]

print(classification_report(y_test_1, predictions))
#print(confusion_matrix(y_test_1, predictions))
df_confusion = pd.crosstab(y_test_1, predictions, rownames=['Actual'], colnames=['Predicted'], margins=False)
print(df_confusion)
import matplotlib.pyplot as plt

print ('accuracy:' + str(round(accuracy_score(y_test_1, predictions)*100,2)) + "%")

importances = clf2.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf2.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print ("-feature importance-")
for i,j in enumerate(list(x.columns)):
    print(str(j)+" :"+str(round(clf2.feature_importances_[i],3)))


logit_roc_auc = roc_auc_score(y_test_1, predictions)
fpr, tpr, thresholds = roc_curve(y_test_1, probabilities)
plt.figure()
plt.plot(fpr, tpr, label='Random Forest')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
print ( 'AUC:' + str(round(metrics.auc(fpr,tpr),5))) 


########################## t2 ##############################
predictions = clf2.predict(x_test_2)
probabilities = clf2.predict_proba(x_test_2)[:,1]

print(classification_report(y_test_2, predictions))
#print(confusion_matrix(y_test_2, predictions))
df_confusion = pd.crosstab(y_test_2, predictions, rownames=['Actual'], colnames=['Predicted'], margins=False)
print(df_confusion)
import matplotlib.pyplot as plt

print ('accuracy:' + str(round(accuracy_score(y_test_2, predictions)*100,2)) + "%")

importances = clf2.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf2.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print ("-feature importance-")
for i,j in enumerate(list(x.columns)):
    print(str(j)+" :"+str(round(clf2.feature_importances_[i],3)))


logit_roc_auc = roc_auc_score(y_test_2, predictions)
fpr, tpr, thresholds = roc_curve(y_test_2, probabilities)
plt.figure()
plt.plot(fpr, tpr, label='Random Forest')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
print ( 'AUC:' + str(round(metrics.auc(fpr,tpr),5))) 

########################## t3 ##############################
predictions = clf2.predict(x_test_3)
probabilities = clf2.predict_proba(x_test_3)[:,1]

print(classification_report(y_test_3, predictions))
#print(confusion_matrix(y_test_3, predictions))
df_confusion = pd.crosstab(y_test_3, predictions, rownames=['Actual'], colnames=['Predicted'], margins=False)
print(df_confusion)
import matplotlib.pyplot as plt

print ('accuracy:' + str(round(accuracy_score(y_test_3, predictions)*100,2)) + "%")

importances = clf2.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf2.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print ("-feature importance-")
for i,j in enumerate(list(x.columns)):
    print(str(j)+" :"+str(round(clf2.feature_importances_[i],3)))


logit_roc_auc = roc_auc_score(y_test_3, predictions)
fpr, tpr, thresholds = roc_curve(y_test_3, probabilities)
plt.figure()
plt.plot(fpr, tpr, label='Random Forest')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
print ( 'AUC:' + str(round(metrics.auc(fpr,tpr),5))) 

########################## t4 ##############################
predictions = clf2.predict(x_test_4)
probabilities = clf2.predict_proba(x_test_4)[:,1]

print(classification_report(y_test_4, predictions))
#print(confusion_matrix(y_test_4, predictions))
df_confusion = pd.crosstab(y_test_4, predictions, rownames=['Actual'], colnames=['Predicted'], margins=False)
print(df_confusion)
import matplotlib.pyplot as plt

print ('accuracy:' + str(round(accuracy_score(y_test_4, predictions)*100,2)) + "%")

importances = clf2.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf2.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print ("-feature importance-")
for i,j in enumerate(list(x.columns)):
    print(str(j)+" :"+str(round(clf2.feature_importances_[i],3)))


logit_roc_auc = roc_auc_score(y_test_4, predictions)
fpr, tpr, thresholds = roc_curve(y_test_4, probabilities)
plt.figure()
plt.plot(fpr, tpr, label='Random Forest')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
print ( 'AUC:' + str(round(metrics.auc(fpr,tpr),5)))

########################## t5 ##############################
predictions = clf2.predict(x_test_5)
probabilities = clf2.predict_proba(x_test_5)[:,1]

print(classification_report(y_test_5, predictions))
#print(confusion_matrix(y_test_5, predictions))
df_confusion = pd.crosstab(y_test_5, predictions, rownames=['Actual'], colnames=['Predicted'], margins=False)
print(df_confusion)
import matplotlib.pyplot as plt

print ('accuracy:' + str(round(accuracy_score(y_test_5, predictions)*100,2)) + "%")

importances = clf2.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf2.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print ("-feature importance-")
for i,j in enumerate(list(x.columns)):
    print(str(j)+" :"+str(round(clf2.feature_importances_[i],3)))


logit_roc_auc = roc_auc_score(y_test_5, predictions)
fpr, tpr, thresholds = roc_curve(y_test_5, probabilities)
plt.figure()
plt.plot(fpr, tpr, label='Random Forest')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
print ( 'AUC:' + str(round(metrics.auc(fpr,tpr),5)))


########################## t6 ##############################
predictions = clf2.predict(x_test_6)
probabilities = clf2.predict_proba(x_test_6)[:,1]

print(classification_report(y_test_6, predictions))
#print(confusion_matrix(y_test_6, predictions))
df_confusion = pd.crosstab(y_test_6, predictions, rownames=['Actual'], colnames=['Predicted'], margins=False)
print(df_confusion)
import matplotlib.pyplot as plt

print ('accuracy:' + str(round(accuracy_score(y_test_6, predictions)*100,2)) + "%")

importances = clf2.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf2.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print ("-feature importance-")
for i,j in enumerate(list(x.columns)):
    print(str(j)+" :"+str(round(clf2.feature_importances_[i],3)))


logit_roc_auc = roc_auc_score(y_test_6, predictions)
fpr, tpr, thresholds = roc_curve(y_test_6, probabilities)
plt.figure()
plt.plot(fpr, tpr, label='Random Forest')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
print ( 'AUC:' + str(round(metrics.auc(fpr,tpr),5)))


###################SVM#######################
############## Simple SVM
#train
from sklearn.svm import SVC
#svclassifier = SVC(kernel='linear')
#svclassifier = SVC(kernel='poly', degree=3)
svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(x_train, y_train)


####################### t0 ####################
#predict
y_pred = svclassifier.predict(x_test)
df_confusion = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=False)
print(df_confusion)
#eval
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
print ('accuracy:' + str(round(accuracy_score(y_test, y_pred)*100,2)) + "%")

#########ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.figure()
plt.plot(fpr, tpr, label='SVM_Sigmoid')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
print ( 'AUC:' + str(round(metrics.auc(fpr,tpr),5)))
print ("-feature coefficients-")
for i,j in enumerate(list(x.columns)):
    print(str(j)+" :"+str(round(svclassifier.coef_[:,i],3)))

####################### t1 ####################
#predict
y_pred = svclassifier.predict(x_test_1)
df_confusion = pd.crosstab(y_test_1, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=False)
print(df_confusion)
#eval
from sklearn.metrics import classification_report
print(classification_report(y_test_1,y_pred))
print ('accuracy:' + str(round(accuracy_score(y_test_1, y_pred)*100,2)) + "%")

#########ROC
fpr, tpr, thresholds = roc_curve(y_test_1, y_pred)
plt.figure()
plt.plot(fpr, tpr, label='SVM_Sigmoid')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
print ( 'AUC:' + str(round(metrics.auc(fpr,tpr),5)))

####################### t2 ####################
#predict
y_pred = svclassifier.predict(x_test_2)
df_confusion = pd.crosstab(y_test_2, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=False)
print(df_confusion)
#eval
from sklearn.metrics import classification_report
print(classification_report(y_test_2,y_pred))
print ('accuracy:' + str(round(accuracy_score(y_test_2, y_pred)*100,2)) + "%")

#########ROC
fpr, tpr, thresholds = roc_curve(y_test_2, y_pred)
plt.figure()
plt.plot(fpr, tpr, label='SVM_Sigmoid')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
print ( 'AUC:' + str(round(metrics.auc(fpr,tpr),5)))


####################### t3 ####################
#predict
y_pred = svclassifier.predict(x_test_3)
df_confusion = pd.crosstab(y_test_3, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=False)
print(df_confusion)
#eval
from sklearn.metrics import classification_report
print(classification_report(y_test_3,y_pred))
print ('accuracy:' + str(round(accuracy_score(y_test_3, y_pred)*100,2)) + "%")

#########ROC
fpr, tpr, thresholds = roc_curve(y_test_3, y_pred)
plt.figure()
plt.plot(fpr, tpr, label='SVM_Sigmoid')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
print ( 'AUC:' + str(round(metrics.auc(fpr,tpr),5)))


####################### t4 ####################
#predict
y_pred = svclassifier.predict(x_test_4)
df_confusion = pd.crosstab(y_test_4, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=False)
print(df_confusion)
#eval
from sklearn.metrics import classification_report
print(classification_report(y_test_4,y_pred))
print ('accuracy:' + str(round(accuracy_score(y_test_4, y_pred)*100,2)) + "%")

#########ROC
fpr, tpr, thresholds = roc_curve(y_test_4, y_pred)
plt.figure()
plt.plot(fpr, tpr, label='SVM_Sigmoid')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
print ( 'AUC:' + str(round(metrics.auc(fpr,tpr),5)))

####################### t5 ####################
#predict
y_pred = svclassifier.predict(x_test_5)
df_confusion = pd.crosstab(y_test_5, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=False)
print(df_confusion)
#eval
from sklearn.metrics import classification_report
print(classification_report(y_test_5,y_pred))
print ('accuracy:' + str(round(accuracy_score(y_test_5, y_pred)*100,2)) + "%")

#########ROC
fpr, tpr, thresholds = roc_curve(y_test_5, y_pred)
plt.figure()
plt.plot(fpr, tpr, label='SVM_Sigmoid')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
print ( 'AUC:' + str(round(metrics.auc(fpr,tpr),5)))


####################### t6 ####################
#predict
y_pred = svclassifier.predict(x_test_6)
df_confusion = pd.crosstab(y_test_6, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=False)
print(df_confusion)
#eval
from sklearn.metrics import classification_report
print(classification_report(y_test_6,y_pred))
print ('accuracy:' + str(round(accuracy_score(y_test_6, y_pred)*100,2)) + "%")

#########ROC
fpr, tpr, thresholds = roc_curve(y_test_6, y_pred)
plt.figure()
plt.plot(fpr, tpr, label='SVM_Sigmoid')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
print ( 'AUC:' + str(round(metrics.auc(fpr,tpr),5)))






































