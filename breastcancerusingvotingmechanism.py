# -*- coding: utf-8 -*-


import pickle
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

#"""LOADING DATA FROM CSV FILE"""

data=pd.read_csv("dataR2.csv")

#"""TO GET INFORMATION ABOUT VARIOUS PARAMETERS"""

data.describe()

data.head()

data['Classification'].value_counts()

X=data.drop(columns='Classification',axis=1)
Y=data['Classification']

print(X)
print(Y)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X)
standard_data=scaler.transform(X)

X=standard_data
Y=data['Classification']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=2,test_size=0.2)

# Initialize individual models
from sklearn.ensemble import RandomForestClassifier
model1 = LogisticRegression()
model2 = DecisionTreeClassifier()
model3 = SVC(probability=True)
model4 = RandomForestClassifier()

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Base models
model1 = LogisticRegression()
model2 = DecisionTreeClassifier()
model3 = SVC(probability=True)
model4 = RandomForestClassifier()

# Bagging
bagging_clf = BaggingClassifier(base_estimator=model1, n_estimators=10, random_state=42)
bagging_clf.fit(X_train, Y_train)
bagging_accuracy = accuracy_score(Y_test, bagging_clf.predict(X_test))
print("Bagging Accuracy:", bagging_accuracy)

# Boosting (AdaBoost)
adaboost_clf = AdaBoostClassifier(base_estimator=model2, n_estimators=50, random_state=42)
adaboost_clf.fit(X_train, Y_train)
adaboost_accuracy = accuracy_score(Y_test, adaboost_clf.predict(X_test))
print("AdaBoost Accuracy:", adaboost_accuracy)

# Stacking
estimators = [('lr', model1), ('dt', model2), ('svm', model3), ('rf', model4)]
voting_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
voting_clf.fit(X_train, Y_train)
stacking_accuracy = accuracy_score(Y_test, voting_clf.predict(X_test))
print("Stacking Accuracy:", stacking_accuracy)


input_data = (48,23.15,70,2.07,0.46,8.8,9.7,7.99,417)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = voting_clf.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person does not have Breast Cancer')
if (prediction[0]==1):
  print('The person has Breast Cancer')

#name of model is voting_clf
pickle.dump(voting_clf,open("breastcancerusingvotingmechanism.pkl","wb"))


