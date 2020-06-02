import pandas as pd
import numpy as np
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

appdata = pd.read_csv("FinalProject7.csv")
appdata = np.array(appdata)

a, b = appdata.shape


X = appdata[:a,2:4]
y = appdata[:a,4:5]



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1)


dt_model = DecisionTreeClassifier()
dt_model = dt_model.fit(y,y)

'''
ensembleDecisionTree = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                         n_estimators=10000,
                                         n_jobs=1,
                                         max_samples=1,
                                         random_state=1).fit(X_train,y_train)

ensembleRandomForest = RandomForestClassifier(
                                                n_estimators=10000,
                                                n_jobs=1,
                                                max_samples=1,
                                                random_state=1).fit(X_train, y_train)
print("학습")
print("DT model : ", dt_model.score(X_train,y_train))
print("Encemble Decision Tree : ", ensembleDecisionTree.score(X_train,y_train))
print("Encemble Random Forest : ", ensembleRandomForest.score(X_train,y_train))

print("테스트")
print("DT model : ", dt_model.score(X_test,y_test))
print("Encemble Decision Tree : ", ensembleDecisionTree.score(X_test,y_test))
print("Encemble Random Forest : ", ensembleRandomForest.score(X_test,y_test))

'''