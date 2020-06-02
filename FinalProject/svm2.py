import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import random
import matplotlib.pyplot as plt
from sklearn import svm

def Evaluationmatrix(y_true, y_predict):
    print ('Mean Squared Error: '+ str(metrics.mean_squared_error(y_true,y_predict)))
    print ('Mean absolute Error: '+ str(metrics.mean_absolute_error(y_true,y_predict)))
    print ('Mean squared Log Error: '+ str(metrics.mean_squared_log_error(y_true,y_predict)))

def Evaluationmatrix_dict(y_true, y_predict, name = 'Linear - Integer'):
    dict_matrix = {}
    dict_matrix['Series Name'] = name
    dict_matrix['Mean Squared Error'] = metrics.mean_squared_error(y_true,y_predict)
    dict_matrix['Mean Absolute Error'] = metrics.mean_absolute_error(y_true,y_predict)
    dict_matrix['Mean Squared Log Error'] = metrics.mean_squared_log_error(y_true,y_predict)
    return dict_matrix

appdata = pd.read_csv("Final0324.csv")



resultsdf = pd.DataFrame()
X = appdata['Rating', 'Reviews']
y = appdata.Installs

X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.30)

model1 = svm.SVR()
model1.fit(X_train,Y_train)

Result1 = model1.predict(X_test)

resultsdf = resultsdf.append(Evaluationmatrix_dict(Y_test,Result1, name = 'SVM - Integer'),ignore_index = True)

plt.figure(figsize=(12,7))
sns.regplot(Result1,Y_test,color='teal', label = 'Integer', marker = 'x')
plt.legend()
plt.title('SVM model')
plt.xlabel('Predicted Installs')
plt.ylabel('Actual Installs')
plt.show()
