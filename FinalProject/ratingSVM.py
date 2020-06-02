import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm


appdata = pd.read_csv("appdata4.csv")
appdata.Rating = appdata.Rating.apply(lambda x: int(x))

def minus(x):
    if x >= 5:
        x = x-4

    return x

appdata.Rating = appdata.Rating.apply(lambda x : minus(x))
appdata.Installs = appdata.Installs.apply(lambda x : np.log10(x))
#appdata.Reviews = appdata.Reviews.apply(lambda x : np.log10(x))

# appdata = np.array(appdata)


features = ['Installs', 'Reviews']
x = appdata[features]
y = appdata.Rating

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=3)

clf = svm.SVC()
clf.fit(x_train,y_train)
predict = clf.score(x_test)
print(predict)

