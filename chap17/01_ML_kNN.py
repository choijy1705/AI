from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from chap17 import mglearn
import matplotlib
import matplotlib.pylab as plt
import sys
print(sys.path)

iris_dataset = load_iris()
print("iris_dataset의 key:\n", iris_dataset.keys())
print(iris_dataset['DESCR'][:193]+'\n...')
print("타깃의 이름 : ", iris_dataset['target_names'])
print("특성의 이름 : ", iris_dataset['feature_names'])
print("data의 타입 : ", type(iris_dataset['data']))
print("data의 크기 : ", iris_dataset['data'].shape)
print("data의 처음 다섯 행 : \n", iris_dataset['data'][:5])
print("target의 타입 : ", type(iris_dataset['target']))
print("target의 크기 : ", iris_dataset['target'].shape)
print("타깃 : \n", iris_dataset['target'])

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
print("X_train 크기 : ", X_train.shape)
print("y_train 크기 : ", y_train.shape)
print("X_test 크기 : ", X_test.shape)
print("y_test 크기 : ", y_test.shape)

# 75프로 train, 25프로 test용

# X_train 데이터를 사용해서 데이터 프레임을 만든다.
# 열의 이름은 iris_dataset.feature_name에 있는 문자열을 사용한다.
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

# 데이터 프레임을 사용해 y_train에 따라 색으로 구분된 산점도 행렬을 만든다.
# 입력으로 전달된 데이터를 시각화 해준다.
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o', hist_kwds={'bins':20}, s=60, alpha=0.8, cmap=mglearn.cm3)
# plt.show()

# kNN 알고리즘
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train) # knn에서 학습

# 예측
X_species = np.array([[5., 2.9, 1., 0.2]])

prediction = knn.predict(X_species)
print("예측 : ", prediction)
print("예측 타겟 이름 : ", iris_dataset['target_names'][prediction])

# 모델 평가하기
y_pred = knn.predict(X_test)
print("테스트 예측값 : ", y_pred)

print("테스트 정확도1 : {:.2f}".format(np.mean(y_pred == y_test)))
print("테스트 정확도2 : {:.2f}".format(knn.score(X_test, y_test))) # score 메서드를 이용하여 정확도를 피드백받을 수 있다.