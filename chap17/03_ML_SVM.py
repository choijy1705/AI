from chap17 import mglearn
from chap17.preamble import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.svm import LinearSVC

font_location = 'C:/Windows/Fonts/malgun.ttf'
font_name = fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font', family=font_name)

X, y = mglearn.datasets.make_blobs(centers=4, random_state=8)
y = y % 2

mglearn.discrete_scatter(X[:,0], X[:,1],y)
plt.xlabel("feature 0")
plt.ylabel("feature 1")
# plt.show()

linear_svm = LinearSVC().fit(X, y)

mglearn.plots.plot_2d_separator(linear_svm, X) # 2d형태로 시각화 해주는 기능, 결정경계의 특성을 시각화 해서 보겠다.
mglearn.discrete_scatter(X[:,0], X[:,1],y)
plt.xlabel("feature 0")
plt.ylabel("feature 1")
plt.show()

# 두 번째 특성을 제곱하여 추가합니다.
# 2차원이엿던 데이터를 3차원으로 가공하여 데이터를 분류할수 있도록 해준다.
X_new = np.hstack([X, X[:, 1:]**2]) # 3차원의 형태로 변경되어진다. 차원을 하나 증가 시켜준다.

from mpl_toolkits.mplot3d import Axes3D, axis3d
figure = plt.figure()

# 3차원 그래프
ax = Axes3D(figure, elev=-152, azim=-26) # 이값을 변경하면서 바라보는 각도를 변경해줄 수 있다.

# y == 0인 포인터를 먼저 그리고, 그 다음 y == 1인 포인트를 그린다.
mask = y == 0
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b', cmap=mglearn.cm2, s=60, edgecolors='k') # 0, 1, 2 축의 차원으로 출력해준다.
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', cmap=mglearn.cm2, s=60, edgecolors='k') # 0, 1, 2 축의 차원으로 출력해준다.
ax.set_xlabel("특성0")
ax.set_ylabel("특성1")
ax.set_zlabel("특성1**2")
plt.show()

linear_svm_3d = LinearSVC().fit(X_new, y)
coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_

# 선형 결정 경계 그리기
# 시각화를 위한 처리
figure = plt.figure()
ax = Axes3D(figure, elev=-152, azim=-26)
xx = np.linspace(X_new[:,0].min() - 2, X_new[:,1].max() + 2, 50)
yy = np.linspace(X_new[:,0].min() - 2, X_new[:,1].max() + 2, 50)

# np.meshgrid() 크기를 입력받아 그 크기 입력에 대한 인폼을 받아 평면을 그려준다.
XX, YY = np.meshgrid(xx,yy)
ZZ = (coef[0] * XX + coef[1] * YY + intercept)/ -coef[2] # z축에 대한 표현방법
ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b', cmap=mglearn.cm2, s=60, edgecolors='k')
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^', cmap=mglearn.cm2, s=60, edgecolors='k')
ax.set_xlabel("특성0")
ax.set_ylabel("특성1")
ax.set_zlabel("특성1**2")
plt.show()

ZZ = YY ** 2
dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
plt.contourf(XX, YY, dec.reshape(XX.shape), levels=[dec.min(), 0, dec.max()],
             cmap=mglearn.cm2, alpha=0.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("특성 0")
plt.ylabel("특성 1")
plt.show()