from chap17 import mglearn
from chap17.preamble import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_location = 'C:/Windows/Fonts/malgun.ttf'
font_name = fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font', family=font_name)


# 데이터셋을 만든다.
X, y = mglearn.datasets.make_forge()

# 산점도
mglearn.discrete_scatter(X[:,0], X[:,1], y) # 첫번째는 0 class 두번째는 1 class
plt.legend(["class 0", "class 1"], loc=4)
plt.xlabel("x")
plt.ylabel("y")
# plt.show()

print("X.shpae : ", X.shape)

X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("feature")
plt.ylabel("target")
# plt.show()

X, y = mglearn.datasets.make_blobs(centers=4, random_state=8)
y = y % 2

mglearn.discrete_scatter(X[:,0], X[:,1],y)
plt.xlabel("feature 0")
plt.ylabel("feature 1")
plt.show()