import numpy as np

X = np.array([5, 10])
W = np.array([[1,2,3],[4,5,6]])
B = np.array([10, 10, 10])

print(X.shape) # (2,)
print(W.shape) # (2, 3)
print(B.shape) # (3,)

Y = np.dot(X, W) + B
print(Y) # [55 70 85]