import numpy as np

# Vector(1차원 배열)
X = np.array([1,2,3,4,5])
print(X.shape) # (5,)

# 3 x 2 행렬과 2 x 3 행렬의 내적 (Matrix)
A = np.array([[1,2],[3,4],[5,6]])
print(A.shape) # (3, 2)
B = np.array([[1,2,3],[4,5,6]])
print(B.shape) # (2, 3)

Z = np.dot(A, B)
print(Z)
'''
[[ 9 12 15]
 [19 26 33]
 [29 40 51]]
'''

# Array(3차원 배열 이상)
C = [[[1,2,3,4],[5,6,7,8],[9,10,11,12]],[[13,14,15,16],[17,18,19,20],[21,22,23,24]]]
X = np.array(A) # 2면 3행 4열
print(X.shape) # (2, 3, 4)
print(np.ndim(X)) # 3차원

# A ● B != B ● A  내적은 교환법칙이 성립하지 않는다.
Z = np.dot(B,A)
print(Z)
'''
[[22 28]
 [49 64]]
'''

# A가 2차원 행렬, B가 1차원 배열 내적
A = np.array([[1,2],[3,4],[5,6]]) # 3행 2열
B = np.array([7,8]) # B.shape : (2,)
print(np.dot(A,B))
'''
[23 53 83]
'''

# 신경망의 내적
X = np.array([1,2]) # x1 = 1, x2 = 2
W = np.array([1,3,5],[2,4,6])
Y = np.dot(X,W)
print(Y)


















