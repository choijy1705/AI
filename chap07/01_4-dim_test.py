import numpy as np

x = np.random.rand(10, 1, 28, 28) # mnist 데이터 10개를 가지는 크기에 랜덤으로 실수값을 가져오겠다. 4차원의 형태로 데이터 초기값을 설정하고 있다.
print(x.shape) # (10, 1, 28, 28)
print(x[0].shape) # (1, 28, 28)
print(x[0][0].shape) # (28, 28)

