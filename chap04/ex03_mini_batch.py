import sys,os
sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train),(x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# 저장되어 있는 데이터 셋의 결과를 one hot encoding 방법으로 반환해달라는 뜻

print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000, 10) 훈련용의, label one hot encoding으로 표현하였기 때문에 10이 나온다.

print(t_train.shape[0]) # 60000

train_size = t_train.shape[0]
batch_size = 10

# 지정한 범위의 수 중에서 무작위로 원하는 개수만 선택.
batch_mask = np.random.choice(train_size, batch_size)
print(batch_mask)
# [44783 17823 48849 11201 42530 30118  6210 27138 45799  1807] 0~59999개에서 랜덤하게 추출한 index값

x_batch = x_train[batch_mask] # 이미지데이터
t_batch = t_train[batch_mask] # labe 데이터터