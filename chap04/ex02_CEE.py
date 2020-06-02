# CEE: Cross Entropy Error)
import numpy as np

def cross_entropy_error(y, p):
    delta = 1e-7
    return -np.sum(y*np.log(p+delta)) # delta를 통하여 y값이 발산하지 않도록 한다.

p = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0] # 2일 확률이 60프로 2라고 예측
y = [0,0,1,0,0,0,0,0,0,0]

print(cross_entropy_error(np.array(y), np.array(p))) # 0.510825457099338
# 실제값과 예측값이 일치할때 작은 값이 나온다.

p1 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
y = [0,0,1,0,0,0,0,0,0,0]

print(cross_entropy_error(np.array(y), np.array(p1))) # 2.302584092994546
# 실제값과 예측값이 다르기 때문에 오차가 크게 나온다.

# MSE보다  예측이 맞고 틀림의 오차의 차이가 더 크게 나오기때문에 정확도가 더 높아진다. 정답의 값의 변화는 줄이고 오차의 크기는 더 크게 한다.