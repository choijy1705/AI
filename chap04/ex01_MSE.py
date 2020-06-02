# MSE(Mean Squared Error) : 손실함수
import numpy as np

def mean_squared_error(y, p):
    return 0.5 * np.sum((y - p)**2)

# 예측 결과 : 2
p = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0] # 2일 확률이 60프로 2라고 예측
y = [0,0,1,0,0,0,0,0,0,0] # One-Hot Encoding 방법으로 출력(실제값)

print(mean_squared_error(np.array(y), np.array(p)))
# 0.09750000000000003

# 예측 결과 : 7, 실제값 : 2
p1 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
y = [0,0,1,0,0,0,0,0,0,0]

print(mean_squared_error(np.array(y), np.array(p1))) # 손실함수는 마지막의 예측된값과 실제값을 비교하는 것
# 0.5975