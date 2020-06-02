import numpy as np
import matplotlib.pylab as plt

# 미분
def numerical_diff(f, x):
    h = 10e-50 # 문제점 1) 0에 가깝게 구현. But, python 의 경우
    # print(np.float32(1e-50)) 는 0.0처리 (반올림 오차)
    return (f(x+h) - f(x)) / h # 문제점 2) h 에 의한 오차 발생

# print(np.float32(1e-50)) # 0.0      0으로 나오면 의미가 없다.

# x 에서의 접선의 기울기(미분) - 왼쪽으로 이동할지 오른쪽으로 이동할지 판단할 수 있다.
def numerical_diff(f, x):
    h = 1e-4 # 0.0001 python 에서 이정도 값은 파이썬에서 0으로 처리되지 않는다
    return (f(x+h) - f(x-h)) / (2*h) # x를 중심으로 그전후의 차분을 계산한다.

# 수치 미분의 예
def function_1(x):
    return 0.01 * x**2 + 0.1 * x

x = np.arange(0.0, 20.0, 0.1) # 0.0부터 19.9까지 0.1의 간격으로 데이터 저장
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()

print(numerical_diff(function_1, 5)) # 0.1999999999990898   0.2에 근접한 값으로 나온다.
print(numerical_diff(function_1, 10)) # 0.2999999999986347  0.3에 근접한 값으로 나온다. 수학적으로의 값이고 h값을 0.001로 정의 하였기때문에 완벽한 수학적 값이 나오지는 못한다.

# 편미분의 예
# - f(x0, x1) = x0**2 + x1**2

def function_2(x):
    return x[0]**2 + x[1]**2
    # 또는 return np.sum(x**2)
def function_tmp1(x0): # x0에 대한 편미분
    return x0*x0 + 4.0**2.0

def function_tmp2(x1): # x1에 대한 편미분
    return 3.0**2.0 + x1 * x1

print(numerical_diff(function_tmp1, 3.0))
# 수치 미분 값 : 6.00000000000378
print(numerical_diff(function_tmp2, 4.0))
# 수치 미분 값 : 7.999999999999119