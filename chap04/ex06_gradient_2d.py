"""
import numpy as np
# 미분 : 특정 값에서의 접선의 기울기, 기울기를 통하여 음수 양수를 판단하여 다음 값의 방향을 설정할수 있다.
def numerical_gradient(f, x): # 수치 미분
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성(0), 0으로 초기화

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h) 계산
        x[idx] = float(tmp_val) + h               # float 실수값이라는 것을 표현, 가독성을 높이기 위한 코드 없어도 동작되는데 상관이 없다.
        fxh1 = f(x) # f(x+h1)

        # f(x-h) 계산
        x[idx] = float(tmp_val) - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 값 복원

    return grad

# - f(x0, x1) = x0**2 + x1**2
def function_2(x):
    return x[0]**2 + x[1]**2
    # 또는 return np.sum(x**2)

rnd1 = numerical_gradient(function_2, np.array([3.0, 4.0]))
print(rnd1) # 3일 때 4일때 편미분한 값 반환 [6. 8.]

rnd2 = numerical_gradient(function_2, np.array([0.0, 2.0]))
print(rnd2) # [0. 4.]

rnd3 = numerical_gradient(function_2, np.array([3.0, 0.0]))
print(rnd3) # [6. 0.]
"""

import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


def _numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # x와 형상이 같은 배열을 생성

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h) 계산
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = float(tmp_val) - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 값 복원

    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)

        return grad


def function_2(x):
    if x.ndim == 1:
        return np.sum(x ** 2)
    else:
        return np.sum(x ** 2, axis=1) # axis = 1 [[]] 안쪽 괄호를 기준으로 합하겠다.


def tangent_line(f, x):
    d = numerical_gradient(f, x)
    print(d)
    y = f(x) - d * x
    return lambda t: d * t + y  # lambda
# f(t) : return d*t + y
# 입력으로 받아서 반환되어지는 값이 간단한 형태일 때 함수의 정의를 한라인으로 간단하게 작업할 수 있는 형식을 제공해주고 있는 키워드 lambda


if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)

    X = X.flatten()
    Y = Y.flatten()

    grad = numerical_gradient(function_2, np.array([X, Y]))

    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1], angles="xy", color="#666666")  # ,headwidth=10,scale=40,color="#444444")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()
