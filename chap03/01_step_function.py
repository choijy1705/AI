# 계단 함수(Step Function)

import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

def step_func_ndarray(x):
    y = x > 0
    return y.astype(np.int)

"""
z = np.array([-1, 1, 2])
print(z)
y = z > 0
print(y) # 조건에 만족하는 값은 true , 아닌것은 false [False  True  True]
x = y.astype(np.int)
print(x) # [0 1 1] false 일때는 0, true 일때는 1
"""
if __name__ == "__main__":
    x = step_function(3)
    print(x) # 1

    x = step_function(-3)
    print(x)  # 0

    # x = step_function(np.array([-1,1,2])) # error 배열을 처리할 수 있는 함수로 설정하지 않았다.
    # print(x)

    x = step_func_ndarray(np.array([-1, 1, 2]))
    print(x) #[0 1 1]

    x = np.arange(-5.0 , 5.0, 0.1)
    y = step_func_ndarray(x)

    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()