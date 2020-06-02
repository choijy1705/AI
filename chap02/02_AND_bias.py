import numpy as np

def AND(x1, x2): # 입력되는 변수가 늘어나면 늘어날 수록 변수의 수를 늘려야 하지만 선형대수로 표현하여 행렬에 담아주면 변수를 하나의 선언으로만 한번에 해결할 수 있다.
    x = np.array([x1,x2])
    w = np.array([0.5, 0.5])
    b = -0.7

    tmp = np.sum(w*x) + b # b(bias : 편향(절편))
    if tmp <= 0:
        return 0
    else:
        return 1

if __name__ == '__main__':
    result = AND(0,0)
    print(result)

    result = AND(0, 1)
    print(result)

    result = AND(1, 0)
    print(result)

    result = AND(1, 1)
    print(result)