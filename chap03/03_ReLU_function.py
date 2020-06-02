import numpy as np
import matplotlib.pylab as plt

def relu(x):
    return np.maximum(0, x) # 두 입력 중 큰 값을 선택해 반환하는 함수.

if __name__ =="__main__":
    print(relu(5)) # 5
    print(relu(-5)) # 0

    x = np.arange(-5, 5, 0.1)
    y = relu(x)

    plt.plot(x, y)
    plt.ylim(-0.1, 5)
    plt.show()
