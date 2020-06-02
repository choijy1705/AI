# sigmoid func.

import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1+np.exp(-x))

if __name__ == "__main__":
    x = np.array([-1, 1, 2])
    y = sigmoid(x)
    print(y) # [0.26894142 0.73105858 0.88079708]
