import sys, os
sys.path.append(os.pardir)

from common.layer import Relu
import numpy as np

relu = Relu()

x = np.array([[1,-2,3],[4, 5, -6]])
relu.forward(x)
