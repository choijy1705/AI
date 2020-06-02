import sys, os
sys.path.append(os.pardir)

import numpy as np
from common.util import im2col

x1 = np.random.rand(10, 3, 7, 7)  # 크기 7x7 인 컬러이미지 10개
col = im2col(x1, 5, 5)
print(col.shape) # (90, 75) 2차원으로 피드백