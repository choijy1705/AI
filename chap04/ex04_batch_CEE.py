import numpy as np

# 이미지 1장당 평균의 CEE 계산
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(y, y.size)

    batch_size = y.shape[0]
    return - np.sum(t * np.log(y)) / batch_size

# 출력이 One-Hot Encoding 방식이 아닐 경우. 즉, '1', '5' emdrhk rkxdms ruddn
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(y, y.size)

    batch_size = y.shape[0]
    return - np.sum(np.log(y[np.arange(batch_size),t])) / batch_size