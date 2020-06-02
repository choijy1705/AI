from common.util import im2col
import numpy as np


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        print("self.mask : " ,self.mask)
        out = x.copy()
        print("out1 : ",out)
        out[self.mask] = 0
        print("out2 : ", out)

        return out


    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Pooling:
    def __init__(self,pool_h,pool_w,stride=1,pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(((H + 2 * self.pad - self.pool_h) / self.stride) + 1)
        out_w = int(((W + 2 * self.pad - self.pool_w) / self.stride) + 1)

        col = im2col(x,self.pool_h,self.pool_w,self.stride,self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        out = np.max(col, axis=1)
        out = out.reshape(N,out_h,out_w,C).transpose(0,3,1,2)

        return out


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout,axis=0)

        return dx


def softmax(a):
    max = np.max(a)
    exp_a = np.exp(a-max) # 오버플로 처리
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y+delta))


class SoftmaxWithLoss:
    def __init__(self):
        self.t = None  # 정답레이블(One-Hot Encoding)
        self.y = None  # softmax의 출력
        self.loss = None # 손실 함수

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx