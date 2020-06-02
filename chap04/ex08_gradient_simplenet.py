# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 정규분포로 초기화  , randn(2,3) : 2행 3열의 행렬을 만들어 정규분포를 따르는 6개의 값을 읽어와 담아준다.
                                      # W값을 정규분포를 따르는 값으로 초기화해준다. 초기값을 어떻게 잡아주는가에 따라 정확도의 차이가 있다.
    def predict(self, x):
        return np.dot(x, self.W) # 입력으로 전달된 x값과 w값 을 내적하고있다.

    def loss(self, x, t): # 손실함수 정의, x:입력값, t:정답
        z = self.predict(x)
        y = softmax(z) # 총합이 1이 되어지는 가중치 값이 출력
        loss = cross_entropy_error(y, t)

        return loss # 예측값과 실제값의 오차 반환


x = np.array([0.6, 0.9]) # 입력의 신호값
t = np.array([0, 0, 1]) # 결과값(정답)

net = simpleNet() # instance 생성
print("W = \n", net.W)

p = net.predict(x)
print("예측값 : ", p) # 행렬 1x3 형태의 행렬으로 예측값이 나온다

idx = np.argmax(p) # 최대값의 인덱스
print("인덱스 : ", idx)

loss = net.loss(x, t)
print("손실함수 :", loss)

f = lambda w: net.loss(x, t) # 함수를 정의한 것을 한줄로 표현해주는 방법, 밑의 주석과 같은 기능
dW = numerical_gradient(f, net.W)

print(dW)

'''
def f(W):
    return net.loss(x, t)

dW = numerical_gradient(f, net.W)
print("기울기 : \n", dW)
'''