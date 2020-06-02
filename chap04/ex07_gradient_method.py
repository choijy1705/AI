# coding: utf-8
import numpy as np
import matplotlib.pylab as plt
from ex06_gradient_2d import numerical_gradient

def gradient_descent(f, init_x, lr=0.01, step_num=100): # (함수, 변수, lr:학습률(사람이 지정해준다.) * 사람이 지정해주는 것을 hyper parameter라고 한다 , step_num: 몇번 반복학습을 시킬 것인가?)
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() ) # init_x 각 계속 추가된다. 하나의 입력값을 계속 반복하면서 그값을 계속 초기화 하겠다는 의미

        grad = numerical_gradient(f, x) # f:함수,
        x -= lr * grad # grad는 입력되어지는 형상의 각각을 편미분 해준 값, 내 현재 값에서 양수이면 빼주고, 음수이면 더해주면서 기울기가 0이 되는 값에 접근해가겠다.
    return x, np.array(x_history) # 리스트로 담겨져 있는 값을 배열로 해서 반환해주고 있다. tuple이라는 자료형에 담겨서 반환해주고 있다. 바깥 괄호가 생략.


def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])    

lr = 0.1
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)
# x는 갱신된 x , x_history 변경되는 x의 값들
plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()
