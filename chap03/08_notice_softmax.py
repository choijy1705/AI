import numpy as np

def softmax(x):
    exp_a = np.exp(x)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

# a = np.array([1.2, 1.5, 1.8]) # [0.23969448 0.3235537  0.43675182]
a = np.array([1010, 1000, 990])  # [nan nan nan] not a number 너무 큰 수가 나오기 때문에 계산을 할수가 없다. 컴퓨터가 표현할 수 있는 숫자는 한계가 있다.
# y = softmax(a)
# print(y)

max = np.max(a) # 배열중 최대값을 반환해준다.

result = np.exp(a-max) / np.sum(np.exp(a-max))
print(result) # [9.99954600e-01 4.53978686e-05 2.06106005e-09] nan이 발생되지 않는다. 가장 큰값으로 빼줬기 때문에

def softmax_computer(a):
    max = np.max(a)
    exp_a = np.exp(a-max) # 오버플로 처리
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

result = softmax_computer(a)
print(result)

a = np.array([1.2, 1.5, 1.8])
print(softmax(a))
print(softmax_computer(a))
# 결과에 전혀 영향을 미치지 않는다. 오버플로 문제만 해결



