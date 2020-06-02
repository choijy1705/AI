import numpy as np
from dataset.mnist import load_mnist
import pickle

def get_data():
    (x_train, t_train),(x_test, t_test) = load_mnist(normalize = True, flatten=True, one_hot_label=False)
    # 이미지 값은 0~255 값이 있는 그값을 특정 범위로 변환시키는 것을 정규화 시킨다고 한다.
    # 입력데이터를 우리가 원하는 형태로 변화시켜보는데 그냥 원데이터를 사용 할꺼냐 정규화시켜서 사용할꺼냐 라는 매개변수, normalize
    # 어느 정도 일정 비율 이면 그대로 사용하여도 되지만, 숫자 값의 차이가 심할때는 정규화를 시켜서 사용해야 한다.
    # 가장 일반적인 정규화방법은 0~1사이의 값으로 해준다.
    return (x_test, t_test)

def init_network():
    with open("sample_weight.pkl", "rb") as file:
        network = pickle.load(file) # 학습된 모델이 담겨있다. w와 bias의 값이 저장

    return network

def sigmoid(x) :
    return 1 / (1 + np.exp(-x))

def softmax(x):
    c = np.max(x)
    exp_a = np.exp(x - c) # 오버플로 처리
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    y = softmax(a3) # 분류분석은 소프트맥스

    return y

if __name__ == '__main__':
    x, t = get_data()
    network = init_network()
    accuracy_cnt = 0

    for i in range(len(x)): # x 는 테스트이미지의 갯수 10000
        y = predict(network, x[i])
        p = np.argmax(y) # 입력으로 전달된 배열의 데이터에서 가장 큰 값을 찾아준다. ( index 값으로)

        if p == t[i]:
            accuracy_cnt += 1

    print("Accuracy : " + str(float(accuracy_cnt / len(x))))

# 대용량의 데이터가 있어야 좀더 정확도를 높일 수 있다.
