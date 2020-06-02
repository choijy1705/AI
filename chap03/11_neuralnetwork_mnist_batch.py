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

x = np.array([1, 2, 1])
y = np.array([1, 2, 0])

print(x == y)
z = np.sum(x==y)
print(z)

if __name__ == '__main__':
    x, t = get_data()
    network = init_network()
    accuracy_cnt = 0
    batch_size = 100 # 배치 크기

    for i in range(0,len(x),batch_size): # x 는 테스트이미지의 갯수 10000
        x_batch = x[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1) # 1차원에서는 axis를 지정안해줘도 되지만 2차원에서는 axis를 지정해주어야한다. 해당축을 선택하여 최댓값을 선택해준다. 지정한 축별로 각각 처리
            # [[],[],[],...,[]] 2차원 일경 우 가장 바깥에 있는 대괄호가 axis=0, 안쪽의 대괄호를 axis=1로 나타내주고 있다. 따라서 안의 대괄호 즉 행별로 최댓값을 구하기 위해서는
            # axis=1 으로 지정해주어 행별로 최댓값을 구하도록 해줘야 한다.( 3차원일 경우 가장 바깥은 axis=0 가장 안쪽은 axis = 2 이다)
        accuracy_cnt += np.sum(p == t[i:i+batch_size]) # t(test label결과 값) 행렬과 행렬ㅇ
        # np.sum() 함수 안에서 true의 값은 1 false의 값은 0으로 계산되어진다.

        
    print("Accuracy : " + str(float(accuracy_cnt / len(x))))


