import numpy as np
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5],[0.2, 0.4, 0.6]])
    network['B1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['B2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['B3'] = np.array([0.1,0.2])

    return network

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def identity_function(x):
    return x

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3'] # unpacking방식으로 한번에 변수에 담아준다.
    b1, b2, b3 = network['B1'], network['B2'], network['B3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3) # 은닉층의 컨셉을 맞추기 위한 것, 값의 변화가 생기지 않는다.

    return y

if __name__ == '__main__':
    network = init_network()

    x = np.array([1.0 , 0.5])
    y = forward(network, x)
    print(y) # [0.31682708 0.69627909]
