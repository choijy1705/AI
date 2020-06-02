import tensorflow as tf
import numpy as np
import matplotlib
import os
import matplotlib.pyplot as plt

tf.set_random_seed(777)

def MinMaxScalar(data): # 0과 1사이의 값으로 data를 정제해주는 과정
    numerator = data - np.min(data, axis =0) # open은 open만 high는 high만 ...본다. 5개의 값이 나온다.
    denominator = np.max(data, axis=0) - np.min(data, axis=0)

    return numerator / (denominator + 1e-7)

# train Parameter
data_dim = 5
output_dim = 1
learning_rate = 0.01
sequence_length = 7
hidden_dim = 10 # 출력의 갯수는 임의로 잡아주면 된다. 출력의 갯수가 증가하면 weight값이 증가하기 때문에 연산의 양이 많아지게 된다. FC와 연결되기 때문에
# 결과가 늦어지는 만큼 정확도가 높아지는 경향이 있다.

training_epochs = 500

# open, high, low, volume, close
xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
xy = xy[::-1] # reverse order 데이터를 재정렬한다.
xy = MinMaxScalar(xy)
# print(xy)

x = xy
y = xy[:,[-1]] # 마지막 close값을 label로 사용한다. 종가의 가격 , 행은 전체 행을 선택

dataX = []
dataY = []

for i in range(0, len(y) - sequence_length):
    _x = x[i:i+sequence_length]
    _y = y[i+sequence_length]
    print(_x, "->",_y)
    dataX.append(_x)
    dataY.append(_y)

# train / test split
train_size = int(len(dataY) * 0.7)
test_size = len(dataY)- train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])


X = tf.placeholder(tf.float32, [None, sequence_length, data_dim])
Y = tf.placeholder(tf.float32, [None, output_dim])

cell = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_dim)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

# FC layer
Y_pred = tf.contrib.layers.fully_connected(outputs[:,-1], output_dim, activation_fn = None) # 가설함수의 개념

loss = tf.reduce_mean(tf.square(Y_pred - Y))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None,1])
predictions = tf.placeholder(tf.float32, [None,1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(training_epochs):
        _, loss_val = sess.run([train, loss], feed_dict={X:trainX, Y:trainY})
        print("[step : {}] loss:{}".format(i,loss_val))

    # Test
    test_predict = sess.run(Y_pred, feed_dict={X:testX})
    rmse_val = sess.run(rmse, feed_dict={targets:testY, predictions:test_predict})

    print("RMSE :{}".format(rmse_val))

    plt.plot(testY)
    plt.plot(test_predict)
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
    plt.show()


