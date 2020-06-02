import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

x_data= [[1, 2, 1], [1, 3, 2], [1, 3, 4], [1, 5, 5], [1, 7, 5], [1, 2, 5], [1, 6, 6], [1, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

# Evaluation our model using this test dataset
# 지도학습(결과가 있기 때문에)
# 분류의 값을 원한다(다중분류)
x_test = [[2, 1, 1], [3, 1, 2], [3, 3, 4]]
y_test = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]

X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 3])

W = tf.Variable(tf.random_normal([3,3]), name = "weight")
b = tf.Variable(tf.random_normal([3]), name = "bias")

# 가설함수
hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)

# 손실함수
# loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
loss = -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.matmul(X,W) + b, labels= Y))

# 경사하강법
# - overshooting(nan => 발산)
# train = tf.train.GradientDescentOptimizer(learning_rate=5.0).minimize(loss)

# - small learning rate
# train = tf.train.GradientDescentOptimizer(learning_rate=1e-10).minimize(loss)

train = tf.train.GradientDescentOptimizer(learning_rate=1000.0).minimize(loss)

prediction = tf.arg_max(hypothesis, 1) # 1 : 열단위로 처리하겠다는 뜻 열에서 가장 큰 값을 뽑아내겠다는 뜻(index)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.arg_max(Y, 1)), tf.float32)) # true , false 값을 1, 0으로 바꿔 정확도를 계산하도록 한다.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        loss_val, W_val, _ = sess.run([loss, W, train], feed_dict={X:x_data, Y:y_data})

        print(step, loss_val, W_val)