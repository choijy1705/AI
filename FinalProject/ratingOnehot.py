import pandas as pd
import numpy as np
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split


appdata = pd.read_csv("appdata4.csv")
appdata.Rating = appdata.Rating.apply(lambda x: int(x))

def minus(x):
    if x >= 5:
        x = x-4

    return x

appdata.Rating = appdata.Rating.apply(lambda x : minus(x))
appdata.Installs = appdata.Installs.apply(lambda x : np.log10(x))
#appdata.Reviews = appdata.Reviews.apply(lambda x : np.log10(x))

appdata = np.array(appdata)

x = appdata[:,[4,5,6]]
y = appdata[:,[3]]


X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.int32, shape=[None, 1])

y_one_hot = tf.one_hot(Y, 5)
Y_one_hot = tf.reshape(y_one_hot,[-1, 5])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=3)

W1 = tf.Variable(tf.random_normal([3,100]))
b1 = tf.Variable(tf.random_normal([100]))
L1 = tf.nn.relu(tf.matmul(X, W1)+ b1)

W2 = tf.Variable(tf.random_normal([100,5]))
b2 = tf.Variable(tf.random_normal([5]))


hypothesis = tf.nn.softmax(tf.matmul(L1,W2) + b2)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.matmul(L1,W2) + b2, labels=Y_one_hot))

train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # 총갯수에 대하여 1의 갯수이기 때문에 정확도가 된다.

# Machine learning

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(4001):
        sess.run(train, feed_dict={X:x_train, Y:y_train})

        if(step % 100 == 0):
            l, a = sess.run([loss, accuracy], feed_dict={X:x_train, Y:y_train})
            print("\nStep:{:5}\tLoss:{:.3f}\tAcc:{:.2%}".format(step, l, a))

    pred = sess.run(prediction, feed_dict={X:x_train})

    for p, y in zip(pred, y_train.flatten()):
        print("[{}] Prediction:{}, Y:{}".format(p==int(y), p, int(y)))

    print("\n\n테스트 데이터 적용")

    # test data
    pred = sess.run(prediction, feed_dict={X: x_test})

    for p, y in zip(pred, y_test.flatten()):
        print("[{}] Prediction:{}, Y:{}".format(p == int(y), p, int(y)))


