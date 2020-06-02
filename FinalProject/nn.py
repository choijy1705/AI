import pandas as pd
import numpy as np
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split

appdata = pd.read_csv("app5.csv")
appdata = np.array(appdata)

a,b = appdata.shape

x = appdata[:,[3,4]]
y = appdata[:,[5]]

def divide(y):
    y = y/10000
    return y

def multiple(x):
    x = x * 10000
    return x

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

X = tf.placeholder(tf.float32,shape=[None,2])
Y = tf.placeholder(tf.float32,shape=[None,1])

W1 = tf.get_variable("W1", shape=[2,100], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([100]))
L1 = tf.nn.relu(tf.matmul(X,W1) + b1)

W2 = tf.get_variable("W2", shape=[100,100], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([100]))
L2 = tf.nn.relu(tf.matmul(L1,W2)+ b2)

W3 = tf.get_variable("W3", shape=[100,1], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([1]))

hypothesis = tf.matmul(L2,W3) + b3

loss = tf.reduce_mean(tf.square(hypothesis-Y))
train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)





with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    training_epochs = 2
    batch_size = 1000

    for epoch in range(training_epochs):
        avg_loss = 0
        total_batch = int(a/batch_size)

        for i in range(total_batch):
            loss_val,_ = sess.run([loss, train], feed_dict={X:x_train,Y:divide(y_train)})
            avg_loss += loss_val/total_batch

        print("Epoch:", "%02d" % (epoch + 1), "loss", " {:.9f}".format(avg_loss))
    print("learning finished")

    prediction = sess.run(hypothesis, feed_dict={X: x_test})

    print("index:", i, "prediction:", prediction[i:i + 1, ], "actual:", divide(y_test[i:i + 1, ]))

    accuracy = tf.reduce_mean(tf.cast(tf.equal(np.round(prediction, 0), divide(y_test)), tf.float32))
    print(sess.run(accuracy, feed_dict={X: x_train}))









