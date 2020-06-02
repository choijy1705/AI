import pandas as pd
import numpy as np
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split

appdata = pd.read_csv("appdata4.csv")

appdata.Installs = appdata.Installs.apply(lambda x : np.log10(x))
#appdata.Reviews = appdata.Reviews.apply(lambda x : np.log10(x))

data = pd.get_dummies(appdata, columns=['Category'])

for k in range(11,42):
    cordat = data[data[data.columns[k]] == 1]
    print(data.columns[k], ":", cordat.Reviews.corr(cordat.Installs))

dataset = data[data[data.columns[21]] == 1]
dataset = np.array(dataset)


print(dataset.shape)
a,b = dataset.shape
print(a)

x = dataset[:,[3]]
y = dataset[:,[4]]
print(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)


X = tf.placeholder(tf.float32,shape=[None,1])
Y = tf.placeholder(tf.float32,shape=[None,1])

W1 = tf.get_variable("W1", shape=[1,100],initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([100]))
L1 = tf.nn.sigmoid(tf.matmul(X,W1) + b1)

W2 = tf.get_variable("W2", shape=[100,100],initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([100]))
L2 = tf.nn.relu(tf.matmul(L1,W2)+ b2)

W3 = tf.get_variable("W3", shape=[100,1],initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([1]))

hypothesis = tf.matmul(L2,W3) + b3

loss = tf.reduce_mean(tf.square(hypothesis-Y))
train = tf.train.AdagradOptimizer(learning_rate=0.001).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    training_epochs = 20
    batch_size = 50

    for epoch in range(training_epochs):
        avg_loss = 0
        total_batch = int(a/batch_size)

        for i in range(total_batch):
            loss_val,_ = sess.run([loss, train], feed_dict={X:x_train,Y:y_train})
            avg_loss += loss_val/total_batch

        print("Epoch:", "%02d" % (epoch + 1), "loss", " {:.9f}".format(avg_loss))
    print("learning finished")

    prediction = sess.run(hypothesis, feed_dict={X: x_test})

    for s in range(1685):
        print("index:", s, "prediction:", prediction[[s], ], "actual:", y_test[[s], ])


    accuracy = tf.reduce_mean(tf.cast(tf.equal(np.round(prediction, 0),y_test), tf.float32))
    print(sess.run(accuracy, feed_dict={X: x_train}))









