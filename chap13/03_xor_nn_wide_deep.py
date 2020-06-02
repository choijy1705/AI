import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

# xor gate
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [  [0],  [1],  [1],  [0]] # XOR

x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

W1 = tf.Variable(tf.random_normal([2,10]), name="weight")
b1 = tf.Variable(tf.random_normal([10]), name="bias")
layer1 = tf.sigmoid(tf.matmul(X, W1)+ b1)

W2 = tf.Variable(tf.random_normal([10,10]), name="weight2")
b2 = tf.Variable(tf.random_normal([10]), name="bias2")
layer2 = tf.sigmoid(tf.matmul(layer1,W2) + b2)

W3 = tf.Variable(tf.random_normal([10,1]), name="weight3")
b3 = tf.Variable(tf.random_normal([10]), name="bias3")
layer3 = tf.sigmoid(tf.matmul(layer2,W3) + b3)

W4 = tf.Variable(tf.random_normal([10,1]), name="weight4")
b4 = tf.Variable(tf.random_normal([1]), name="bias4")

# 가설함수
hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4)

# 손실함수(Cross Entropy Error를 전개한 식)
loss = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))

# 경사하강법
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32) # 0.5 이상이면 true, true이면 1, false이면 0
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10101):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step % 100 == 0:
            print(step, sess.run(loss, feed_dict={X:x_data, Y:y_data}))

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})

    print("\nHypothesis:",h, "\nCorrect:", c, "\nAccuracy:", a)
