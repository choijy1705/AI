import tensorflow as tf

tf.set_random_seed(777)

x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5], [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.float32, [None, 3])

W = tf.Variable(tf.random_normal([4,3]), name ='weight')
b = tf.Variable(tf.random_normal([3]), name = 'bias')

# hypothesis (softmax classification)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# loss func. - Cross Entropy Error
loss = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis = 1))

# Gradient Descent Algorithm
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(train, feed_dict={X:x_data, Y:y_data}) # placehold를 사용하였기때문에 데이터도 전달해줘야 한다.

        if(step%200 == 0):
            print(step, sess.run(loss, feed_dict={X:x_data, Y:y_data}))

    a = sess.run(hypothesis, feed_dict={X:[[1, 11, 7, 9]]})
    print(a, sess.run(tf.argmax(a, 1)))

    b = sess.run(hypothesis, feed_dict={X:[[1, 3, 4, 3]]})
    print(b, sess.run(tf.argmax(b,1)))

    all = sess.run(hypothesis, feed_dict={X:[[1, 11, 7, 9],[1, 3, 4, 3]]})
    print(all, sess.run(tf.argmax(all, 1)))