import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])

W1 = tf.Variable(tf.random_normal([784,256]))
b1 = tf.Variable(tf.random_normal([256]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([256,256]))
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.Variable(tf.random_normal([256,10]))
b3 = tf.Variable(tf.random_normal([10]))

hypothesis = tf.matmul(L2,W3) + b3

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))

train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    training_epochs = 15
    batch_size = 100

    for epoch in range(training_epochs):
        avg_loss = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            feed_dict = {X:batch_xs, Y:batch_ys}
            c, _ = sess.run([loss, train], feed_dict=feed_dict)
            avg_loss += c / total_batch

        print('Epoch: ', '%02d' % (epoch+1), 'loss=', '{:.9f}'.format(avg_loss))

    print('Learning Finished!')

    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    feed = {X:mnist.test.images, Y:mnist.test.labels}
    print('Accuracy : ', sess.run(accuracy, feed_dict=feed))

    r = random.randint(0, mnist.test.num_examples-1)
    print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1))) # r:r+1 10개의 원핫 인코딩을 가져와라는뜻, 10개의 데이터를 읽어온다. 행단위의 단위 지정.
    # 행단위 전체를 읽어오겠다. 벡터데이터를 읽어온다.
    print("Prediction: ", sess.run(tf.arg_max(hypothesis, 1), feed_dict={X:mnist.test.images[r:r+1]}))

    plt.imshow(mnist.test.images[r:r+1].reshape(28,28))
    plt.show()