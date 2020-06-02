import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random
import matplotlib.pyplot as plt

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True) # 결과의 값을 one hot encoding방식으로 다운로드 해준다.

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.random_normal([784,10]))
b = tf.Variable(tf.random_normal([10]))

hypothesis = tf.nn.softmax(tf.matmul(X,W) + b)
loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1)), tf.float32))

# parameters
training_epochs = 50
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_loss = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            loss_val, _ = sess.run([loss, train], feed_dict={X:batch_xs, Y:batch_ys})

            avg_loss += loss_val / total_batch

        print('Epoch:', '%04d' % (epoch+1),'loss=', '{:.9f}'.format(avg_loss))

    print("Accuracy : ", accuracy.eval(session=sess, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))

    # 시각화
    r = random.randint(0, mnist.test.num_examples - 1)
    print("label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))
    print("prediction: ", sess.run(tf.argmax(hypothesis, 1),feed_dict={X:mnist.test.images[r:r+1]}))

    plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap='Greys')
    plt.show()