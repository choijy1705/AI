import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, shape=[None,784])
Y = tf.placeholder(tf.float32, shape=[None,10])

keep_prob = tf.placeholder(tf.float32)


W1 = tf.get_variable("W1", shape=[784,512], initializer=tf.contrib.layers.xavier_initializer()) # xavier 변수를 불러온다. 특정 데이터에 구애받지 않고 잘 적용된다.
# 루트 n분의 1 = 입력노드의 갯수(사비르가 제안한 알고리즘)
b1 = tf.Variable(tf.random_normal([512]))
L1 = tf.nn.relu(tf.matmul(X, W1)+b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob) # keep_prob : 전체중에서 몇개를 활성화 노드로 사용할지 지정해주는 파라메터 ex) 100% : 1, 70% : 0.7

W2 = tf.get_variable("W2", shape=[512,512], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.relu(tf.matmul(L1, W2)+b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[512,256], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([256]))
L3 = tf.nn.relu(tf.matmul(L2, W3)+b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable("W4", shape=[256,256], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([256]))
L4 = tf.nn.relu(tf.matmul(L3, W4)+b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable("W5", shape=[256,10], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))

# 가설
hypothesis = tf.matmul(L4, W5) + b5


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))

train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    training_epochs = 15
    batch_size = 100

    for epoch in range(training_epochs):
        avg_loss = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            feed_dict = {X:batch_xs, Y:batch_ys, keep_prob:0.7} # 학습시에는 0.7, 0.8 아무 값이나 상관없다. 결과를 확인할때는 1이여야만 한다.
            loss_val, _ = sess.run([loss, train], feed_dict=feed_dict)
            avg_loss += loss_val / total_batch

        print('Epoch: ', '%04d' % (epoch+1), 'loss = ', '{:.9f}'.format(avg_loss))

    print('Learning Finished!!')

    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    feed = {X: mnist.test.images, Y: mnist.test.labels, keep_prob:1} # 이때는 반드시 전체노드를 통하여 확인해봐야한다.
    print('Accuracy : ', sess.run(accuracy, feed_dict=feed))

    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label:",
          sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))  # r:r+1 10개의 원핫 인코딩을 가져와라는뜻, 10개의 데이터를 읽어온다. 행단위의 단위 지정.
    # 행단위 전체를 읽어오겠다. 벡터데이터를 읽어온다.
    print("Prediction: ", sess.run(tf.arg_max(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1],keep_prob:1}))

    plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28))
    plt.show()




