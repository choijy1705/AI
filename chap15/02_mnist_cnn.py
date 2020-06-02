import tensorflow as tf
import random
import matplotlib as plt
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001

training_epochs = 15
batch_size = 100

X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1]) # 흑백일때는 1채널 컬러일때는 3채널
Y = tf.placeholder(tf.float32, shape=[None, 10])

W1 = tf.Variable(tf.random_normal([3,3,1,32],stddev=0.01)) # 필터의 갯수는 임의로 지정하면 된다.[행, 열, 채널, 필터의 수]

# layer 1
L1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding='SAME')
# print("conv2d : ", L1) # shape=(?, 28, 28, 32)
L1 = tf.nn.relu(L1) # convoulution의 결과는 연산을 통하여 나온 실수값, 그값이 음수일경우 0 양수일 경우 그 값 그대로 출력이 되도록 해준다.
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # 2x2의 풀링, dimension을 4차원으로 구성 (?, 14, 14, 32)
# 위의 padding과 pool 에서의 padding의 의미는 다르다.
# print("pooling : ", L1) # shape=(?, 14, 14, 32)

# layer 2
W2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01)) # 채널은 반드시 입력의 크기와 동일하여야 한다. layer1의 결과로 채널이 32가 된다.

L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME') # (?,14,14,64)
L2 = tf.nn.relu(L2) # convoulution의 결과는 연산을 통하여 나온 실수값, 그값이 음수일경우 0 양수일 경우 그 값 그대로 출력이 되도록 해준다.
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # 2x2의 풀링, dimension을 4차원으로 구성 (?, 7, 7, 64)

L2_flat = tf.reshape(L2, [-1,7*7*64]) # 3136

# Final FC 3136 inputs -> 10 outputs
W3 = tf.get_variable('W3', shape=[7*7*64, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]),name='bias')

# 가설 함수
hypothesis = tf.matmul(L2_flat, W3) + b

# 손실 함수
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))

# 알고리즘
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_loss = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            feed_dict = {X:batch_xs, Y:batch_ys}

            loss_val, _ = sess.run([loss, train], feed_dict= feed_dict)
            avg_loss += loss_val / total_batch

        print('Epoch: ', '%04d' % (epoch+1), 'loss = ', '{:.9f}'.format(avg_loss))

    print('Learning Finished!!')

    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    feed = {X: mnist.test.images, Y: mnist.test.labels}  # 이때는 반드시 전체노드를 통하여 확인해봐야한다.
    print('Accuracy : ', sess.run(accuracy, feed_dict=feed))

    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label:",
          sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))  # r:r+1 10개의 원핫 인코딩을 가져와라는뜻, 10개의 데이터를 읽어온다. 행단위의 단위 지정.
    # 행단위 전체를 읽어오겠다. 벡터데이터를 읽어온다.
    print("Prediction: ", sess.run(tf.arg_max(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))








