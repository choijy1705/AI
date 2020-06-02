import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

# 8행 5열
xy= np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973],
              [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
              [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
              [816, 820.958984, 1008100, 815.48999, 819.23999],
              [819.359985, 823, 1188100, 818.469971, 818.97998],
              [819, 823, 1198100, 816, 820.450012],
              [811.700012, 815.25, 1098100, 809.780029, 813.669983],
              [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])

# 전처리를 하지 않으면 문제점이 발생할 수 있다.

x_data = xy[:, :-1]
y_data = xy[:,[-1]]

X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([4,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# 가설함수
hypothesis = tf.matmul(X,W)+b # 입력이 여러개일 경유 행렬을 이용하여 한번에 계산하기 위하여 matmul 을 사용한다.
loss = tf.reduce_mean(tf.square(hypothesis - Y))
train = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)
# 한컬럼의 값이 다른 컬럼과 많이 다르기 때문에 learning_rate과 상관없이 발산하게 된다. 전처리가 필요하다.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        loss_val, hy_val, _ = sess.run([loss, hypothesis, train], feed_dict={X:x_data, Y:y_data})

        if step % 100 == 0:
            print(step, "Loss:", loss_val, "\nPrediction:\n",hy_val)