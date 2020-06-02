import tensorflow as tf
import matplotlib.pyplot as plt

tf.set_random_seed(777)

X = [1, 2, 3]
Y = [2, 4, 6]

W = tf.placeholder(tf.float32)

# 가설 함수
hypothesis = X * W

# 손실 함수
loss = tf.reduce_mean(tf.square(hypothesis - Y))

sess = tf.Session()

W_history = []
loss_history = []

for i in range(-30, 81):
    curr_W = i * 0.1
    curr_loss = sess.run(loss, feed_dict={W:curr_W})
    W_history.append(curr_W)  # 누적
    loss_history.append(curr_loss)

plt.plot(W_history, loss_history)
plt.show()