import tensorflow as tf

tf.set_random_seed(777)

x_data = [1,2,3]
y_data = [3,6,9]

W = tf.Variable(tf.random_normal([1]), name='weight')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 가설함수
hypothsis = X * W

# 손실함수
loss = tf.reduce_mean(tf.square(hypothsis - Y))

# 직접 구현한 경사하강법 알고리즘
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)   # 담아주겠다.assign()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(1001):
    sess.run(update, feed_dict={X:x_data, Y:y_data})

    if step % 20 == 0:
        print(step, sess.run(loss, feed_dict={X:x_data, Y:y_data}),sess.run(W))
