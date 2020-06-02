import tensorflow as tf

tf.set_random_seed(777)

x_data = [1,5,10]
y_data = [5,25,50]

W = tf.Variable(-3.0)  # -3으로 초기화

# 가설함수
hypothsis = x_data * W

# loss함수
loss = tf.reduce_mean(tf.square(hypothsis - y_data))

# 경사하강법
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run(loss), sess.run(W))
    sess.run(train)