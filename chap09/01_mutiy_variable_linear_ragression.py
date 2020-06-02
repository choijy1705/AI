import tensorflow as tf

tf.set_random_seed(777)

quiz1 = [73.,93.,89.,96.,73.]  # 숫자뒤에 . 쓰면 자동으로 실수값으로 인식
quiz2 = [80.,88.,91.,98.,66.]
midterm = [75.,93.,90.,100.,70.]

finalterm = [152.,185.,180.,196.,142.]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name='weigth1')
w2 = tf.Variable(tf.random_normal([1]), name='weigth2')
w3 = tf.Variable(tf.random_normal([1]), name='weigth3')

b = tf.Variable(tf.random_normal([1]), name='bias')

# 가설함수
hyptosis = x1*w1 + x2*w2 + x3*w3 + b

# 손실함수
loss = tf.reduce_mean(tf.square(hyptosis - Y))

# 경사하강법
optimaizer = tf.train.GradientDescentOptimizer(learning_rate=0.00004)
train = optimaizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(13001):
    loss_val, hy_val, _ = sess.run([loss, hyptosis, train], feed_dict={x1:quiz1, x2:quiz2, x3:midterm, Y:finalterm})

    if step % 10 == 0:
        print(step, 'Loss :', loss_val, '\nPrediction :\n', hy_val)
# 왜 딱 0을 찾지 못하는가. 직선위에 다 지나는 선은 찾지 못하기 때문에 그나마 제일 근접한 기울기를 찾자고 하는것이다. 그래서
# 딱 맞아 떨어지지 않는것이다.