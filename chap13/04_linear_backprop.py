import tensorflow as tf

tf.set_random_seed(777)

x_data = [[1.],[2.],[3.]]
y_data = [[1.],[2.],[3.]]

X = tf.placeholder(tf.float32, shape=[None, 1])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.truncated_normal([1,1]))
b = tf.Variable(5.) # 임의의 초기값 설정

# 순전파
hypothesis = tf.matmul(X,W) + b

# diff
assert hypothesis.shape.as_list() == Y.shape.as_list() # 참일 경우 바로실행, 거짓일 경우 의미가 없어진다 이경우 assortion예외를 발생시켜 강제로 종료시켜버린다.
diff = (hypothesis - Y)

# 역전파
d_l1 = diff
d_b = d_l1
d_w = tf.matmul(tf.transpose(X), d_l1)

print(X,W,d_l1,d_w)

learning_rate = 0.1
step = [tf.assign(W, W-learning_rate*d_w), tf.assign(b, b-learning_rate*tf.reduce_mean(d_b))]

'''
a = tf.Variable(1)
a = a + 2
print("1 => ", a)
a = a + 1
print("2 => ", a)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

print(sess.run(a))
print(sess.run(b))
'''

mse = tf.reduce_mean(tf.square(Y - hypothesis))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(20001):
        if i % 100 == 0:
            print(i, sess.run([step, mse], feed_dict={X:x_data, Y:y_data}))
# 실수값은 전체적인 오차가 발생할 수 밖에 없다.
    print(sess.run(hypothesis, feed_dict={X:x_data, Y:y_data}))