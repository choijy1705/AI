import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
# print(xy)

# 훈련용(749명)
x_train_data = xy[:-10, : -1] # -는 뒤에서부터 카운트, 맨끝 데이터를 제외하고 다 사용하겠다는 뜻
y_train_data = xy[:-10, [-1]] # 맨끝 결과 데이터

# 테스트용(10명)
x_test_data = xy[-10:,:-1]
y_test_data = xy[-10:,[-1]]

X = tf.placeholder(tf.float32, shape=[None,8])
Y = tf.placeholder(tf.float32, shape=[None,1])

W = tf.Variable(tf.random_normal([8,1]))
b = tf.Variable(tf.random_normal([1]))

# 가설함수
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# 손실함수 (분류에서는 크로스 엔트로피 에러 수식을 사용 하고있다.)
# c(H(x),y) = - ylog(H(x)) - (l - y)log(1 - H(x))
loss = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))  # 로그함수 tf.log(),  (H(x)):가설함수

# 경사하강법
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, Y), dtype=tf.float32)) # 정확도.   tf.equal() 정확도계산.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(20001):
        loss_val, _ = sess.run([loss, train], feed_dict={X:x_train_data, Y:y_train_data})

        if step % 200 ==0:
            print(step, loss_val)

    h, p, a = sess.run([hypothesis, predict, accuracy], feed_dict={X:x_train_data, Y:y_train_data})
    print('\nHypothesis:', h,'\npredict:', p,'\naccuracy:', a)

    # 추론(10명의 데이터)
    print("=============test================")
    h, p, y = sess.run([hypothesis, predict, Y], feed_dict={X: x_train_data, Y: y_train_data})
    print('\nHypothesis:', h, '\npredict:', p, '\ny:', y)


