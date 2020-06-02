import tensorflow as tf

tf.set_random_seed(777)

x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [ [0],  [0],  [0],  [1],  [1],  [1] ]

X = tf.placeholder(tf.float32, shape=[None,2])
Y = tf.placeholder(tf.float32, shape=[None,1])

W = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# 가설함수:Logistic Regression(sigmoid 함수 적용)
hypothesis = tf.sigmoid(tf.matmul(X,W) + b)

# 손실함수 (분류에서는 크로스 엔트로피 에러 수식을 사용 하고있다.)
# c(H(x),y) = - ylog(H(x)) - (l - y)log(1 - H(x))
loss = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))  # 로그함수 tf.log(),  (H(x)):가설함수

# 경사하강법
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

pridict = tf.cast(hypothesis > 0.5, dtype=tf.float32)  # 자료형을 캐스팅하겠다. 자료형변환. true일 경우 실수로 형변환 한다. 1.0으로 변환
# false면 0.0 으로 실수로 변환
accuracy = tf.reduce_mean(tf.cast(tf.equal(pridict, Y), dtype=tf.float32)) # 정확도.   tf.equal() 정확도계산.

with tf.Session() as sess:  # sess = tf.Session()  파일을 읽어오는 문법 with 따로 close를 안해도 된다. open의 의미와 동일
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        loss_val, _ = sess.run([loss, train], feed_dict={X:x_data, Y:y_data})

        if step % 200 ==0:
            print(step, loss_val)

    h, p, a = sess.run([hypothesis, pridict, accuracy], feed_dict={X    :x_data, Y:y_data})
    print('\nHypothesis:', h,'\npridict:', p,'\naccuracy:', a)
    print('4시간 수업, 2시간 자율학습', sess.run(pridict, feed_dict={X:[[4,2]]}))