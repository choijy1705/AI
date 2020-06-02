import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float)  # 파일을 읽어올수 있는 함수

#print(xy) # 배열형으로 반환

score = xy[:, 0:-1] #행전체 :, 열
final_term = xy[:, [-1]]

X = tf.placeholder(tf.float32, shape=[None,3]) #  행은 제한을 두지 않겠다, 피쳐는 3개  [None,3] 행렬 표현
Y = tf.placeholder(tf.float32, shape=[None,1]) # 열은 1열

W = tf.Variable(tf.random_normal([3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# 가설함수
hyptosis = tf.matmul(X,W)+b  # 내적 구하는 함수 matmul()

# 손실함수
loss = tf.reduce_mean(tf.square(hyptosis - Y))

# 경사하강법
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(18001):
    loss_val, hy_val, _ = sess.run([loss, hyptosis, train], feed_dict={X:score, Y:final_term})

    if step % 100 == 0:
        print(step, 'Loss:', loss_val, '\nhyptosis :\n', hy_val)

print('Test-Set:', sess.run(hyptosis,feed_dict={X:[[75., 70., 72.]]}))
