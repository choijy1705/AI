import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)

x_train_data = xy[:-10,:-1] # 10개의 데이터를 제외한 나머지 데이터 가져오기, 열은 마지막 데이터만 남기고 가져오기
y_train_data = xy[:-10,[-1]]

x_test_data = xy[-10:,:-1]
y_test_data = xy[-10:,[-1]]

X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1]) # 0~6 : 7 classfication // index로 동작하여야 하기 때문에 반드시 int32 형으로 선언되어야 한다.

# shape
Y_one_hot = tf.one_hot(Y, 7) # 3 차원형태 [[[1,0,0,0,0,0,0],[],...,[]],[[],[]]..]
Y_one_hot = tf.reshape(Y_one_hot,[-1, 7])  # 7열로 만들고 거기에 맞게 행을 정렬하라는 뜻 , 2차원 형태로 바뀌게 된다


W = tf.Variable(tf.random_normal([16, 7]), name = 'weight')
b = tf.Variable(tf.random_normal([7]), name = 'bias')

# 가설함수
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# 손실함수 (분류에서는 크로스 엔트로피 에러 수식을 사용 하고있다.)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= tf.matmul(X,W)+b , labels=Y_one_hot))
# logits 선형의 방정식을 , label은 cross entropy를 계산할수 있는 정답의 인코딩 값

# 경사하강법 알고리즘
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # 총갯수에 대하여 1의 갯수이기 때문에 정확도가 된다.

# Machine learning

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(train, feed_dict={X:x_train_data, Y:y_train_data}) # placehold를 사용하였기때문에 데이터도 전달해줘야 한다.

        if(step % 100 == 0):
            l, a = sess.run([loss, accuracy], feed_dict={X:x_train_data, Y:y_train_data})
            print("\nStep:{:5}\tLoss:{:.3f}\tAcc:{:.2%}".format(step, l, a))

    pred = sess.run(prediction, feed_dict={X:x_train_data})

    for p, y in zip(pred, y_train_data.flatten()): # zip함수는 지퍼가 왼쪽과 오른쪽의 틈이 합쳐지는 형태 첫번째와 두번째 매개변수를 맵핑하면서 반환해준다.
        # flatten함수는 y_train_data와 pred를 비교하기위하여 2차원을 1차원으로 변경해주고 있다. 차원을 맞춰 주고 있다.
        print("[{}] Prediction:{}, Y:{}".format(p==int(y), p, int(y)))

    print("\n\n테스트 데이터 적용")

    # test data
    pred = sess.run(prediction, feed_dict={X: x_test_data})

    for p, y in zip(pred, y_test_data.flatten()):  # zip함수는 지퍼가 왼쪽과 오른쪽의 틈이 합쳐지는 형태 첫번째와 두번째 매개변수를 맵핑하면서 반환해준다.
        # flatten함수는 y_train_data와 pred를 비교하기위하여 2차원을 1차원으로 변경해주고 있다. 차원을 맞춰 주고 있다.
        print("[{}] Prediction:{}, Y:{}".format(p == int(y), p, int(y)))