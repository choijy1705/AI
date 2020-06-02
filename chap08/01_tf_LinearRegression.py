# tensorflow를 이용한 Linear Regression(선형 회귀)
import tensorflow as tf

tf.set_random_seed(777)  # 똑같이 랜덤한 값을 뽑아내는 함수

# X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# 가설함수 : y = Wx + b  최적의 w 와 b 값을 구하자
W = tf.Variable(tf.random_normal([1]), name='weight') # 값을 업그레이드 해야하는데 값이 변경되어야 한다. 이 함수를 이용해서 변수를 하나의 값을 생성해서 가져서
                                                      # 변수에 저장해라  name 매개변수는 이 변수에 이름을 부여해 줄 수 도 있다. 불러올 수도 있다.
                                                      # ([1]) : 한개의 값만 넣는다.
b = tf.Variable(tf.random_normal([1]), name='bias')  # 최적의 값을 찾기위한 tensorflow용 변수를 선언했다.

hypothesis = x_train * W + b  # 예측값

# Loss(cost) function
loss = tf.reduce_mean(tf.square(hypothesis - y_train))  # 예측값 - 실제값 제곱(tf.square)/ tf.reduce_mean(평균화) : 선형회귀 민스퀘어에러

# Lost Minimise
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)   # GradientDescentOptimizer : 경사하강법
train = optimizer.minimize(loss)  # 경사하강법의 최소값(minimize)을 찾아가자

session = tf.Session()  # tensorflow 활성화 방법
session.run(tf.global_variables_initializer())  # w와 b는 우리가 변수를 지정만 한 건데 이걸 활성화를 시켜야한다. 이 해당 변수들을 생성해야하는
# 과정이 필요하다 그 과정이  global_variables_initializer 라는 함수. 호출해주면 위에서 선언되어있던 변수를 실질적으로 메모리 할당해서
# 활성화 시켜주는 초기화 과정이 이 함수이다.

#학습
for step in range(2001):  # 2001번 반복 왜? 메세지 출력 때문에 그렇다.
    session.run(train)

    if step % 20 == 0:  # 20번에 한번씩 계산한 결과를 출력해보겠다.
        print(step, session.run(loss), session.run(W), session.run(b))