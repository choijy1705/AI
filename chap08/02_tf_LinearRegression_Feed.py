import tensorflow as tf

tf.set_random_seed(777)

# y = Wx + b
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

X = tf.placeholder(tf.float32, shape=[None])  # 엑스와 와잉의 변수를 선언하고 이 변수는 좀따가 텐서를 수행할때 그때 값을 넣어 ㅌ\줄테니까 이 변수를 정의하고 사용할 준비를 하고잇다.라는 의미
# 추후에 값을 셋팅해서 실행하는 순간에 값을 전달해주면서 활성화시켜줄테니까 그렇게 알고잇어 함수. 독립화시켜주는 개념이 강함.
# float32 : 실수의 값으로 담아줘. shape=[None] : 형태자체는 여기서 지정하지 않고 전달받은 값으로 처리해라
Y = tf.placeholder(tf.float32, shape=[None])

# 가설함수 H(x) 정의
hypothsis = X * W + b

# Loss 함수
loss = tf.reduce_mean(tf.square(hypothsis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

session = tf.Session()
session.run(tf.global_variables_initializer())

for step in range(2001):
   loss_val, w_val, b_val, _ = session.run([loss, W, b, train], feed_dict={X:[1,2,3], Y:[1,2,3]})
    # 활성화 시키고싶은 텐서들을 리스트로 알려주면 된다.
    # feed_dict={X:[1,2,3], Y:[1,2,3]} : 우리가 위에서 정의 해주었던 tf.placeholder에 정의되었던 변수를 활성화시켜주는 매개변수.
                                        # 딕셔너리 자료형으로 전달해준다. 키 벨류값
   # 활성화를 시키되 피드백 받을 필요가 없을 경우 언더바를 사용 ' _ '
   if step % 20 == 0:
       print(step, loss_val, w_val, b_val)

print(session.run(hypothsis, feed_dict={X:[5]}))   # 예측도 해보자.
print(session.run(hypothsis, feed_dict={X:[2.5]}))
print(session.run(hypothsis, feed_dict={X:[1.5, 3.5]}))