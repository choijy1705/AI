import tensorflow as tf
import numpy as np



inputs = np.array([[[1, 2]]]) # 데이터의 확장성을 위해서 리스트로, 3차원으로 지정해주자. 텐서플로우 이용할때 3차원으로 shape을 입력해줘야 RNN이 동작된다.

tf.reset_default_graph() # 텐서를 초기화해주는 역할,
tf.set_random_seed(777)
tf_inputs = tf.constant(inputs, dtype=tf.float32) # 1.0, 2.0 으로 변환되면서 상수의 주소를 반환해준다.constant는 상수형을 정의해준다 변수는 Variable

rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=3) # 셀을 만들어주는 함수, num_units은 출력을 몇개로 내보낼것인지에 대한 setting
output, state = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=tf_inputs, dtype=tf.float32) # rnn내부의 그래픽을 실질적으로 동작을 취하는 함수
# 사용하고자할 입력데이터 지정 inputs =  / 활성화함수까지 만들어져있다, output출력, state메모리셀 둘은 값이 같아야한다.
variables_names = [v.name for v in tf.trainable_variables()] # 훈련시킬 임의의 변수를 뽑아내고싶다.  tf.trainable_variables() 리스트 형태로 담아준다.

print(output) # shape=(1, 1, 3)면,행,렬 1행 3열  입력값이 한개라서 1개
print(state) # shape=(1, 3)
print("weight : ")
for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    print(v)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output_run, state_run = sess.run([output, state])
    print("output values : ", output_run) # [[[-0.9314169   0.75578666 -0.6819246 ]]] tanh를 거쳐서 나온 결과 값
    print("state values : ", state_run) # [[-0.9314169   0.75578666 -0.6819246 ]] output과 state의 값이 동일하다.

    print("weight : ")
    values = sess.run(variables_names)
    for k, v in zip(variables_names, values):
        print(k,v)

