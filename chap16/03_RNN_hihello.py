import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

# Teach hello : hihell(input) -> ihello(output)

idx2char = ['h', 'i', 'e', 'l', 'o']

x_data = [[0,1,0,2,3,3]] # hihell
x_one_hot = [[
                [1,0,0,0,0], # h -> 0
                [0,1,0,0,0], # i -> 1
                [1,0,0,0,0], # h -> 0
                [0,0,1,0,0], # e -> 2
                [0,0,0,1,0], # l -> 3
                [0,0,0,1,0]  # l -> 3
            ]]

y_data = [[1,0,2,3,3,4]] # (h)ihello

input_dim = 5
hidden_size = 5 # 출력값의 열 갯수를 의미한다.
batch_size = 1
sequence_length = 6
learning_rate = 0.1

X = tf.placeholder(tf.float32, [None,sequence_length,input_dim]) # [None,sequence의 length, input의 차원]
Y = tf.placeholder(tf.int32, [None, sequence_length]) # Y label

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
initial_state = cell.zero_state(batch_size,tf.float32) # batch_size만큼 cell을 0으로 채우겠다.
output, _state = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)

# FC layer

X_for_fc = tf.reshape(output, [-1, hidden_size]) # 입력데이터
'''
fc_w = tf.get_variable("fc_w", [hidden_size,5])
fc_b = tf.get_variable("fc_b",[5])

hypothesis = tf.matmul(X_for_fc, fc_w) + fc_b
'''

hypothesis = tf.contrib.layers.fully_connected(inputs=X_for_fc, num_outputs=5, activation_fn=None)

outputs = tf.reshape(hypothesis, [batch_size, sequence_length,hidden_size])
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)

loss = tf.reduce_mean(sequence_loss)

# FC 에연결하는 코드 아예 모듈처럼 생각하는 것이 좋다.

train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        loss_val, _ = sess.run([loss, train], feed_dict={X:x_one_hot, Y:y_data})

        result = sess.run(prediction, feed_dict={X:x_one_hot})
        print(i, 'loss:', loss_val, "prediction :", result, "Y_data : ", y_data)

        result_str = [idx2char[c] for c in np.squeeze(result)] # 1차원화 해서 결과를 반환해준다.
        print("\nPrediction str : ", ''.join(result_str))


