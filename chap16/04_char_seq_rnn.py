# character sequence RNN
import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

sample = 'if you want you'
idx2char = list(set(sample)) # set(): 데이터의 중복을 허용하지 않는다. 중복되어지는 것들은 하나만 꺼내와 데이터를 구성하게된다
char2idx = {c:i for i, c in enumerate(idx2char)} # char -> index

# hyper parameters
dic_size = len(char2idx)     # RNN input size(one hot size)
hidden_size = len(char2idx)  # RNN output size
num_classes = len(char2idx)  # final output size
batch_size = 1               # one sample data, one batch
sequence_length = len(sample) - 1 # number of lstm rolling(unit #)
learning_rate = 0.1

sample_idx = [char2idx[c] for c in sample] # sample 의 순서대로 맞춰준다
x_data = [sample_idx[:-1]] # 마지막 항목 제외
y_data = [sample_idx[1:]] # 제일 처음 항목만 뺀 나머지 부분이 원하는 결과값이다.

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

x_one_hot = tf.one_hot(X, num_classes)

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=initial_state, dtype=tf.float32)

# FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)
outputs = tf.reshape(outputs,[batch_size, sequence_length, num_classes])
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y,weights=weights)

loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(200):
        loss_val, _ = sess.run([loss, train], feed_dict={X:x_data, Y:y_data})
        result = sess.run(prediction, feed_dict={X:x_data})

        result_str = [idx2char[c] for c in np.squeeze(result)]
        print(i, "loss: ", loss_val,"Prediction: ", ''.join(result_str))

