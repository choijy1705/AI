import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

tf.set_random_seed(777)

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

char_set = list(set(sentence))
char_dic = {w:i for i, w in enumerate(char_set)}

data_dim = len(char_set)
hidden_size = len(char_set)
num_classes = len(char_set)
sequence_length = 10
learning_rate = 0.1

def lstm_cell():
    cell = rnn.BasicLSTMCell(num_units=hidden_size)
    return cell

dataX = []
dataY = []
for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i+sequence_length]
    y_str = sentence[i+1:i+sequence_length+1]
    # print(i, x_str, '->', y_str) # 170

    x = [char_dic[c] for c in x_str]
    y = [char_dic[c] for c in y_str]

    dataX.append(x)
    dataY.append(y)

batch_size = len(dataX)
# print(batch_size) # 170

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

# One-hot encoding
X_one_hot = tf.one_hot(X, num_classes) # 2차원이 3차원으로 변하게 된다.

# cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
cell = lstm_cell()
multi_cells = rnn.MultiRNNCell([cell] * 2) # 은닉층을 잘연결해서 만들어주는 기능 은닉층의 갯수만 입력해주면된다.

outputs, state = tf.nn.dynamic_rnn(multi_cells, X_one_hot, dtype=tf.float32)

# FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y,weights= weights)

loss = tf.reduce_mean(sequence_loss)

train = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(500):
        _, loss_val, hypothesis = sess.run([train, loss, outputs], feed_dict={X:dataX, Y:dataY})

        for j, result in enumerate(hypothesis):
            index = np.argmax(result, axis=1)
            print(i,j,''.join([char_set[t] for t in index]), loss_val)

        hypothesis = sess.run(outputs, feed_dict={X:dataX})
        for i, result in enumerate(hypothesis):
            index = np.argmax(result, axis=1)
            if i is 0:
                print(''.join([char_set[t] for t in index]), end ='')
            else:
                print(char_set[index[-1]], end='') # end를 통하여 줄바꿈 되지 않도록 해준다.
