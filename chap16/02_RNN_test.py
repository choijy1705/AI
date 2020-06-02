import tensorflow as tf
import numpy as np

# I       [1,0,0,0]
# Work    [0,1,0,0]
# at      [0,0,1,0]
# google  [0,0,0,1]

# I work at goolge. [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]

# I google at work. [[1,0,0,0], [0,0,0,1], [0,0,1,0], [0,1,0,0]]

# 3차원의 결과로 데이터를 담아준다.
inputs = np.array([
                    [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]],
                    [[1,0,0,0], [0,0,0,1], [0,0,1,0], [0,1,0,0]]
                   ])


tf.reset_default_graph()
tf.set_random_seed(777)
tf_inputs = tf.constant(inputs, dtype=tf.float32)

rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=4)
output, state = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=tf_inputs, dtype=tf.float32)
variables_names = [v.name for v in tf.trainable_variables()]

print(output)
print(state)
print("weight : ")
for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    print(v)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output_run, state_run = sess.run([output, state])
    print("output values : ", output_run)
    print("state values : ", state_run)

    print("weight : ")
    values = sess.run(variables_names)
    for k, v in zip(variables_names, values):
        print(k,v)



