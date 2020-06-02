import tensorflow as tf
import numpy as np

tf.set_random_seed(777)
a = tf.constant(2)
b = tf.constant(3)
c = tf.multiply(a,b)

# sess = tf.Session()
sess = tf.InteractiveSession() # 디폴트 세션을 자동으로 할당해준다.

sess.run(tf.global_variables_initializer())

# print(sess.run(c))
print(c.eval())

image = np.array([[[[1],[2],[3]],
                   [[1],[2],[3]],
                   [[1],[2],[3]]]],
                 dtype=np.float32)
print(image.shape)


sess.close() # with 구문을 사용하지 않으면 반드시 close를 해줘야 한다.




