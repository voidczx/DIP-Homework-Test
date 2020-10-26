# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/10/1016:31
# File：  graphDemo.py
# Engine：PyCharm

# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()

g1 = tf.Graph()
with g1.as_default():
    v1 = tf.constant((4.2, 1.8))
    v2 = tf.constant((7.8, 9.5))
    with tf.name_scope('X1'):
        x1 = tf.Variable(tf.constant(0.1, shape=(2, 2)), name='x1')

g2 = tf.Graph()
with g2.as_default():
    v1 = tf.constant((6.6, 6.6))
    v2 = tf.constant((0, 0), dtype=tf.float32)
    with tf.name_scope('X1'):
        x1 = tf.Variable(tf.constant(0.5, shape=(1, 3)), name='x1')

with tf.Session(graph=g2) as sess:
    init = tf.initialize_all_variables()
    summ = v1 + v2
    sess.run(init)
    print(sess.run(summ))
    print(sess.run(x1))
