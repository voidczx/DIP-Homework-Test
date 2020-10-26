# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/9/1717:33
# File：  New04.py
# Engine：PyCharm


# noinspection PyUnresolvedReferences
import tensorflow.compat.v1 as tf


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()

    input01 = tf.placeholder(tf.float32)
    input02 = tf.placeholder(tf.float32)
    input03 = tf.placeholder(tf.float32, [2, 2])

    output = tf.multiply(input01, input02)

    with tf.Session() as sess:
        print(sess.run(output, feed_dict={input01: [9.1],
                                          input02: [5.4]}))
