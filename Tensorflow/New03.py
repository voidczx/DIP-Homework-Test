# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/9/1717:09
# File：  New03.py
# Engine：PyCharm


import tensorflow.compat.v1 as tf


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()

    v1 = tf.Variable(0, name='counter')
    one = tf.constant(1)
    temp = tf.add(v1, one)
    process = tf.assign(v1, temp)
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(v1))
        for i in range(3):
            sess.run(process)
            print(sess.run(v1))
