# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/9/1217:10
# File：  New01Re.py
# Engine：PyCharm

import tensorflow.compat.v1 as tf
import numpy as np


def main():
    tf.compat.v1.disable_eager_execution()

    ### Data Build Begin ###
    xData = np.random.rand(500)
    yData = xData * 1.4 + 5
    ### Data Build End ###


    ### Tensorflow Structure Begin ###
    weight = tf.Variable(tf.random_uniform([1], -5.0, 5.0))
    biasis = tf.Variable(tf.zeros([1]))
    y = weight * xData + biasis
    loss = tf.reduce_mean(tf.square(y - yData))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)
    initial = tf.initialize_all_variables()
    ### Tensorflow Structure End ###


    sess = tf.Session()
    sess.run(initial)
    for step in range(0, 1001):
        sess.run(train)
        if step % 50 == 0:
            print(step, sess.run(weight), sess.run(biasis))


if __name__ == '__main__':
    main()
