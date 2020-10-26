# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/9/1216:15
# File：  New01.py
# Engine：PyCharm

import tensorflow.compat.v1 as tf
import numpy as np

def main():
    tf.compat.v1.disable_eager_execution()

    ### Data Build Begin ###
    xData = np.random.rand(100).astype(np.float32)
    yData = xData * 0.1 + 0.3
    ### Data Build End ###


    ### Tensorflow Structure Begin ###
    weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    bias = tf.Variable(tf.zeros([1]))
    y = xData * weight + bias
    loss = tf.reduce_mean(tf.square(y - yData))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)
    inits = tf.initialize_all_variables()
    ### Tensorflow Structure End ###

    sess = tf.Session()
    sess.run(inits)
    for step in range(0, 201):
        sess.run(train)
        if step % 20 == 0:
            print(step, ' ', sess.run(weight), sess.run(bias))


if __name__ == '__main__':
    main()
