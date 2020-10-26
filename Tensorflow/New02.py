# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/9/1316:46
# File：  New02.py
# Engine：PyCharm


import tensorflow.compat.v1 as tf
import numpy as np


def main():
    tf.compat.v1.disable_eager_execution()

    matrix01 = tf.constant([[3, 3, 5]])
    matrix02 = tf.constant([[4], [5], [6]])
    product = tf.matmul(matrix01, matrix02)

    sess = tf.Session()
    print(sess.run(product))
    sess.close()


if __name__ == '__main__':
    main()
