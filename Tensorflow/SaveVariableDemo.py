# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/9/2816:24
# File：  SaveVariableDemo.py
# Engine：PyCharm


import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np

tf.compat.v1.disable_eager_execution()

#
# Weight = tf.Variable([[4, 5, 7],
#                       [8, 9, 5.4]], dtype=tf.float32, name='weight')
# bias = tf.Variable([4, 7, 2], dtype=tf.float32, name='bias')

Weight01 = tf.Variable(np.zeros([2, 3]), dtype=tf.float32, name='weight')
Bias01 = tf.Variable(np.arange(3), dtype=tf.float32, name='bias')

saver = tf.train.Saver()
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
# save_path = saver.save(sess, './myNet/testSave01.ckpt')
# print(save_path)
saver.restore(sess, './myNet/testSave01.ckpt')
print(sess.run(Weight01))
print(sess.run(Bias01))


