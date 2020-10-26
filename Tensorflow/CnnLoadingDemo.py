# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/10/612:24
# File：  CnnLoadingDemo.py
# Engine：PyCharm

#import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"  # 选择哪一块gpu,如果是-1，就是调用cpu
config = tf.ConfigProto()  # 对session进行参数配置
config.allow_soft_placement = True  # 如果你指定的设备不存在，允许TF自动分配设备
config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 分配百分之七十的显存给程序使用，避免内存溢出，可以自己调整
config.gpu_options.allow_growth = True  # 按需分配显存，这个比较重要
sess = tf.Session(config=config)

tf.compat.v1.disable_eager_execution()
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

loader = tf.train.import_meta_graph('./MnistNetWork/CnnNetWork.ckpt.meta')
loader.restore(sess, tf.train.latest_checkpoint('./MnistNetWork'))

graph = tf.get_default_graph()
xs = graph.get_tensor_by_name('Input/inputX:0')
prediction = graph.get_tensor_by_name('NormalLayer02/prediction:0')
dropProb = graph.get_tensor_by_name('Input/Placeholder:0')

correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(mnist.test.labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
print(sess.run(accuracy, feed_dict={xs: mnist.test.images, dropProb: 1.0}))
sess.close()
