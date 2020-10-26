# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/10/719:49
# File：  AutoLoadDemo.py
# Engine：PyCharm

#import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
import numpy as np

tf.compat.v1.disable_eager_execution()
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.Session()

loader = tf.train.import_meta_graph('./AutoEncoder/AutoEncoder.ckpt.meta')
loader.restore(sess, tf.train.latest_checkpoint('./AutoEncoder'))

graph = tf.get_default_graph()
xs = graph.get_tensor_by_name('Input/inputX:0')
dropProb = graph.get_tensor_by_name('Input/inputDrop:0')
prediction = graph.get_tensor_by_name('Decoder02/Layer/layer:0')

predictionImage = sess.run(prediction, feed_dict={xs: mnist.test.images, dropProb: 1.0})

image, order = plt.subplots(2, 10, figsize=(10, 2))
for i in range(10):
    order[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)), cmap='gray')
    order[1][i].imshow(np.reshape(predictionImage[i], (28, 28)), cmap='gray')
image.show()
plt.show()


