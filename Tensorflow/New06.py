# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/9/2412:18
# File：  New06.py
# Engine：PyCharm


#import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow.examples.tutorials.mnist import input_data


def ComputeAccurate(testX, testY):
    global prediction
    predictionY = sess.run(prediction, feed_dict={xs: testX})
    correct = tf.equal(tf.argmax(predictionY, 1), tf.argmax(testY, 1))
    mean = tf.reduce_mean(tf.cast(correct, tf.float32))
    result = sess.run(mean, feed_dict={xs: testX, ys: testY})
    return result


def AddLayer(input, inSize, outSize, nLayer, activitionFuntion=None):
    LayerName = 'layer%s' % nLayer
    Weight = tf.Variable(tf.random_normal([inSize, outSize]))
    biasis = tf.Variable(tf.zeros([1, outSize]) + 0.1)
    result = tf.matmul(input, Weight) + biasis
    if activitionFuntion == None:
        output = result
    else:
        output = activitionFuntion(result)
    return output


tf.compat.v1.disable_eager_execution()
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])
prediction = AddLayer(xs, 784, 10, 1, tf.nn.softmax)
loss = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))
train = tf.train.GradientDescentOptimizer(0.8).minimize(loss)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
for i in range(0, 5000):
    keyX, keyY = mnist.train.next_batch(100)
    sess.run(train, feed_dict={xs: keyX, ys: keyY})
    if i % 200 == 0:
        print(ComputeAccurate(keyX, keyY))




