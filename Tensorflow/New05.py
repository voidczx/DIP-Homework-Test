# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/9/1820:30
# File：  New05.py
# Engine：PyCharm


import tensorflow.compat.v1 as tf
#import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


def main():
    tf.compat.v1.disable_eager_execution()
    xData = np.linspace(-5, 5, 1000, dtype=np.float32)[:, np.newaxis]
    noise = np.random.normal(0.0, 0.05, xData.shape)
    yData = np.square(xData) - 0.5 + noise
    with tf.name_scope('inputs'):
        xs = tf.placeholder(tf.float32, [None, 1], name='xInput')
        ys = tf.placeholder(tf.float32, [None, 1], name='yInput')

    layerOne = AddLayer(xs, 1, 10,nLayer=1, activitionFunction=tf.nn.relu)
    with tf.name_scope('Prediction'):
        prediction = AddLayer(layerOne, 10, 1,nLayer=2, activitionFunction=None)
        tf.summary.histogram('/prediction', prediction)
    with tf.name_scope('Loss'):
        loss = tf.reduce_mean(tf.reduce_sum(tf.abs(ys - prediction), reduction_indices=[1]))
        tf.summary.scalar('loss', loss)
    with tf.name_scope('Train'):
        train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    init = tf.initialize_all_variables()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(xData, yData)

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        sess.run(init)
        writer = tf.summary.FileWriter('logs/', sess.graph)
        for i in range(10000):
            sess.run(train, feed_dict={xs: xData, ys: yData})
            if i % 500 == 0:
                #print(sess.run(loss, feed_dict={xs: xData, ys: yData}))
                result = sess.run(prediction, feed_dict={xs: xData, ys: yData})
                boardResult = sess.run(merged, feed_dict={xs: xData, ys: yData})
                writer.add_summary(boardResult)
                try:
                    ax.lines.remove(lines[0])
                except Exception:
                    pass
                lines = ax.plot(xData, result, 'r-', lw=5)
                plt.pause(0.2)


def AddLayer(inputs, inSize, outSize, nLayer, activitionFunction=None):
    layerName = 'layer%s' % nLayer
    with tf.name_scope('layers'):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([inSize, outSize]), name='Weight')
            tf.summary.histogram(layerName+'/Weights', Weights)
        with tf.name_scope('Biases'):
            biases = tf.Variable(tf.zeros([1, outSize]) + 0.1, name='Biase')
            tf.summary.histogram(layerName+'/Biases', biases)
        with tf.name_scope('Results'):
            result = tf.matmul(inputs, Weights) + biases
            if activitionFunction == None:
                outputs = result
            else:
                outputs = activitionFunction(result)
            return outputs


if __name__ == '__main__':
    main()
