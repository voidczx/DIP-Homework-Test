# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/10/711:45
# File：  AutoEncoder.py
# Engine：PyCharm

#import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow.examples.tutorials.mnist import input_data
from pathlib import Path


tf.compat.v1.disable_eager_execution()
mnist = input_data.read_data_sets('./MNIST_data', one_hot=False)


inputSize = 784
outputSize = 10
hidden01Size = 256
hidden02Size = 128
batchSize = 100
dropProb = 0.9


def FeedDict(trainFlag):
    if trainFlag:
        x, y = mnist.train.next_batch(batchSize)
        z = dropProb
    else:
        x, y = mnist.test.images, mnist.test.labels
        z = 1.0
    return {xs: x, drop: z}


def GetWeight(shape):
    with tf.name_scope('Weight'):
        weight = tf.truncated_normal(shape, stddev=0.1)
        tf.summary.histogram('weightHist', weight)
        return tf.Variable(weight, name='weight')


def GetBias(shape):
    with tf.name_scope('Bias'):
        bias = tf.constant(0.1, shape=shape)
        tf.summary.histogram('biasHist', bias)
        return tf.Variable(bias, name='bias')


def GetNormalLayer(input, weight, bias, nLayer=None, activation=None):
    with tf.name_scope('Layer', str(nLayer)):
        layer = tf.matmul(input, weight) + bias
        if activation is None:
            result = layer
        else:
            result = activation(layer, name='layer')
        return result


with tf.name_scope('Input'):
    xs = tf.placeholder(tf.float32, [None, inputSize], name='inputX')
    ys = tf.placeholder(tf.float32, [None, outputSize], name='inputY')
    drop = tf.placeholder(tf.float32, name='inputDrop')

with tf.name_scope('Encoder01'):
    weight01 = GetWeight([inputSize, hidden01Size])
    bias01 = GetBias([hidden01Size])
    layer01 = GetNormalLayer(xs, weight01, bias01, 1, tf.nn.sigmoid)
    dropLayer01 = tf.nn.dropout(layer01, drop)

with tf.name_scope('Encoder02'):
    weight02 = GetWeight([hidden01Size, hidden02Size])
    bias02 = GetWeight([hidden02Size])
    layer02 = GetNormalLayer(dropLayer01, weight02, bias02, 2, tf.nn.sigmoid)
    dropLayer02 = tf.nn.dropout(layer02, drop)

with tf.name_scope('Decoder01'):
    weight03 = GetWeight([hidden02Size, hidden01Size])
    bias03 = GetBias([hidden01Size])
    layer03 = GetNormalLayer(dropLayer02, weight03, bias03, 3, tf.nn.sigmoid)
    dropLayer03 = tf.nn.dropout(layer03, drop)

with tf.name_scope('Decoder02'):
    weight04 = GetWeight([hidden01Size, inputSize])
    bias04 = GetBias([inputSize])
    prediction = GetNormalLayer(dropLayer03, weight04, bias04, 4, tf.nn.sigmoid)

with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.square(prediction - xs), name='theLoss')
    tf.summary.scalar('loss', loss)

with tf.name_scope('Train'):
    train = tf.train.RMSPropOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
saver = tf.train.Saver()
merge = tf.summary.merge_all()
sess = tf.Session()
trainWriter = tf.summary.FileWriter('./logs/auto', sess.graph)

if Path('./AutoEncoder').exists():
    saver.restore(sess, './AutoEncoder/AutoEncoder.ckpt')
else:
    sess.run(init)

for i in range(500):
    trainInfo, _ = sess.run([merge, train], feed_dict=FeedDict(True))
    trainWriter.add_summary(trainInfo, i)

saver.save(sess, './AutoEncoder/AutoEncoder.ckpt')
trainWriter.close()
sess.close()

