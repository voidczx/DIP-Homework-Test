# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/9/2910:28
# File：  New09.py
# Engine：PyCharm


#import tensorflow as tf
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

tf.compat.v1.disable_eager_execution()

batchSize = 128
stepSize = 28
numSize = 28
hideSize = 150
outSize = 10

trainCount = 0
testCount = 0

def FeedDict(trainFlag):
    global trainCount
    global testCount
    if trainFlag:
        x, y = mnist.train.next_batch(batchSize)
        # x = x[trainCount*batchSize:(trainCount+1)*batchSize, :]
        # y = y[trainCount*batchSize:(trainCount+1)*batchSize, :]
        # trainCount = trainCount + 1
        x = x.reshape([batchSize, stepSize, numSize])
    else:
        x, y = mnist.test.next_batch(batchSize)
        # x = x[testCount*batchSize:(testCount+1)*batchSize, :]
        # y = y[testCount*batchSize:(testCount+1)*batchSize, :]
        # testCount = testCount + 1
        x = x.reshape([batchSize, stepSize, numSize])
    return {xs: x, ys: y}


def GetWeight(shape):
    with tf.name_scope('Weight'):
        weight = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(weight, name='weight')


def GetBias(shape):
    with tf.name_scope('Bias'):
        bias = tf.constant(0.1, shape=shape)
        return tf.Variable(bias, name='bias')


def AddLayer(input, weight, bias, nLayer=None):
    with tf.name_scope('Layer'+str(nLayer)):
        layer = tf.nn.relu(tf.matmul(input, weight) + bias, name='layer'+str(nLayer))
        return layer


with tf.name_scope('Input'):
    xs = tf.placeholder(tf.float32, [None, stepSize, numSize], name='inputX')
    ys = tf.placeholder(tf.float32, [None, outSize], name='inputY')

with tf.name_scope('HideLayer01'):
    input01 = tf.reshape(xs, [-1, numSize], name='dim2Input01')
    weight01 = GetWeight([numSize, hideSize])
    bias01 = GetBias([hideSize])
    layer01 = AddLayer(input01, weight01, bias01, nLayer=1)

with tf.name_scope('RnnLayer'):
    input02 = tf.reshape(layer01, [-1, stepSize, hideSize], name='dim3Input02')
    rnnFrame = tf.nn.rnn_cell.BasicLSTMCell(hideSize, forget_bias=1.0, state_is_tuple=True)
    initialState = rnnFrame.zero_state(batch_size=batchSize, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(rnnFrame, input02, initial_state=initialState, time_major=False)

with tf.name_scope('HideLayer02'):
    prediction = AddLayer(states[1], GetWeight([hideSize, outSize]), GetBias([outSize]), 2)

with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=prediction))
    tf.summary.scalar('loss', loss)

with tf.name_scope('Train'):
    train = tf.train.AdamOptimizer(2e-4).minimize(loss)

with tf.name_scope('Accuracy'):
    correct = tf.equal(tf.argmax(ys, axis=1), tf.argmax(prediction, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))
    tf.summary.scalar('accuracy', accuracy)

init = tf.initialize_all_variables()
sess = tf.Session()
merge = tf.summary.merge_all()
trainWriter = tf.summary.FileWriter('./logs/train', sess.graph)
testWriter = tf.summary.FileWriter('./logs/test')

sess.run(init)
for i in range(0, 1200):
    trainInfo, _ = sess.run([merge, train], feed_dict=FeedDict(True))
    trainWriter.add_summary(trainInfo, i)
    if i % 100 == 0:
        testInfo, accuracyInfo = sess.run([merge, accuracy], feed_dict=FeedDict(False))
        testWriter.add_summary(testInfo, i)

trainWriter.close()
testWriter.close()
sess.close()
