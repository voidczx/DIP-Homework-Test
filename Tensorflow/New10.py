# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/10/313:03
# File：  New10.py
# Engine：PyCharm

#import tensorflow as tf
import numpy as np
import tensorflow.compat.v1 as tf
from matplotlib import pyplot as plt
import tensorflow_addons as tfa


tf.compat.v1.disable_eager_execution()

batchSize = 50
stepSize = 20
batchStart = 0
hiddenSize = 10
inputSize = 1
outputSize = 1


def GetBatch():
    global batchStart
    global theX, res, seq
    theX = np.arange(batchStart, batchStart+batchSize*stepSize).reshape([batchSize, stepSize])
    seq = np.sin(theX)
    res = np.cos(theX)
    batchStart = batchStart + stepSize
    return seq[:, :, np.newaxis], res[:, :, np.newaxis], theX


def FeedDict(i):
    global state
    seq, res, x = GetBatch()
    if i == 0:
        return {xs: seq, ys: res}
    else:
        return {xs: seq, ys: res, theState: state}


def GetWeight(shape):
    with tf.name_scope('Weight'):
        weight = tf.truncated_normal(shape, stddev=0.1)
    tf.summary.histogram('weight', weight)
    return tf.Variable(weight, name='weight')


def GetBias(shape):
    with tf.name_scope('Bias'):
        bias = tf.constant(0.1, shape=shape)
    tf.summary.histogram('bias', bias)
    return tf.Variable(bias, name='bias')


def GetLayer(input, weight, bias, activation=None, nLayer=None):
    with tf.name_scope('Layer'+str(nLayer)):
        layer = tf.matmul(input, weight) + bias
        if activation == None:
            result = layer
        else:
            result = activation(layer, name='layer')
    return result


def CalcMistake(labels, logits):
    return tf.abs(tf.subtract(labels, logits))

with tf.name_scope('Input'):
    xs = tf.placeholder(tf.float32, [None, stepSize, inputSize], name='inputX')
    ys = tf.placeholder(tf.float32, [None, stepSize, inputSize], name='inputY')

with tf.name_scope('Layer01'):
    input01 = tf.reshape(xs, [-1, inputSize], name='dim2Input01')
    weight01 = GetWeight([inputSize, hiddenSize])
    bias01 = GetBias([hiddenSize])
    layer01 = GetLayer(input01, weight01, bias01, nLayer=1)

with tf.name_scope('RnnLayer'):
    input02 = tf.reshape(layer01, [-1, stepSize, hiddenSize], name='dim3Input02')
    rnnFrame = tf.nn.rnn_cell.BasicLSTMCell(hiddenSize, forget_bias=1.0, state_is_tuple=True)
    theState = rnnFrame.zero_state(batch_size=batchSize, dtype=tf.float32)
    outputs, finalState = tf.nn.dynamic_rnn(rnnFrame, input02, initial_state=theState, time_major=False)

with tf.name_scope('Layer02'):
    input03 = tf.reshape(outputs, [-1, hiddenSize], name='dim2Input03')
    weight02 = GetWeight([hiddenSize, outputSize])
    bias02 = GetBias([outputSize])
    prediction = GetLayer(input03, weight02, bias02, nLayer=2)

# with tf.name_scope('Loss'):
# #     losses = tfa.seq2seq.sequence_loss(
# #         tf.reshape(prediction, [batchSize, stepSize, outputSize]),
# #         tf.cast(tf.reshape(ys, [batchSize, stepSize]), tf.int64),
# #         tf.ones([batchSize, stepSize], dtype=tf.float32),
# #         average_across_timesteps=True,
# #         average_across_batch=True,
# #         name='losses'
# #     )
    losses = CalcMistake(tf.reshape(prediction, [-1]), tf.reshape(ys, [-1]))
    with tf.name_scope('AverageLoss'):
        loss = tf.divide(tf.reduce_sum(losses), batchSize, name='averageLoss')
tf.summary.scalar('loss', loss)

with tf.name_scope('Train'):
    train = tf.train.AdamOptimizer(0.006).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
merge = tf.summary.merge_all()
Writer = tf.summary.FileWriter('./logs/New10', sess.graph)

plt.ion()
plt.show()

for i in range(200):
    _, trainInfo, state, pred = sess.run([train, merge, finalState, prediction], feed_dict=FeedDict(i))
    Writer.add_summary(trainInfo, i)
    plt.plot(theX[0, :], res[0].flatten(), 'r', theX[0, :], pred.flatten()[:stepSize], 'b--')
    plt.ylim((-1.2, 1.2))
    plt.draw()
    plt.pause(0.1)  # 每 0.3 s 刷新一次
Writer.close()
sess.close()
