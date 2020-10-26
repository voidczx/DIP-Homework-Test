# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/9/2617:02
# File：  CnnLayerDemo.py
# Engine：PyCharm


#import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow.examples.tutorials.mnist import input_data
from pathlib import Path

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"  # 选择哪一块gpu,如果是-1，就是调用cpu
config = tf.ConfigProto()  # 对session进行参数配置
config.allow_soft_placement = True  # 如果你指定的设备不存在，允许TF自动分配设备
config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 分配百分之七十的显存给程序使用，避免内存溢出，可以自己调整
config.gpu_options.allow_growth = True  # 按需分配显存，这个比较重要
sess = tf.Session(config=config)

# sess = tf.Session()

tf.compat.v1.disable_eager_execution()
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def FeedBack(trainFlag):
    if trainFlag:
        x, y = mnist.train.next_batch(100)
        z = 0.9
    else:
        x, y = mnist.test.images, mnist.test.labels
        z = 1.0
    return {xs: x, ys: y, dropProb: z}


def GetWeight(shape):
    with tf.name_scope('Weight'):
        weight = tf.truncated_normal(shape, stddev=0.1)
        tf.summary.histogram('weightHist', weight)
        return tf.Variable(weight, name='weight')


def GetBiasis(shape):
    with tf.name_scope('Biasis'):
        biasis = tf.constant(0.1, shape=shape)
        tf.summary.histogram('biasisHist', biasis)
        return tf.Variable(biasis, name='biasis')


def CNNLayer(input, weight):
    with tf.name_scope('Layer'):
        # strides=[1,?,?,1]
        return tf.nn.conv2d(input, weight, strides=[1, 1, 1, 1], padding='SAME', name='cnnLayer')


def GetPooling(input):
    with tf.name_scope('Pooling'):
        return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='poolingLayer')


with tf.name_scope('Input'):
    xs = tf.placeholder(tf.float32, [None, 784], name='inputX')
    ys = tf.placeholder(tf.float32, [None, 10], name='inputY')
    dropProb = tf.placeholder(tf.float32)
    tf.summary.scalar('dropProb', dropProb)
    image3d = tf.reshape(xs, [-1, 28, 28, 1])
    tf.summary.image('image', image3d, max_outputs=10)
# 5, 5 is patch's size; 1 is input height; 32 is output height
with tf.name_scope('CnnLayer01'):
    CnnWeight01 = GetWeight([5, 5, 1, 32])
    CnnBiasis01 = GetBiasis([32])
    CnnLayer01 = tf.nn.relu(CNNLayer(image3d, CnnWeight01) + CnnBiasis01)  # output size is (28, 28, 32)
    CnnPooling01 = GetPooling(CnnLayer01)  # output size is (14, 14, 32)

with tf.name_scope('CnnLayer02'):
    CnnWeight02 = GetWeight([5, 5, 32, 64])
    CnnBiasis02 = GetBiasis([64])
    CnnLayer02 = tf.nn.relu(CNNLayer(CnnPooling01, CnnWeight02) + CnnBiasis02)  # output size is (14, 14, 64)
    CnnPooling02 = GetPooling(CnnLayer02)   # output size is (7, 7, 64)

CnnPooling02Flat = tf.reshape(CnnPooling02, [-1, 7*7*64], name='flatPoledInput')

with tf.name_scope('NormalLayer01'):
    normalWeight01 = GetWeight([7*7*64, 1024])
    normalBiasis01 = GetBiasis([1024])
    normalLayer01 = tf.nn.relu(tf.matmul(CnnPooling02Flat, normalWeight01) + normalBiasis01)
    droppedLayer01 = tf.nn.dropout(normalLayer01, dropProb)

with tf.name_scope('NormalLayer02'):
    normalWeight02 = GetWeight([1024, 10])
    normalBiasis02 = GetBiasis([10])
    prediction = tf.nn.softmax(tf.matmul(droppedLayer01, normalWeight02) + normalBiasis02, name='prediction')

with tf.name_scope('Loss'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=prediction)
    with tf.name_scope('MeanDiff'):
        loss = tf.reduce_mean(diff)
        tf.summary.scalar('total mean loss', loss)
tf.summary.histogram('loss hist', loss)

with tf.name_scope('Train'):
    train = tf.train.AdamOptimizer(2e-4).minimize(loss)

with tf.name_scope('Accuracy'):
    correct = tf.equal(tf.argmax(ys, 1), tf.argmax(prediction, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
tf.summary.scalar('accuracy', accuracy)

saver = tf.train.Saver()
init = tf.initialize_all_variables()
#sess = tf.Session()
merge = tf.summary.merge_all()
trainWriter = tf.summary.FileWriter('./logs/train', sess.graph)
testWriter = tf.summary.FileWriter('./logs/test')

if Path('./MnistNetWork').exists():
    saver.restore(sess, './MnistNetWork/CnnNetWork.ckpt')
else:
    sess.run(init)

for i in range(1000):
    trainInfo, _ = sess.run([merge, train], feed_dict=FeedBack(True))
    trainWriter.add_summary(trainInfo, i)
    if i % 20 == 0:
        testInfo, accuracyInfo = sess.run([merge, accuracy], feed_dict=FeedBack(False))
        testWriter.add_summary(testInfo, i)

saver.save(sess, './MnistNetWork/CnnNetWork.ckpt')

testWriter.close()
trainWriter.close()
sess.close()

