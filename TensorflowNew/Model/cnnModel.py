# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/10/1116:34
# File：  cnnModel.py
# Engine：PyCharm

# import tensorflow as tf
import tensorflow.compat.v1 as tf
import Config
import os
from pathlib import Path
import cv2 as cv
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"  # 选择哪一块gpu,如果是-1，就是调用cpu
config = tf.ConfigProto()  # 对session进行参数配置
config.allow_soft_placement = True  # 如果你指定的设备不存在，允许TF自动分配设备
config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 分配百分之七十的显存给程序使用，避免内存溢出，可以自己调整
config.gpu_options.allow_growth = True  # 按需分配显存，这个比较重要
sess = tf.Session(config=config)

tf.compat.v1.disable_eager_execution()

configInfo = {}
dataDict = {}
configInfo = Config.GetConfig(configPath='./Config.ini')

batchSize = configInfo['batchsize']
widthSize = configInfo['widthsize']
heightSize = configInfo['heightsize']
channelSize = configInfo['channelsize']
learnRate = configInfo['learnrate']
totalSize = configInfo['totalsize']
cnnFirstSize = configInfo['cnnfirstsize']
cnnSecondSize = configInfo['cnnsecondsize']
normalFirstSize = configInfo['normalfirstsize']
normalSecondSize = configInfo['normalsecondsize']
outputSize = configInfo['outputsize']
dropRate = configInfo['droprate']
sideSize = configInfo['sidesize']
trainLogPath = configInfo['trainlogpath']
testLogPath = configInfo['testlogpath']
saverPath = configInfo['saverpath']
saverDistPath = configInfo['saverdistpath']
graphMetaPath = configInfo['graphmetapath']


def UnpickleData(dataPath):
    import pickle
    with open(dataPath, 'rb') as fo:
        dataDict.update(pickle.load(fo, encoding='bytes'))


def GetMeta(dataPath):
    import pickle
    with open(dataPath, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

UnpickleData(configInfo['databatchpath01'])
dataX = dataDict[b'data']
dataY = dataDict[b'labels']
UnpickleData(configInfo['testbatchpath'])
testX = dataDict[b'data']
testY = dataDict[b'labels']

def FeedDict(trainFlag):
    x = list()
    y = list()
    if trainFlag:
        rand = np.random.randint(0, 10000, batchSize)
        for index in range(rand.size):
            xWait = dataX[index]
            xWait = np.float32(xWait)
            cv.normalize(xWait, xWait, 0, 1, cv.NORM_MINMAX)
            # print(xWait)
            xWait = np.reshape(np.reshape(xWait, [3, 1024]).T, [32, 32, 3])
            x.append(xWait)
            hotY = np.zeros([10])
            hotY[int(dataY[index])] = 1
            y.append(hotY)
        # print(x[3:10])
        # x = tf.reshape(x, [-1, widthSize*heightSize*channelSize])
        drop = dropRate
    else:
        x = list()
        for singleX in testX:
            xWait = singleX
            xWait = np.float32(xWait)
            cv.normalize(xWait, xWait, 0, 1, cv.NORM_MINMAX)
            xWait = np.reshape(np.reshape(xWait, [3, 1024]).T, [32, 32, 3])
            x.append(xWait)
        y = list()
        for singleY in testY:
            hotY = np.zeros([10])
            hotY[int(singleY)] = 1
            y.append(hotY)
        # print(x[5:20])
        drop = 1.0
    return {xs: x, ys: y, dropProb: drop}


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


def GetNormalLayer(input, weight, bias, activation=None, nLayer=None):
    with tf.name_scope('Layer'+str(nLayer)):
        layer = tf.matmul(input, weight) + bias
        if activation is None:
            result = layer
        else:
            result = activation(layer, name='layer')
        return result


def CnnLayer(input, weight, nLayer=None):
    with tf.name_scope('CnnLayer'+str(nLayer)):
        layer = tf.nn.conv2d(input, weight, strides=[1, 1, 1, 1], padding='SAME', name='cnnLayer')
        return layer


def PoolingLayer(input):
    with tf.name_scope('Pooling'):
        pool = tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='poolingLayer')
        return pool


with tf.name_scope('Input'):
    xs = tf.placeholder(tf.float32, [None, widthSize, heightSize, channelSize])
    ys = tf.placeholder(tf.float32, [None, outputSize])
    dropProb = tf.placeholder(tf.float32)
    tf.summary.scalar('dropProb', dropProb)

with tf.name_scope('cnnLayer01'):
    input01 = tf.reshape(xs, [-1, widthSize, heightSize, channelSize], name='dim3Input01')
    weight01 = GetWeight([sideSize, sideSize, channelSize, cnnFirstSize])
    bias01 = GetBias([cnnFirstSize])
    cnnLayer01 = tf.nn.relu(CnnLayer(input01, weight01, nLayer=1) + bias01, name='cnnLayer01')     #(32, 32, 32)
    pooling01 = PoolingLayer(cnnLayer01)    #(16, 16, 32)

with tf.name_scope('cnnLayer02'):
    weight02 = GetWeight([sideSize, sideSize, cnnFirstSize, cnnSecondSize])
    bias02 = GetBias([cnnSecondSize])
    cnnLayer02 = tf.nn.relu(CnnLayer(pooling01, weight02, nLayer=2) + bias02, name='cnnLayer02')  #(16, 16, 64)
    pooling02 = PoolingLayer(cnnLayer02)    #(8, 8, 64)

with tf.name_scope('NormalLayer01'):
    input02 = tf.reshape(pooling02, [-1, int(widthSize/2/2)*int(heightSize/2/2)*cnnSecondSize], name='dim1Input02')
    weight03 = GetWeight([int(widthSize/2/2)*int(heightSize/2/2)*cnnSecondSize, normalFirstSize])
    bias03 = GetBias([normalFirstSize])
    normalLayer01 = GetNormalLayer(input02, weight03, bias03, activation=tf.nn.relu, nLayer=3)
    dropLayer = tf.nn.dropout(normalLayer01, keep_prob=dropProb)

with tf.name_scope('NormalLayer02'):
    weight04 = GetWeight([normalFirstSize, normalSecondSize])
    bias04 = GetBias([normalSecondSize])
    normalLayer02 = GetNormalLayer(dropLayer, weight04, bias04, activation=tf.nn.relu, nLayer=4)
    dropLayer02 = tf.nn.dropout(normalLayer02, keep_prob=dropProb)

with tf.name_scope('SoftMax'):
    weight05 = GetWeight([normalSecondSize, outputSize])
    bias05 = GetBias([outputSize])
    prediction = GetNormalLayer(dropLayer02, weight05, bias05, activation=tf.nn.softmax, nLayer=5)

with tf.name_scope('Loss'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(tf.clip_by_value(prediction, 1e-15, 1.0)), reduction_indices=[1]))
    # tf.add_to_collection('losses', cross_entropy)  # 将交叉熵加入损失函数集合losses
    # loss = tf.add_n(tf.get_collection('losses'))    # 将losses全部结果相加
    tf.summary.scalar('loss', cross_entropy)

with tf.name_scope('Train'):
    # train = tf.train.AdamOptimizer(learnRate).minimize(cross_entropy)
    train = tf.train.MomentumOptimizer(learnRate, momentum=0.9).minimize(cross_entropy)

with tf.name_scope('Accuracy'):
    correct = tf.equal(tf.argmax(ys, 1), tf.argmax(prediction, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

saver = tf.train.Saver()
init = tf.initialize_all_variables()
merge = tf.summary.merge_all()
trainWriter = tf.summary.FileWriter(trainLogPath, sess.graph)
testWriter = tf.summary.FileWriter(testLogPath)

if Path(saverDistPath).exists():
    saver.restore(sess, saverPath)
else:
    sess.run(init)

# sess.run(init)

# batch01

for i in range(500):
    _, trainInfo, predictionInfo = sess.run([train, merge, prediction], feed_dict=FeedDict(True))
    # print(predictionInfo)
    trainWriter.add_summary(trainInfo, i)
    if i % 50 == 0:
        testInfo, accuracyInfo = sess.run([merge, accuracy], feed_dict=FeedDict(False))
        print(accuracyInfo)
        testWriter.add_summary(testInfo, i)

saver.save(sess, saverPath)

trainWriter.close()
testWriter.close()
sess.close()

# print(GetMeta(configInfo['metapath']))
# print(configInfo)
# UnpickleData(configInfo['databatchpath01'])
# UnpickleData(configInfo['databatchpath02'])
# UnpickleData(configInfo['databatchpath03'])
# print(dataDict.keys())
# print(dataDict[b'batch_label'])
# print(dataDict[b'labels'])
# print(dataDict[b'data'].shape)