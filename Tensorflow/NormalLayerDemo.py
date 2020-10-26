# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/9/2414:13
# File：  NormalLayerDemo.py
# Engine：PyCharm


#import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow.examples.tutorials.mnist import input_data


tf.compat.v1.disable_eager_execution()


def AddLayer(input, inSize, outSize, nLayer, activitionFunction):
    layerName = 'layer%s' % nLayer
    with tf.name_scope(layerName):
        with tf.name_scope('Weight'):
            Weight = tf.Variable(tf.random_normal([inSize, outSize], name='weight'))
        with tf.name_scope('Biasis'):
            biasis = tf.Variable(tf.zeros([1, outSize]) + 0.1, name='biasis')
        with tf.name_scope('LinearCompute'):
            compute = tf.matmul(input, Weight) + biasis
            tf.summary.histogram('linear compute', compute)
        if activitionFunction == None:
            output = compute
        else:
            output = activitionFunction(compute, name='activition')
            tf.summary.histogram('activition function', output)
    return output


def FeedBack(trainFlag):
    if trainFlag:
        x, y = mnist.train.next_batch(100)
        z = 0.9
    else:
        x, y = mnist.test.images, mnist.test.labels
        z = 1.0
    return {xs: x, ys: y, drop: z}


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

with tf.name_scope('Input'):
    xs = tf.placeholder(tf.float32, [None, 784], name='inputX')
    ys = tf.placeholder(tf.float32, [None, 10], name='inputY')

with tf.name_scope('Image'):
    #tensorflow.reshape中shape参数的第一个-1是为了使重组后矩阵大小不变
    image = tf.reshape(xs, [-1, 28, 28, 1])
    tf.summary.image('input image', image, max_outputs=10)

with tf.name_scope('LayerOne'):
    layerOne = AddLayer(xs, 784, 500, 1, activitionFunction=tf.nn.relu)

with tf.name_scope('DropLayer'):
    drop = tf.placeholder(tf.float32)
    tf.summary.scalar('drop', drop)
    dropped = tf.nn.dropout(layerOne, drop)

with tf.name_scope('OutputLayer'):
    predictionY = AddLayer(dropped, 500, 10, 2, tf.identity)

with tf.name_scope('Loss'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=predictionY)
    with tf.name_scope('TotalMean'):
        loss = tf.reduce_mean(diff)
tf.summary.scalar('total mean loss', loss)

with tf.name_scope('Train'):
    train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

with tf.name_scope('Accuracy'):
    with tf.name_scope('Correct'):
        correct = tf.equal(tf.argmax(ys, 1), tf.argmax(predictionY, 1))
    with tf.name_scope('MeanAccuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
merge = tf.summary.merge_all()
trainWriter = tf.summary.FileWriter('./logs/train', sess.graph)
testWriter = tf.summary.FileWriter('./logs/test')

for i in range(1000):
    trainInfo, _ = sess.run([merge, train], feed_dict=FeedBack(True))
    trainWriter.add_summary(trainInfo, i)
    if i % 10 == 0:
        testInfo, accuracyInfo = sess.run([merge, accuracy], feed_dict=FeedBack(False))
        testWriter.add_summary(testInfo, i)

testWriter.close()
trainWriter.close()
sess.close()




