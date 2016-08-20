# coding:utf-8

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', './', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

# 初始化生成随机的权重(变量)，避免神经元输出恒为0
def weight_variable(shape):
    # 以正态分布生成随机值
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 初始化生成随机的偏置项(常量)，避免神经元输出恒为0
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 卷积采用1步长，0边距，保证输入输出大小相同
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 池化采用2×2模板
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1], padding='SAME')

# 28*28=784
x = tf.placeholder(tf.float32, [None, 784])
# 输出类别共10个：0-9
y_ = tf.placeholder("float", [None,10])

# 第一层卷积权重，视野是5*5，输入通道1个，输出通道32个
W_conv1 = weight_variable([5, 5, 1, 32])
# 第一层卷积偏置项有32个
b_conv1 = bias_variable([32])

# 把x变成4d向量，第二维和第三维是图像尺寸，第四维是颜色通道数1
x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积权重，视野是5*5，输入通道32个，输出通道64个
W_conv2 = weight_variable([5, 5, 32, 64])
# 第二层卷积偏置项有64个
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 第二层池化后尺寸编程7*7，第三层是全连接，输入是64个通道，输出是1024个神经元
W_fc1 = weight_variable([7 * 7 * 64, 1024])
# 第三层全连接偏置项有1024个
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 按float做dropout，以减少过拟合
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 最后的softmax层生成10种分类
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
# Adam优化器来做梯度最速下降
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print "step %d, training accuracy %g"%(i, train_accuracy)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print "test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
