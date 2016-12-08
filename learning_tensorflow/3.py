# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

# 随机生成1000个点，围绕在y=0.1x+0.3的直线周围
num_points = 4
vectors_set = []
for i in xrange(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])

# 生成一些样本
x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]
print "x_data=", x_data


# 生成1维的W矩阵，取值是[-1,1]之间的随机数
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='W')
# 生成1维的b矩阵，初始值是0
b = tf.Variable(tf.zeros([1]), name='b')
# 经过计算得出预估值y
y = W * x_data + b
print "y=", y

# 以预估值y和实际值y_data之间的均方误差作为损失
loss = tf.reduce_mean(tf.square(y - y_data), name='loss')
# 采用梯度下降法来优化参数
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 训练的过程就是最小化这个误差值
train = optimizer.minimize(loss, name='train')

sess = tf.Session()
# 输出图结构
#print sess.graph_def

init = tf.initialize_all_variables()
sess.run(init)

# 初始化的W和b是多少
print "W =", sess.run(W), "b =", sess.run(b), "loss =", sess.run(loss)
# 执行20次训练
for step in xrange(20):
    sess.run(train)
    # 输出训练好的W和b
    print "W =", sess.run(W), "b =", sess.run(b), "loss =", sess.run(loss)
# 生成summary文件，用于tensorboard使用
writer = tf.train.SummaryWriter("./tmp", sess.graph)
