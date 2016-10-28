# coding:utf-8

from __future__ import print_function
import numpy as np
import tensorflow as tf
import sys

vocab_size=256
learning_rate=0.1
# 暂时只试验一个bucket
buckets=[(10, 10)]
bucket_id=0
# 填充0
PAD=[0]
sample_size=20
# LSTM中的记忆单元数目
num_units=100
# 多少层的lstm
num_layers=2

# sample_size个样本，每个样本有一个question、answer、weights，question、answer分别是10维的向量
# 这sample_size个样本有时间序上的依赖关系
question_sample_list = [map(ord, "hello?") + PAD * 4] * sample_size
answer_sample_list = [map(ord, "world!") + PAD * 4] * sample_size
init_weights_list = [[1.0]*7 + [0.0]*3] *sample_size # mask padding. todo: redundant --

with tf.Session() as session:

    # 初始化神经网络单元
    cell = single_cell = tf.nn.rnn_cell.LSTMCell(num_units)
    if num_layers > 1:
        cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

    # 定义函数
    def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
        return tf.nn.seq2seq.embedding_rnn_seq2seq(
             encoder_inputs, decoder_inputs, cell,
             num_encoder_symbols=vocab_size,
             num_decoder_symbols=vocab_size,
             embedding_size=num_units,
             feed_previous=do_decode)

    # 初始化训练用的变量，如果是多个层，权重共享
    encoder_inputs = []
    decoder_inputs = []
    weights = []
    for i in xrange(sample_size):
        encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
    for i in xrange(sample_size):
        decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
        weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))
    targets = [decoder_inputs[i] for i in xrange(len(decoder_inputs))]

    # 创建模型及损失计算方法
    buckets_outputs, losses = tf.nn.seq2seq.model_with_buckets(
         encoder_inputs, decoder_inputs, targets,
         weights, buckets,
         lambda x, y: seq2seq_f(x, y, False))


    # 梯度更新算法
    updates=[]
    for b in xrange(len(buckets)):
        updates.append(tf.train.AdamOptimizer(learning_rate).minimize(losses[b]))

    # 用于保存模型
    saver = tf.train.Saver(tf.all_variables())

    # 初始化
    session.run(tf.initialize_all_variables())

    while True:
        encoder_size = len(encoder_inputs)
        decoder_size = len(decoder_inputs)

        # 初始化feed_dict数据
        feed_dict = {}
        for i in xrange(encoder_size):
            feed_dict[encoder_inputs[i].name] = question_sample_list[i]
        for i in xrange(decoder_size):
            feed_dict[decoder_inputs[i].name] = answer_sample_list[i]
            feed_dict[weights[i].name] = init_weights_list[i]

        # 初始化fetches模型相关信息,fetches就是想拿什么就拿什么，比如updates就是拿更新值，losses就是拿损失值，buckets_outputs就是拿输出值
        fetches = [updates[bucket_id], losses[bucket_id]]
        fetches.append(buckets_outputs[bucket_id][0])
        # 这一句是为了拿输出，训练过程可以不要
        for i in xrange(len(buckets_outputs[bucket_id])):
            fetches.append(buckets_outputs[bucket_id][i])

        # 参数传递进去的是数据和计算逻辑，具体执行时可以传到各种介质中执行
        fetches_outputs = session.run(fetches, feed_dict)
        perplexity = fetches_outputs[1]
        outputs = fetches_outputs[2:]
        print ("perplexity =", perplexity)
        words = np.argmax(outputs, axis=2)
        word = "".join(map(chr, words[0])).replace('\x00', '').replace('\n', '')
        print("output: %s" % word)
