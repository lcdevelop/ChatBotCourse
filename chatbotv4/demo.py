# coding:utf-8
import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq

# 输入序列长度
input_seq_len = 5
# 输出序列长度
output_seq_len = 5
# 空值填充0
PAD_ID = 0
# 输出序列起始标记
GO_ID = 1
# 结尾标记
EOS_ID = 2
# LSTM神经元size
size = 8
# 最大输入符号数
num_encoder_symbols = 10
# 最大输出符号数
num_decoder_symbols = 16
# 学习率
learning_rate = 0.1


def get_samples():
    """构造样本数据

    :return:
        encoder_inputs: [array([0, 0], dtype=int32), array([0, 0], dtype=int32), array([1, 3], dtype=int32),
                        array([3, 5], dtype=int32), array([5, 7], dtype=int32)]
        decoder_inputs: [array([1, 1], dtype=int32), array([7, 9], dtype=int32), array([ 9, 11], dtype=int32),
                        array([11, 13], dtype=int32), array([0, 0], dtype=int32)]
    """
    train_set = [[[5, 7, 9], [11, 13, 15, EOS_ID]], [[5, 7, 9], [11, 13, 15, EOS_ID]]]
    encoder_input_0 = [PAD_ID] * (input_seq_len - len(train_set[0][0])) + train_set[0][0]
    encoder_input_1 = [PAD_ID] * (input_seq_len - len(train_set[1][0])) + train_set[1][0]
    decoder_input_0 = [GO_ID] + train_set[0][1] + [PAD_ID] * (output_seq_len - len(train_set[0][1]) - 1)
    decoder_input_1 = [GO_ID] + train_set[1][1] + [PAD_ID] * (output_seq_len - len(train_set[1][1]) - 1)

    encoder_inputs = []
    decoder_inputs = []
    target_weights = []
    for length_idx in xrange(input_seq_len):
        encoder_inputs.append(np.array([encoder_input_0[length_idx], encoder_input_1[length_idx]], dtype=np.int32))
    for length_idx in xrange(output_seq_len):
        decoder_inputs.append(np.array([decoder_input_0[length_idx], decoder_input_1[length_idx]], dtype=np.int32))
        target_weights.append(np.array([
            0.0 if length_idx == output_seq_len - 1 or decoder_input_0[length_idx] == PAD_ID else 1.0,
            0.0 if length_idx == output_seq_len - 1 or decoder_input_1[length_idx] == PAD_ID else 1.0,
        ], dtype=np.float32))
    return encoder_inputs, decoder_inputs, target_weights


def get_model(feed_previous=False):
    """构造模型
    """
    encoder_inputs = []
    decoder_inputs = []
    target_weights = []
    for i in xrange(input_seq_len):
        encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
    for i in xrange(output_seq_len + 1):
        decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
    for i in xrange(output_seq_len):
        target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))

    # decoder_inputs左移一个时序作为targets
    targets = [decoder_inputs[i + 1] for i in xrange(output_seq_len)]

    cell = tf.contrib.rnn.BasicLSTMCell(size)

    # 这里输出的状态我们不需要
    outputs, _ = seq2seq.embedding_attention_seq2seq(
                        encoder_inputs,
                        decoder_inputs[:output_seq_len],
                        cell,
                        num_encoder_symbols=num_encoder_symbols,
                        num_decoder_symbols=num_decoder_symbols,
                        embedding_size=size,
                        output_projection=None,
                        feed_previous=feed_previous,
                        dtype=tf.float32)

    # 计算加权交叉熵损失
    loss = seq2seq.sequence_loss(outputs, targets, target_weights)
    # 梯度下降优化器
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    # 优化目标：让loss最小化
    update = opt.apply_gradients(opt.compute_gradients(loss))
    # 模型持久化
    saver = tf.train.Saver(tf.global_variables())
    return encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver


def train():
    """
    训练过程
    """
    with tf.Session() as sess:
        sample_encoder_inputs, sample_decoder_inputs, sample_target_weights = get_samples()
        encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver = get_model()

        input_feed = {}
        for l in xrange(input_seq_len):
            input_feed[encoder_inputs[l].name] = sample_encoder_inputs[l]
        for l in xrange(output_seq_len):
            input_feed[decoder_inputs[l].name] = sample_decoder_inputs[l]
            input_feed[target_weights[l].name] = sample_target_weights[l]
        input_feed[decoder_inputs[output_seq_len].name] = np.zeros([2], dtype=np.int32)

        # 全部变量初始化
        sess.run(tf.global_variables_initializer())

        # 训练200次迭代，每隔10次打印一次loss
        for step in xrange(200):
            [loss_ret, _] = sess.run([loss, update], input_feed)
            if step % 10 == 0:
                print 'step=', step, 'loss=', loss_ret

        # 模型持久化
        saver.save(sess, './model/demo')


def predict():
    """
    预测过程
    """
    with tf.Session() as sess:
        sample_encoder_inputs, sample_decoder_inputs, sample_target_weights = get_samples()
        encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver = get_model(feed_previous=True)
        # 从文件恢复模型
        saver.restore(sess, './model/demo')

        input_feed = {}
        for l in xrange(input_seq_len):
            input_feed[encoder_inputs[l].name] = sample_encoder_inputs[l]
        for l in xrange(output_seq_len):
            input_feed[decoder_inputs[l].name] = sample_decoder_inputs[l]
            input_feed[target_weights[l].name] = sample_target_weights[l]
        input_feed[decoder_inputs[output_seq_len].name] = np.zeros([2], dtype=np.int32)

        # 预测输出
        outputs = sess.run(outputs, input_feed)
        # 一共试验样本有2个，所以分别遍历
        for sample_index in xrange(2):
            # 因为输出数据每一个是num_decoder_symbols维的，因此找到数值最大的那个就是预测的id，就是这里的argmax函数的功能
            outputs_seq = [int(np.argmax(logit[sample_index], axis=0)) for logit in outputs]
            # 如果是结尾符，那么后面的语句就不输出了
            if EOS_ID in outputs_seq:
                outputs_seq = outputs_seq[:outputs_seq.index(EOS_ID)]
            outputs_seq = [str(v) for v in outputs_seq]
            print " ".join(outputs_seq)


if __name__ == "__main__":
    if sys.argv[1] == 'train':
        train()
    else:
        predict()
