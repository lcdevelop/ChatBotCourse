# coding: utf-8
# 自动编解码器实现自动问答

import sys
import jieba
import struct
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


class MyLSTM(object):
    def __init__(self):
        self.max_abs_weight = 32  # 最大权重绝对值，用来对词向量做正规化
        self.max_seq_len = 8  # 最大句子长度(词)
        self.word_vec_dim = 0  # 词向量维度，读vectors.bin二进制时动态确定
        self.epoch = 10000
        self.word_vector_dict = {}  # 词向量词典，加载vectors.bin读入
        self.vectors_bin_file = './vectors.bin'  # 词向量二进制
        self.model_dir = './model/model'  # 模型文件路径
        self.n_hidden = 1000  # lstm隐藏状态单元数目
        self.learning_rate = 0.01  # 学习率

    def load_word_vectors(self):
        """加载词向量二进制到内存"""
        float_size = 4  # 一个浮点数4字节
        max_w = 50  # 最大单词字数
        input_file = open(self.vectors_bin_file, "rb")
        # 获取词表数目及向量维度
        words_and_size = input_file.readline()
        words_and_size = words_and_size.strip()
        words = long(words_and_size.split(' ')[0])
        self.word_vec_dim = long(words_and_size.split(' ')[1])
        print("词表总词数：%d" % words)
        print("词向量维度：%d" % self.word_vec_dim)

        for b in range(0, words):
            a = 0
            word = ''
            # 读取一个词
            while True:
                c = input_file.read(1)
                word = word + c
                if False == c or c == ' ':
                    break
                if a < max_w and c != '\n':
                    a = a + 1
            word = word.strip()
            vector = []

            for index in range(0, self.word_vec_dim):
                m = input_file.read(float_size)
                (weight,) = struct.unpack('f', m)
                f_weight = float(weight)
                vector.append(f_weight)

            # 将词及其对应的向量存到dict中
            try:
                self.word_vector_dict[word.decode('utf-8')] = vector[0:self.word_vec_dim]
            except:
                # 异常的词舍弃掉
                # print('bad word:' + word)
                pass

        input_file.close()
        print "finish"

    def next_batch(self):
        """获取训练样本"""
        XY = []  # lstm的训练输入
        Y = []  # lstm的训练输出
        EOS = [np.ones(self.word_vec_dim)]
        sample_file_object = open('./samples/1', 'r')
        lines = sample_file_object.readlines()
        for line in lines:
            line = line.strip()
            split = line.split('|')
            if len(split) == 2:
                question = split[0]
                answer = split[1]
                print('question:[%s] answer:[%s]' % (question, answer))

                question_seq = [np.zeros(self.word_vec_dim)] * self.max_seq_len
                answer_seq = [np.zeros(self.word_vec_dim)] * self.max_seq_len
                segments = jieba.cut(question)
                for index, word in enumerate(segments):
                    if word in self.word_vector_dict:
                        vec = np.array(self.word_vector_dict[word]) / self.max_abs_weight
                        # 防止词过多越界
                        if self.max_seq_len - index - 1 < 0:
                            break
                        # 问题不足max_seq_len在前面补零，存储时倒序存储
                        question_seq[self.max_seq_len - index - 1] = vec

                segments = jieba.cut(answer)
                for index, word in enumerate(segments):
                    if word in self.word_vector_dict:
                        vec = np.array(self.word_vector_dict[word]) / self.max_abs_weight
                        # 防止词过多越界
                        if index >= self.max_seq_len:
                            break
                        answer_seq[index] = vec

                xy = question_seq + EOS + answer_seq[0:-1]
                y = answer_seq
                XY.append(xy)
                Y.append(y)

        sample_file_object.close()

        return XY, Y

    def model(self, x, y, weights, biases, predict=False):
        encoder_inputs = tf.slice(x, [0, 0, 0], [1, self.max_seq_len, self.word_vec_dim])
        encoder_inputs = tf.unstack(encoder_inputs, self.max_seq_len, 1)

        if predict:
            decoder_inputs = tf.slice(x, [0, self.max_seq_len, 0], [1, 1, self.word_vec_dim])
            decoder_inputs = tf.unstack(decoder_inputs, 1, 1)
        else:
            decoder_inputs = tf.slice(x, [0, self.max_seq_len, 0], [1, self.max_seq_len, self.word_vec_dim])
            decoder_inputs = tf.unstack(decoder_inputs, self.max_seq_len, 1)

        target_outputs = tf.slice(y, [0, 0, 0], [1, self.max_seq_len, self.word_vec_dim])
        target_outputs = tf.unstack(target_outputs, self.max_seq_len, 1)

        encoder = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
        decoder = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)

        encoder_outputs, states = rnn.static_rnn(encoder, encoder_inputs, dtype=tf.float32, scope='encoder')
        if predict:
            decoder_output, states = rnn.static_rnn(decoder, decoder_inputs, initial_state=states, dtype=tf.float32, scope='decoder')
        else:
            decoder_outputs, states = rnn.static_rnn(decoder, decoder_inputs, initial_state=states, dtype=tf.float32, scope='decoder')

        optimizer = None
        cost = None

        if predict:
            decoder_outputs = []
            decoder_outputs.append(decoder_output)

            for i in range(self.max_seq_len - 1):
                decoder_output = tf.unstack(decoder_output, axis=1)[0]
                decoder_output = tf.matmul(decoder_output, weights['out']) + tf.slice(biases['out'], [i, 0],
                                                                                      [1, self.word_vec_dim])
                decoder_output, states = rnn.static_rnn(decoder, [decoder_output], initial_state=states,
                                                        dtype=tf.float32,
                                                        scope='decoder')
                decoder_outputs.append(decoder_output)
            decoder_outputs = tf.unstack(decoder_outputs, axis=1)[0]
            decoder_outputs = tf.unstack(decoder_outputs, axis=1)[0]
            decoder_outputs = tf.matmul(decoder_outputs, weights['out']) + biases['out']
        else:
            decoder_outputs = tf.unstack(decoder_outputs, axis=1)[0]
            decoder_outputs = tf.matmul(decoder_outputs, weights['out']) + biases['out']
            target_outputs = tf.unstack(target_outputs, axis=1)[0]

            cost = tf.losses.mean_squared_error(decoder_outputs, target_outputs)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        return optimizer, cost, decoder_outputs, target_outputs, encoder_inputs, decoder_inputs

    def train(self):
        x = tf.placeholder("float", [None, self.max_seq_len * 2, self.word_vec_dim])
        y = tf.placeholder("float", [None, self.max_seq_len, self.word_vec_dim])

        weights = {
            'out': tf.Variable(tf.random_normal([self.n_hidden, self.word_vec_dim]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([self.max_seq_len, self.word_vec_dim]))
        }

        optimizer, cost, decoder_outputs, target_outputs, encoder_inputs, decoder_inputs = self.model(x, y, weights, biases)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        XY, Y = self.next_batch()
        n_steps = len(XY)

        for i in range(self.epoch):
            for step in range(n_steps):
                train_XY = XY[step:]
                train_Y = Y[step:]
                sess.run(optimizer, feed_dict={x: train_XY, y: train_Y})
                loss = sess.run(cost, feed_dict={x: train_XY, y: train_Y})
                if i % 100 == 0 and step == 0:
                    print 'i=%d, loss=%f' % (i, loss)

        saver = tf.train.Saver()
        saver.save(sess, self.model_dir)

    def predict(self):
        x = tf.placeholder("float", [None, self.max_seq_len * 2, self.word_vec_dim])
        y = tf.placeholder("float", [None, self.max_seq_len, self.word_vec_dim])

        weights = {
            'out': tf.Variable(tf.random_normal([self.n_hidden, self.word_vec_dim]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([self.max_seq_len, self.word_vec_dim]))
        }

        optimizer, cost, decoder_outputs, target_outputs, encoder_inputs, decoder_inputs = self.model(x, y, weights, biases, predict=True)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, self.model_dir)

        question = '你是谁'
        XY = []  # lstm的训练输入
        Y = []
        EOS = [np.ones(self.word_vec_dim)]
        question_seq = [np.zeros(self.word_vec_dim)] * self.max_seq_len
        segments = jieba.cut(question)
        for index, word in enumerate(segments):
            if word in self.word_vector_dict:
                vec = np.array(self.word_vector_dict[word]) / self.max_abs_weight
                # 防止词过多越界
                if self.max_seq_len - index - 1 < 0:
                    break
                question_seq[self.max_seq_len - index - 1] = vec

        xy = question_seq + EOS + [np.zeros(self.word_vec_dim)] * (self.max_seq_len-1)
        XY.append(xy)
        Y.append([np.zeros(self.word_vec_dim)] * self.max_seq_len)
        print sess.run(decoder_outputs, feed_dict={x: XY, y: Y})


def main(op):
    lstm = MyLSTM()
    lstm.load_word_vectors()
    if op == 'train':
        lstm.train()
    elif op == 'predict':
        lstm.predict()
    else:
        print 'Usage:'

if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print 'Usage:'
