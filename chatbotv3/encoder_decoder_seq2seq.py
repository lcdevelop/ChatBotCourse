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
        self.epoch = 1000
        self.word_vector_dict = {}  # 词向量词典，加载vectors.bin读入
        self.one_hot_word_vector_dict = {}  # 根据样本词汇生成的softmax用的词向量
        self.word_id_word_dict = {}
        self.one_hot_word_vectors_dim = 1  # softmax用的词向量维度，从1开始，保留0作为EOS的word_id
        self.eos_word_id = 0
        self.eos_word = 'EOS'
        self.vectors_bin_file = './vectors.bin'  # 词向量二进制
        self.model_dir = './model/model'  # 模型文件路径
        self.n_hidden = 1000  # lstm隐藏状态单元数目
        self.learning_rate = 0.01  # 学习率

    def load_one_hot_word_vectors(self):

        word_id_dict = {}
        sample_file_object = open('./samples/1', 'r')
        lines = sample_file_object.readlines()
        for line in lines:
            line = line.strip()
            split = line.split('|')
            if len(split) == 2:
                answer = split[1]
                segments = jieba.cut(answer)
                for word in segments:
                    if word not in word_id_dict:
                        word_id_dict[word] = self.one_hot_word_vectors_dim
                        self.word_id_word_dict[self.one_hot_word_vectors_dim] = word
                        self.one_hot_word_vectors_dim = self.one_hot_word_vectors_dim + 1

        # 添加一个结尾符
        vector = np.zeros(self.one_hot_word_vectors_dim)
        vector[self.eos_word_id] = 1
        self.one_hot_word_vector_dict[self.eos_word] = vector
        self.word_id_word_dict[self.eos_word_id] = self.eos_word

        for line in lines:
            line = line.strip()
            split = line.split('|')
            if len(split) == 2:
                answer = split[1]
                segments = jieba.cut(answer)
                for word in segments:
                    if word not in self.one_hot_word_vector_dict:
                        word_id = word_id_dict[word]
                        print word, word_id
                        vector = np.zeros(self.one_hot_word_vectors_dim)
                        vector[word_id] = 1
                        self.one_hot_word_vector_dict[word] = vector

        sample_file_object.close()

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

                good_sample = True
                question_seq = [np.zeros(self.word_vec_dim)] * self.max_seq_len
                answer_seq = [np.zeros(self.word_vec_dim)] * self.max_seq_len
                answer_seq_one_hot = [np.zeros(self.one_hot_word_vectors_dim)] * self.max_seq_len
                segments = jieba.cut(question)
                for index, word in enumerate(segments):
                    if word in self.word_vector_dict:
                        vec = np.array(self.word_vector_dict[word]) / self.max_abs_weight
                        # 防止词过多越界
                        if self.max_seq_len - index - 1 < 0:
                            good_sample = False
                            break
                        # 问题不足max_seq_len在前面补零，存储时倒序存储
                        question_seq[self.max_seq_len - index - 1] = vec
                    else:
                        good_sample = False

                segments = jieba.cut(answer)
                last_index = 0
                for index, word in enumerate(segments):
                    if word in self.word_vector_dict:
                        vec = np.array(self.word_vector_dict[word]) / self.max_abs_weight
                        # 防止词过多越界
                        if index >= self.max_seq_len - 1:
                            good_sample = False
                            break
                        answer_seq[index] = vec
                    else:
                        good_sample = False

                    if word in self.one_hot_word_vector_dict:
                        vec = self.one_hot_word_vector_dict[word]
                        answer_seq_one_hot[index] = vec
                    else:
                        good_sample = False
                    last_index = index
                # 句子末尾加上EOS
                answer_seq_one_hot[last_index + 1] = self.one_hot_word_vector_dict[self.eos_word]  # EOS

                if good_sample:
                    xy = question_seq + EOS + answer_seq[0:-1]
                    y = answer_seq_one_hot
                    XY.append(xy)
                    Y.append(y)

        sample_file_object.close()

        return XY, Y

    def model(self, x, y, weights, biases, training=True):
        # 注：以下的6是one_hot_word_vectors_dim
        # 取第一个样本的ABC
        encoder_inputs = tf.slice(x, [0, 0, 0], [1, self.max_seq_len, self.word_vec_dim])  # shape=(1, 8, 128)
        # 展开成2-D Tensor
        encoder_inputs = tf.unstack(encoder_inputs, self.max_seq_len, 1)  # [<tf.Tensor shape=(1, 128)>,...] 内含8个Tensor

        # 取第一个样本的<EOS>WXYZ
        decoder_inputs = tf.slice(x, [0, self.max_seq_len, 0], [1, self.max_seq_len, self.word_vec_dim])  # shape=(1, 8, 128)
        decoder_inputs = decoder_inputs[0]  # shape=(8, 128)
        # 转成解码器的输入输出形状
        decoder_inputs = tf.matmul(decoder_inputs, weights['enc2dec']) + biases['enc2dec']
        # 展开成2-D Tensor
        decoder_inputs = tf.unstack([decoder_inputs], axis=1)  # [<tf.Tensor shape=(1, 6)>,...] 内含8个Tensor

        # 取第一个样本的WXYZ
        target_outputs = tf.slice(y, [0, 0, 0], [1, self.max_seq_len, self.one_hot_word_vectors_dim])  # shape=(1, 8, 6)
        target_outputs = target_outputs[0]  # shape=(8, 6)

        # 构造网络结构：两层结构
        encoder_layer1 = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
        encoder_layer2 = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
        decoder_layer1 = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
        decoder_layer2 = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)

        # 输入是8个shape=(1, 128)的Tensor，输出是8个shape=(1, 1000)的Tensor
        encoder_layer1_outputs, encoder_layer1_states = rnn.static_rnn(encoder_layer1, encoder_inputs, dtype=tf.float32, scope='encoder_layer1')
        # 输入是8个shape=(1, 1000)的Tensor，输出是8个shape=(1, 1000)的Tensor
        encoder_layer2_outputs, encoder_layer2_states = rnn.static_rnn(encoder_layer2, encoder_layer1_outputs, dtype=tf.float32, scope='encoder_layer2')
        # 取解码器输入的<EOS>
        # 输入是1个shape=(1, 6)的Tensor(<EOS>)，输出是1个shape=(1, 1000)的Tensor
        decoder_layer1_outputs, decoder_layer1_states = rnn.static_rnn(decoder_layer1, decoder_inputs[:1], initial_state=encoder_layer1_states, dtype=tf.float32, scope='decoder_layer1')
        # 输入是1个shape=(1, 1000)的Tensor，输出是1个shape=(1, 1000)的Tensor
        decoder_layer2_outputs, decoder_layer2_states = rnn.static_rnn(decoder_layer2, decoder_layer1_outputs, initial_state=encoder_layer2_states, dtype=tf.float32, scope='decoder_layer2')

        decoder_layer2_outputs_combine = []
        decoder_layer2_outputs_combine.append(decoder_layer2_outputs)
        for i in range(self.max_seq_len - 1):
            decoder_layer2_outputs = tf.unstack(decoder_layer2_outputs, axis=1)[0]
            decoder_layer2_outputs = tf.matmul(decoder_layer2_outputs, weights['hid2tar']) + biases['hid2tar'][i]
            # 输入是1个shape=(1, 6)的Tensor，输出是1个shape=(1, 1000)的Tensor
            if training:
                decoder_layer1_outputs, decoder_layer1_states = rnn.static_rnn(decoder_layer1, decoder_inputs[i+1:i+2], initial_state=decoder_layer1_states, dtype=tf.float32, scope='decoder_layer1')
            else:
                decoder_layer1_outputs, decoder_layer1_states = rnn.static_rnn(decoder_layer1, [decoder_layer2_outputs], initial_state=decoder_layer1_states, dtype=tf.float32, scope='decoder_layer1')
            # 输入是1个shape=(1, 1000)的Tensor，输出是1个shape=(1, 1000)的Tensor
            decoder_layer2_outputs, decoder_layer2_states = rnn.static_rnn(decoder_layer2, decoder_layer1_outputs, initial_state=decoder_layer2_states, dtype=tf.float32, scope='decoder_layer2')
            decoder_layer2_outputs_combine.append(decoder_layer2_outputs)

        # 下面的过程把8个shape=(1, 1000)的数组转成8个shape=(1, 1000)的Tensor
        decoder_layer2_outputs_combine = tf.unstack(decoder_layer2_outputs_combine, axis=1)[0]
        decoder_layer2_outputs_combine = tf.unstack(decoder_layer2_outputs_combine, axis=1)[0]
        decoder_layer2_outputs_combine = tf.unstack([decoder_layer2_outputs_combine], axis=1)
        # 重新对decoder_layer2_outputs赋值
        decoder_layer2_outputs = decoder_layer2_outputs_combine

        decoder_layer2_outputs = tf.unstack(decoder_layer2_outputs, axis=1)[0]  # shape=(8, 1000)
        decoder_layer2_outputs = tf.matmul(decoder_layer2_outputs, weights['hid2tar']) + biases['hid2tar']  # shape=(8, 6)

        cost = tf.losses.mean_squared_error(decoder_layer2_outputs, target_outputs)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        return optimizer, cost, decoder_layer2_outputs

    def train(self):
        x = tf.placeholder("float", [None, self.max_seq_len * 2, self.word_vec_dim])
        y = tf.placeholder("float", [None, self.max_seq_len, self.one_hot_word_vectors_dim])

        weights = {
            'enc2dec': tf.Variable(tf.random_normal([self.word_vec_dim, self.one_hot_word_vectors_dim])),
            'hid2tar': tf.Variable(tf.random_normal([self.n_hidden, self.one_hot_word_vectors_dim])),
        }
        biases = {
            'enc2dec': tf.Variable(tf.random_normal([self.max_seq_len, self.one_hot_word_vectors_dim])),
            'hid2tar': tf.Variable(tf.random_normal([self.max_seq_len, self.one_hot_word_vectors_dim])),
        }

        optimizer, cost, decoder_layer2_outputs = self.model(x, y, weights, biases)

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
                if i % 1 == 0 and step == 0:
                    print 'i=%d, loss=%f' % (i, loss)

        saver = tf.train.Saver()
        saver.save(sess, self.model_dir)

    def test(self):
        x = tf.placeholder("float", [None, self.max_seq_len * 2, self.word_vec_dim])
        y = tf.placeholder("float", [None, self.max_seq_len, self.one_hot_word_vectors_dim])

        weights = {
            'enc2dec': tf.Variable(tf.random_normal([self.word_vec_dim, self.one_hot_word_vectors_dim])),
            'hid2tar': tf.Variable(tf.random_normal([self.n_hidden, self.one_hot_word_vectors_dim])),
        }
        biases = {
            'enc2dec': tf.Variable(tf.random_normal([self.max_seq_len, self.one_hot_word_vectors_dim])),
            'hid2tar': tf.Variable(tf.random_normal([self.max_seq_len, self.one_hot_word_vectors_dim])),
        }

        optimizer, cost, decoder_layer2_outputs = self.model(x, y, weights, biases, training=False)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, self.model_dir)

        XY, Y = self.next_batch()
        n_steps = len(XY)
        for step in range(n_steps):
            train_XY = XY[step:]
            train_Y = Y[step:]
            loss = sess.run(cost, feed_dict={x: train_XY, y: train_Y})
            print sess.run(decoder_layer2_outputs, feed_dict={x: train_XY, y: train_Y})
            print 'loss=%f' % loss

    def predict(self):
        x = tf.placeholder("float", [None, self.max_seq_len * 2, self.word_vec_dim])
        y = tf.placeholder("float", [None, self.max_seq_len, self.one_hot_word_vectors_dim])

        weights = {
            'enc2dec': tf.Variable(tf.random_normal([self.word_vec_dim, self.one_hot_word_vectors_dim])),
            'hid2tar': tf.Variable(tf.random_normal([self.n_hidden, self.one_hot_word_vectors_dim])),
        }
        biases = {
            'enc2dec': tf.Variable(tf.random_normal([self.max_seq_len, self.one_hot_word_vectors_dim])),
            'hid2tar': tf.Variable(tf.random_normal([self.max_seq_len, self.one_hot_word_vectors_dim])),
        }

        optimizer, cost, decoder_layer2_outputs = self.model(x, y, weights, biases, training=False)

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
        Y.append([np.zeros(self.one_hot_word_vectors_dim)] * self.max_seq_len)
        output_seq = sess.run(decoder_layer2_outputs, feed_dict={x: XY, y: Y})
        print output_seq
        for vector in output_seq:
            word_id = np.argmax(vector, axis=0)
            print self.word_id_word_dict[word_id]


def main(op):
    np.set_printoptions(threshold='nan')
    lstm = MyLSTM()
    lstm.load_word_vectors()
    lstm.load_one_hot_word_vectors()
    if op == 'train':
        lstm.train()
    elif op == 'predict':
        lstm.predict()
    elif op == 'test':
        lstm.test()
    else:
        print 'Usage:'

if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print 'Usage:'
