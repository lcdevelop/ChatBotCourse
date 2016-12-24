# -*- coding: utf-8 -*-

import sys
import math
import tflearn
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn
import chardet
import numpy as np
import struct

seq = []

max_w = 50
float_size = 4
word_vector_dict = {}
word_vec_dim = 200
max_seq_len = 16

def load_vectors(input):
    """从vectors.bin加载词向量，返回一个word_vector_dict的词典，key是词，value是200维的向量
    """
    print "begin load vectors"

    input_file = open(input, "rb")

    # 获取词表数目及向量维度
    words_and_size = input_file.readline()
    words_and_size = words_and_size.strip()
    words = long(words_and_size.split(' ')[0])
    size = long(words_and_size.split(' ')[1])
    print "words =", words
    print "size =", size

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
        for index in range(0, size):
            m = input_file.read(float_size)
            (weight,) = struct.unpack('f', m)
            vector.append(float(weight))

        # 将词及其对应的向量存到dict中
        #word_vector_dict[word.decode('utf-8')] = vector
        word_vector_dict[word.decode('utf-8')] = vector[0:word_vec_dim]

    input_file.close()

    print "load vectors finish"

def init_seq():
    """读取切好词的文本文件，加载全部词序列
    """
    file_object = open('zhenhuanzhuan.segment', 'r')
    vocab_dict = {}
    while True:
        line = file_object.readline()
        if line:
            for word in line.decode('utf-8').split(' '):
                if word_vector_dict.has_key(word):
                    seq.append(word_vector_dict[word])
        else:
            break
    file_object.close()

def vector_sqrtlen(vector):
    len = 0
    for item in vector:
        len += item * item
    len = math.sqrt(len)
    return len

def vector_cosine(v1, v2):
    if len(v1) != len(v2):
        sys.exit(1)
    sqrtlen1 = vector_sqrtlen(v1)
    sqrtlen2 = vector_sqrtlen(v2)
    value = 0
    for item1, item2 in zip(v1, v2):
        value += item1 * item2
    return value / (sqrtlen1*sqrtlen2)


def vector2word(vector):
    max_cos = -10000
    match_word = ''
    for word in word_vector_dict:
        v = word_vector_dict[word]
        cosine = vector_cosine(vector, v)
        if cosine > max_cos:
            max_cos = cosine
            match_word = word
    return (match_word, max_cos)


class MySeq2Seq(object):
    """
    思路：输入输出序列一起作为input，然后通过slick和unpack切分
    完全按照论文说的编码器解码器来做
    输出的时候把解码器的输出按照词向量的200维展平，这样输出就是(?,seqlen*200)
    这样就可以通过regression来做回归计算了，输入的y也展平，保持一致
    """
    def __init__(self, max_seq_len = 16, word_vec_dim = 200):
        self.max_seq_len = max_seq_len
        self.word_vec_dim = word_vec_dim

    def generate_trainig_data(self):
        load_vectors("./vectors.bin")
        init_seq()
        xy_data = []
        y_data = []
        for i in range(30,40,10):
            # 问句、答句都是16字，所以取32个
            start = i*self.max_seq_len*2
            middle = i*self.max_seq_len*2 + self.max_seq_len
            end = (i+1)*self.max_seq_len*2
            sequence_xy = seq[start:end]
            sequence_y = seq[middle:end]
            print "right answer"
            for w in sequence_y:
                (match_word, max_cos) = vector2word(w)
                print match_word
            sequence_y = [np.ones(self.word_vec_dim)] + sequence_y
            xy_data.append(sequence_xy)
            y_data.append(sequence_y)

        return np.array(xy_data), np.array(y_data)


    def model(self, feed_previous=False):
        # 通过输入的XY生成encoder_inputs和带GO头的decoder_inputs
        input_data = tflearn.input_data(shape=[None, self.max_seq_len*2, self.word_vec_dim], dtype=tf.float32, name = "XY")
        encoder_inputs = tf.slice(input_data, [0, 0, 0], [-1, self.max_seq_len, self.word_vec_dim], name="enc_in")
        decoder_inputs_tmp = tf.slice(input_data, [0, self.max_seq_len, 0], [-1, self.max_seq_len-1, self.word_vec_dim], name="dec_in_tmp")
        go_inputs = tf.ones_like(decoder_inputs_tmp)
        go_inputs = tf.slice(go_inputs, [0, 0, 0], [-1, 1, self.word_vec_dim])
        decoder_inputs = tf.concat(1, [go_inputs, decoder_inputs_tmp], name="dec_in")

        # 编码器
        # 把encoder_inputs交给编码器，返回一个输出(预测序列的第一个值)和一个状态(传给解码器)
        (encoder_output_tensor, states) = tflearn.lstm(encoder_inputs, self.word_vec_dim, return_state=True, scope='encoder_lstm')
        encoder_output_sequence = tf.pack([encoder_output_tensor], axis=1)

        # 解码器
        # 预测过程用前一个时间序的输出作为下一个时间序的输入
        # 先用编码器的最后一个输出作为第一个输入
        if feed_previous:
            first_dec_input = go_inputs
        else:
            first_dec_input = tf.slice(decoder_inputs, [0, 0, 0], [-1, 1, self.word_vec_dim])
        decoder_output_tensor = tflearn.lstm(first_dec_input, self.word_vec_dim, initial_state=states, return_seq=False, reuse=False, scope='decoder_lstm')
        decoder_output_sequence_single = tf.pack([decoder_output_tensor], axis=1)
        decoder_output_sequence_list = [decoder_output_tensor]
        # 再用解码器的输出作为下一个时序的输入
        for i in range(self.max_seq_len-1):
            if feed_previous:
                next_dec_input = decoder_output_sequence_single
            else:
                next_dec_input = tf.slice(decoder_inputs, [0, i+1, 0], [-1, 1, self.word_vec_dim])
            decoder_output_tensor = tflearn.lstm(next_dec_input, self.word_vec_dim, return_seq=False, reuse=True, scope='decoder_lstm')
            decoder_output_sequence_single = tf.pack([decoder_output_tensor], axis=1)
            decoder_output_sequence_list.append(decoder_output_tensor)

        decoder_output_sequence = tf.pack(decoder_output_sequence_list, axis=1)
        real_output_sequence = tf.concat(1, [encoder_output_sequence, decoder_output_sequence])

        net = tflearn.regression(real_output_sequence, optimizer='sgd', learning_rate=0.1, loss='mean_square')
        model = tflearn.DNN(net)
        return model

    def train(self):
        trainXY, trainY = self.generate_trainig_data()
        model = self.model(feed_previous=False)
        model.fit(trainXY, trainY, n_epoch=1000, snapshot_epoch=False)
        model.save('./model/model')
        return model

    def load(self):
        model = self.model(feed_previous=True)
        model.load('./model/model')
        return model

if __name__ == '__main__':
    phrase = sys.argv[1]
    my_seq2seq = MySeq2Seq(word_vec_dim=word_vec_dim, max_seq_len=max_seq_len)
    if phrase == 'train':
        my_seq2seq.train()
    else:
        model = my_seq2seq.load()
        trainXY, trainY = my_seq2seq.generate_trainig_data()
        predict = model.predict(trainXY)
        for sample in predict:
            print "predict answer"
            for w in sample[1:]:
                (match_word, max_cos) = vector2word(w)
                print match_word, max_cos
