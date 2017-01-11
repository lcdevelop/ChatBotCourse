# -*- coding: utf-8 -*-

import sys
import math
import tflearn
import chardet
import numpy as np
import struct

seq = []

max_w = 50
float_size = 4
word_vector_dict = {}

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
            vector.append(weight)

        # 将词及其对应的向量存到dict中
        word_vector_dict[word.decode('utf-8')] = vector

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


def main():
    load_vectors("./vectors.bin")
    init_seq()
    xlist = []
    ylist = []
    test_X = None
    #for i in range(len(seq)-100):
    for i in range(1000):
        sequence = seq[i:i+20]
        xlist.append(sequence)
        ylist.append(seq[i+20])
        if test_X is None:
            test_X = np.array(sequence)
            (match_word, max_cos) = vector2word(seq[i+20])
            print "right answer=", match_word, max_cos

    X = np.array(xlist)
    Y = np.array(ylist)
    net = tflearn.input_data([None, 20, 200])
    net = tflearn.lstm(net, 200)
    net = tflearn.fully_connected(net, 200, activation='linear')
    net = tflearn.regression(net, optimizer='sgd', learning_rate=0.1,
                                     loss='mean_square')
    model = tflearn.DNN(net)
    model.fit(X, Y, n_epoch=500, batch_size=100,snapshot_epoch=False,show_metric=True)
    model.save("model")
    predict = model.predict([test_X])
    #print predict
    #for v in test_X:
    #    print vector2word(v)
    (match_word, max_cos) = vector2word(predict[0])
    print "predict=", match_word, max_cos

main()
#init_seq()
#
#x1 = [1,2,3,4,5]
#x2 = [0,1,2,3]
#y = [2,4,6,8,10]
#X = np.array([x1,x2])
#Y = np.array([y])
#net = tflearn.input_data([None])
##net = tflearn.embedding(net, input_dim=4, output_dim=2)
#net = tflearn.single_unit(net)
##net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,loss='categorical_crossentropy')
#net = tflearn.regression(net, optimizer='sgd', loss='mean_square',
#                                        metric='R2', learning_rate=0.01)
##layer1 = tflearn.fully_connected(net, 2)
#model = tflearn.DNN(net)
#model.fit(x1, y, n_epoch=1000, snapshot_epoch=False,show_metric=True,batch_size=1)
#print model.predict(x1)
##net = tflearn.embedding(net, input_dim=10000, output_dim=16)
#net = tflearn.lstm(net, 16, dropout=0.8)
#net = tflearn.fully_connected(net, 2, activation='softmax')
#net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,loss='categorical_crossentropy')
#model = tflearn.DNN(net, tensorboard_verbose=0)
#model.fit(X, Y, show_metric=True,batch_size=32)


#sys.exit(0)
# 下面都是测试用的，不用看

def test_case1():
    x = [1,2,3]
    y = [0.01,0.99]
    # 多组x作为输入样本
    X = np.array(np.repeat([x], 1, axis=0))
    # 多组y作为样本的y值
    Y = np.array(np.repeat([y], 1, axis=0))

    #X = np.array([x1,x2], dtype=np.float32)
    #Y = np.array([y1,y2])

    # 这里的第二个数对应了x是多少维的向量
    net = tflearn.input_data(shape=[None, 3])
    #net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 2)
    # 这里的第二个参数对应了输出的y是多少维的向量
    #net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net)


    model = tflearn.DNN(net)
    model.fit(X, Y, n_epoch=1000, batch_size=1, show_metric=True, snapshot_epoch=False)
    pred = model.predict([x])
    print(pred)

def case_linear_regression():
    x = [1,2,3,4,5]
    y = [2,4,6,8,10]
    net = tflearn.input_data([None])
    linear = tflearn.single_unit(net)
    net = tflearn.regression(linear, optimizer='sgd', loss='mean_square',
                                            metric='R2', learning_rate=0.01)
    model = tflearn.DNN(net)
    model.fit(x, y, n_epoch=200, snapshot_epoch=False,show_metric=True,batch_size=1)
    print model.predict([8,9]) # [15.990408897399902, 17.988374710083008]
    print model.get_weights(linear.W) # [ 1.99796414]
    print model.get_weights(linear.b) # [ 0.00669619]

#case_linear_regression()


#X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
#Y_xor = [[0.], [1.], [1.], [0.]]

# 如何输出每一步的输出值
# You can re-use a new model that share a same session (to use same weights): . Note that you could also save your 'm' model and load it with 'm2', that gives similar results.
## Graph definition
#with tf.Graph().as_default():
#    tnorm = tflearn.initializations.uniform(minval=-1.0, maxval=1.0)
#    net = tflearn.input_data(shape=[None, 2], name='inputLayer')
#    layer1 = tflearn.fully_connected(net, 2, activation='sigmoid', weights_init=tnorm, name='layer1')
#    layer2 = tflearn.fully_connected(layer1, 1, activation='softmax', weights_init=tnorm, name='layer2')
#    regressor = tflearn.regression(layer2, optimizer='sgd', learning_rate=2., loss='mean_square', name='layer3')
#
#    # Training
#    m = tflearn.DNN(regressor)
#    m.fit(X, Y_xor, n_epoch=100, snapshot_epoch=False) 
#
#    # Testing
#    print("Testing XOR operator")
#    print("0 xor 0:", m.predict([[0., 0.]]))
#    print("0 xor 1:", m.predict([[0., 1.]]))
#    print("1 xor 0:", m.predict([[1., 0.]]))
#    print("1 xor 1:", m.predict([[1., 1.]]))
#
#    # You can create a new model, that share the same session (to get same weights)
#    # Or you can also simply save and load a model
#    m2 = tflearn.DNN(layer1, session=m.session)
#    print(m2.predict([[0., 0.]]))
