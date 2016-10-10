# coding:utf-8

import sys
import struct
import math
import numpy as np

reload(sys)
sys.setdefaultencoding( "utf-8" )

max_w = 50
float_size = 4

def load_vectors(input):
    print "begin load vectors"

    input_file = open(input, "rb")

    # 获取词表数目及向量维度
    words_and_size = input_file.readline()
    words_and_size = words_and_size.strip()
    words = long(words_and_size.split(' ')[0])
    size = long(words_and_size.split(' ')[1])
    print "words =", words
    print "size =", size

    word_vector = {}

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

        # 读取词向量
        vector = np.empty([200])
        for index in range(0, size):
            m = input_file.read(float_size)
            (weight,) = struct.unpack('f', m)
            vector[index] = weight

        # 将词及其对应的向量存到dict中
        word_vector[word.decode('utf-8')] = vector

    input_file.close()

    print "load vectors finish"
    return word_vector

if __name__ == '__main__':
    if 2 != len(sys.argv):
        print "Usage: ", sys.argv[0], "vectors.bin"
        sys.exit(-1)
    d = load_vectors(sys.argv[1])
    print d[u'真的']
