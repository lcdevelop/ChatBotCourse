"""
样本加载
"""
# coding=utf8

import sys
import random
import jieba
import numpy as np
from word_vectors_loader import get_words_sizes, load_vectors

VECTORS_BIN = 'data/wordvec/vectors.bin'
TEST_COUNT = 5


class DimInfo(object):
    """
    维度信息
    """

    def __init__(self):
        # 词向量有多少维
        self.vec_dim = 0
        # 样本输入的x有多少维
        self.x_dim = 0
        # 当前最大的词编号是多大
        self.max_word_id = -1

    def get_vec_dim(self):
        """
        get_vec_dim
        """
        return self.vec_dim

    def get_x_dim(self):
        """
        get_x_dim
        """
        return self.x_dim


class Maps(object):
    """
    各种映射表
    """

    def __init__(self):
        self.local_word_id_map = {}
        self.local_group_id_map = {"1":0, "2":1, "3":2, "4":3, "5":4}
        self.real_group_id_map = {}
        for key in self.local_group_id_map:
            value = str(self.local_group_id_map[key])
            self.real_group_id_map[value] = int(key)

    def get_local_word_id_map(self):
        """
        get_local_word_id_map
        """
        return self.local_word_id_map

    def get_local_group_id_map(self):
        """
        get_local_group_id_map
        """
        return self.local_group_id_map

    def group_id_size(self):
        """
        获取local_group的数量
        """
        return len(self.local_group_id_map)


class InputData(object):
    """
    样本加载类
    """

    def __init__(self):
        self.data = []
        self.test_data = []
        self.pos = 0
        self.word_vector_dict, self.word_id_dict = load_vectors(VECTORS_BIN)
        self.dim_info = DimInfo()
        self.maps = Maps()
        _, self.dim_info.vec_dim = get_words_sizes(VECTORS_BIN)
        self.dim_info.x_dim = len(self.word_vector_dict) * self.dim_info.vec_dim
        self.maps.local_word_id_map = {}

    def clear_word_vector(self):
        """
        清理点内存
        """
        self.word_vector_dict.clear()
        self.word_id_dict.clear()

    @staticmethod
    def read_data_sets(file_name):
        """
        读取文件，加载数据
        """
        instance = InputData()
        file_object = open(file_name, 'r')
        while True:
            line = file_object.readline(1024)
            if line:
                line = line.strip()
                if len(line) == 0:
                    continue
                split = line.split(' ')
                group_id = 0
                try:
                    group_id = int(split[0])
                except ValueError:
                    continue
                txt = ' '.join(split[1:])
                txt = txt.replace('None', '').strip()
                if len(txt) == 0:
                    continue

                vectors = {}
                seg_list = jieba.cut(txt)
                for seg in seg_list:
                    seg_unicode = seg.encode('utf-8')
                    if seg_unicode in instance.word_vector_dict:
                        word_id = instance.word_id_dict[seg_unicode]
                        if word_id in instance.maps.local_word_id_map:
                            local_word_id = instance.maps.local_word_id_map[word_id]
                            vectors[local_word_id] = instance.word_vector_dict[seg_unicode]
                        else:
                            local_word_id = instance.dim_info.max_word_id
                            instance.maps.local_word_id_map[word_id] = local_word_id
                            vectors[local_word_id] = instance.word_vector_dict[seg_unicode]
                            instance.dim_info.max_word_id = instance.dim_info.max_word_id + 1

                # 稀疏向量
                item = {'vectors':vectors,
                        'local_group_id':instance.maps.local_group_id_map[str(group_id)]}
                instance.data.append(item)
            else:
                break
        file_object.close()

        random.shuffle(instance.data)
        for _ in range(TEST_COUNT):
            instance.test_data.append(instance.data.pop())
        instance.dim_info.x_dim = instance.dim_info.max_word_id * instance.dim_info.vec_dim
        print("max_word_id=", instance.dim_info.max_word_id)
        print("x_dim=", instance.dim_info.x_dim)
        return instance

    def generate_xs(self, txt):
        """
        根据文本生成输入向量
        """
        x_s = []
        vectors = {}
        seg_list = jieba.cut(txt)
        for seg in seg_list:
            seg_unicode = seg.encode('utf-8')
            if seg_unicode in self.word_vector_dict:
                word_id = self.word_id_dict[seg_unicode]
                if word_id in self.maps.local_word_id_map:
                    local_word_id = self.maps.local_word_id_map[word_id]
                    vectors[local_word_id] = self.word_vector_dict[seg_unicode]

        x_array = np.zeros([self.dim_info.x_dim], dtype=np.float)
        for word_id in vectors:
            vector = vectors[word_id]
            for index, weight in enumerate(vector):
                x_array[word_id*self.dim_info.vec_dim+index] = weight
        x_s.append(x_array)
        return x_s


    def next_batch(self, count):
        """
        获取一批样本数据
        """
        x_s = []
        y_s = []
        if self.pos >= len(self.data):
            print("error")
            sys.exit(1)
        while count > 0:
            item = self.data[self.pos]
            vectors = item['vectors']
            local_group_id = item['local_group_id']
            x_array = np.zeros([self.dim_info.x_dim], dtype=np.float)
            y_array = np.zeros(self.maps.group_id_size(), dtype=np.float)
            y_array[local_group_id] = 1
            for word_id in vectors:
                vector = vectors[word_id]
                for index, weight in enumerate(vector):
                    x_array[word_id*self.dim_info.vec_dim+index] = weight
            x_s.append(x_array)
            y_s.append(y_array)
            self.pos = (self.pos + 1) % len(self.data)
            count = count - 1
        return x_s, y_s

    def test_sets(self):
        """
        获取测试样本集
        """
        x_s = []
        y_s = []
        for item in self.test_data:
            vectors = item['vectors']
            local_group_id = item['local_group_id']
            x_array = np.zeros([self.dim_info.x_dim], dtype=np.float)
            y_array = np.zeros(self.maps.group_id_size(), dtype=np.float)
            y_array[local_group_id] = 1
            for word_id in vectors:
                vector = vectors[word_id]
                for index, weight in enumerate(vector):
                    x_array[word_id*self.dim_info.vec_dim+index] = weight
            x_s.append(x_array)
            y_s.append(y_array)
        return x_s, y_s


if __name__ == '__main__':
    CLUES = InputData.read_data_sets('./data/sample/samples')
    XS, YS = CLUES.next_batch(2)
    print(XS)
    print(XS[0].shape)
    print(YS)
