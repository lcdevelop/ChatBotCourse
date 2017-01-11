# coding:utf-8

import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import rnn_cell
import tflearn

max_seq_len = 8
learning_rate = 0.01
id_word_dict = {}

# 得到了单词转id的词典是word_id_dict, 最大单词id是max_word_id
def init_word_id_dict():
    word_id_dict = {}
    max_word_id = 0
    threshold = max_seq_len
    vocab_dict = {}
    # 把每个词映射到一个整数编号word_id
    file_object = open("chat_dev.data", "r")
    while True:
        line = file_object.readline()
        if line:
            line = line.strip()
            words = []
            for word in line.split(" "):
                if len(word) > 0:
                    words.append(word)
            if len(words) > threshold:
                continue

            for word in words:
                if len(word)>0:
                    if vocab_dict.has_key(word):
                        vocab_dict[word] = vocab_dict[word] + 1
                    else:
                        vocab_dict[word] = 1
        else:
            break
    file_object.close()

    vocab_dict = sorted(vocab_dict.items(), key=lambda d: d[1], reverse = True)

    uuid = 1

    max_word_id=2000
    for (word, freq) in vocab_dict:
        word_id_dict[word] = uuid
        id_word_dict[uuid] = word
        #if freq > 20:
        #    print word, uuid, freq
        print word, uuid, freq
        uuid = uuid + 1
        if uuid > max_word_id:
            break

    return (word_id_dict, max_word_id)


def sequence_loss(y_pred, y_true):
    logits = tf.unpack(y_pred, axis=1)
    targets = tf.unpack(y_true, axis=1)
    weights = [tf.ones_like(yp, dtype=tf.float32) for yp in targets]
    return seq2seq.sequence_loss(logits, targets, weights)

def accuracy(y_pred, y_true, x_in):
    pred_idx = tf.to_int32(tf.argmax(y_pred, 2))
    return tf.reduce_mean(tf.cast(tf.equal(pred_idx, y_true), tf.float32), name='acc')

def create_model(max_word_id, is_test=False):
    GO_VALUE = max_word_id + 1
    network = tflearn.input_data(shape=[None, max_seq_len + max_seq_len], dtype=tf.int32, name="XY")
    encoder_inputs = tf.slice(network, [0, 0], [-1, max_seq_len], name="enc_in")
    encoder_inputs = tf.unpack(encoder_inputs, axis=1)
    decoder_inputs = tf.slice(network, [0, max_seq_len], [-1, max_seq_len], name="dec_in")
    decoder_inputs = tf.unpack(decoder_inputs, axis=1)
    go_input = tf.mul( tf.ones_like(decoder_inputs[0], dtype=tf.int32), GO_VALUE )
    decoder_inputs = [go_input] + decoder_inputs[: max_max_seq_len-1]
    num_encoder_symbols = max_word_id + 1 # 从0起始
    num_decoder_symbols = max_word_id + 2 # 包括GO

    cell = rnn_cell.BasicLSTMCell(16*max_seq_len, state_is_tuple=True)

    model_outputs, states = seq2seq.embedding_rnn_seq2seq(
            encoder_inputs,
            decoder_inputs,
            cell,
            num_encoder_symbols=num_encoder_symbols,
            num_decoder_symbols=num_decoder_symbols,
            embedding_size=max_word_id,
            feed_previous=is_test)

    network = tf.pack(model_outputs, axis=1)




    targetY = tf.placeholder(shape=[None, max_seq_len], dtype=tf.float32, name="Y")

    network = tflearn.regression(
            network,
            placeholder=targetY,
            optimizer='adam',
            learning_rate=learning_rate,
            loss=sequence_loss,
            metric=accuracy,
            name="Y")

    print "begin create DNN model"
    model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path=None)
    print "create DNN model finish"
    return model

def print_sentence(list, msg):
    sentence = msg
    for item in list:
        if item != 0:
            sentence = sentence + id_word_dict[item]
    print sentence

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        is_test = True
    else:
        is_test = False
    (word_id_dict, max_word_id) = init_word_id_dict()
    print "max_word_id =", max_word_id

    model = create_model(max_word_id, is_test)

    threshold = max_seq_len
    file_object = open("chat_dev.data", "r")
    last_line_no = 0
    cur_line_no = 0
    last_words = []
    last_line = ""
    XY = []
    Y = []
    sample_count = 0
    while True:
        line = file_object.readline()
        cur_line_no = cur_line_no + 1
        if line:
            line = line.strip()
            words = []
            for word in line.split(" "):
                if len(word) > 0:
                    words.append(word)
            if len(words) > threshold:
                continue

            # 保证连续的话才参与训练
            if last_line_no != 0 and last_line_no == cur_line_no - 1:
                question_id_list = []
                question = ""
                answer = ""
                question_array = np.zeros(max_seq_len + max_seq_len)
                answer_array = np.zeros(max_seq_len)
                idx = 0
                question_has_word = False
                answer_has_word = False
                for word in last_words:
                    if len(word)>0 and word_id_dict.has_key(word):
                        word_id = word_id_dict[word]
                        question_id_list.append(word_id)
                        question = question + word
                        question_array[idx] = word_id
                        question_has_word = True
                        idx = idx + 1
                for i in range(max_seq_len - len(question_id_list)):
                    question_id_list.append(0)

                answer_id_list = []

                idx = 0
                for word in words:
                    if len(word)>0 and word_id_dict.has_key(word):
                        word_id = word_id_dict[word]
                        answer_id_list.append(word_id)
                        answer = answer + word
                        question_array[max_seq_len + idx] = word_id
                        answer_array[idx] = word_id
                        answer_has_word = True
                        idx = idx + 1
                for i in range(2*max_seq_len - len(question_id_list)):
                    answer_id_list.append(0)
                question_id_list.extend(answer_id_list)

                if question_has_word and answer_has_word:
                    #print "question =", question
                    #print "answer =", answer
                    XY.append(question_array)
                    Y.append(answer_array)
                    sample_count = sample_count + 1

            last_words = words
            last_line = line
            last_line_no = cur_line_no

        else:
            break
    file_object.close()

    if not is_test:
        model.fit(
                XY,
                Y,
                n_epoch=3000,
                validation_set=0.01,
                batch_size=64,
                shuffle=True,
                show_metric=True,
                snapshot_step=5000,
                snapshot_epoch=False,
                run_id="my_lstm_test")

        model.save("./weights")
    else:
        model.load("./weights")


    # predict
    for i in range(100):
        TEST_XY = [XY[i]]
        TEST_XY[0][max_seq_len:2*max_seq_len] = 0
        #TEST_XY[0][0:2*max_seq_len] = 0
        #TEST_XY[0][0] = 5
        #TEST_XY[0][1] = 4
        #TEST_XY[0][2] = 109

        res = model.predict(TEST_XY)
        res = np.array(res)
        num_decoder_symbols = max_word_id + 2
        y = res.reshape(max_seq_len, num_decoder_symbols)
        prediction = np.argmax(y, axis=1)
        if 0 != np.sum(prediction):
            print_sentence(TEST_XY[0], "input ")
            print_sentence(Y[i], "desire ")
            print_sentence(prediction, "prediction ")
