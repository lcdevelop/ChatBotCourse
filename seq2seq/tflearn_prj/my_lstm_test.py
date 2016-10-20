# coding:utf-8


import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import rnn_cell
import tflearn

np.set_printoptions(threshold=np.nan)

class Primes:
    def __init__(self):
        self.primes = list()
        for i in range(2, 100):
            is_prime = True
            for j in range(2, i-1):
                if i % j == 0:
                    is_prime = False
            if is_prime:
                self.primes.append(i)
        self.primes_count = len(self.primes)
    def get_sample(self, x_dim, y_dim, index):
        result = np.zeros((x_dim+y_dim))
        for i in range(index, index + x_dim + y_dim):
            result[i-index] = self.primes[i%self.primes_count]
        return result


max_input_len = 10
max_output_len = 10
embedding_size = 20
max_int = 100
GO_VALUE = max_int + 1
learning_rate = 0.01

network = tflearn.input_data(shape=[None, max_input_len + max_output_len], dtype=tf.int32, name="XY")
encoder_inputs = tf.slice(network, [0, 0], [-1, max_input_len], name="enc_in")
encoder_inputs = tf.unpack(encoder_inputs, axis=1)
decoder_inputs = tf.slice(network, [0, max_input_len], [-1, max_output_len], name="dec_in")
decoder_inputs = tf.unpack(decoder_inputs, axis=1)
go_input = tf.mul( tf.ones_like(decoder_inputs[0], dtype=tf.int32), GO_VALUE )
decoder_inputs = [go_input] + decoder_inputs[: max_output_len-1]
num_encoder_symbols = max_int + 1 # 从0起始
num_decoder_symbols = max_int + 2 # 包括GO
print encoder_inputs
print decoder_inputs

cell = rnn_cell.BasicLSTMCell(16, state_is_tuple=True)

model_outputs, states = seq2seq.embedding_rnn_seq2seq(
        encoder_inputs,
        decoder_inputs,
        cell,
        num_encoder_symbols=num_encoder_symbols,
        num_decoder_symbols=num_decoder_symbols,
        embedding_size=embedding_size,
        feed_previous=False)

network = tf.pack(model_outputs, axis=1)



def sequence_loss(y_pred, y_true):
    logits = tf.unpack(y_pred, axis=1)
    targets = tf.unpack(y_true, axis=1)
    weights = [tf.ones_like(yp, dtype=tf.float32) for yp in targets]
    return seq2seq.sequence_loss(logits, targets, weights)

def accuracy(y_pred, y_true, x_in):
    pred_idx = tf.to_int32(tf.argmax(y_pred, 2))
    return tf.reduce_mean(tf.cast(tf.equal(pred_idx, y_true), tf.float32), name='acc')

targetY = tf.placeholder(shape=[None, max_output_len], dtype=tf.int32, name="Y")

network = tflearn.regression(
        network,
        placeholder=targetY,
        optimizer='adam',
        learning_rate=learning_rate,
        loss=sequence_loss,
        metric=accuracy,
        name="Y")

model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path=None)

primes = Primes()
XY = [ primes.get_sample(10, 10, i)[0:20] for i in range(10) ]
Y = [ primes.get_sample(10, 10, i)[10:20] for i in range(10) ]
model.fit(
        XY,
        Y,
        n_epoch=10,
        validation_set=0.01,
        batch_size=1,
        shuffle=True,
        show_metric=True,
        snapshot_step=50,
        snapshot_epoch=False,
        run_id="my_lstm_test")


TEST_XY = [XY[0]]
TEST_XY[0][10:20]=0
res = model.predict(TEST_XY)
print TEST_XY
res = np.array(res)
print res.shape
y = res.reshape(max_output_len, num_decoder_symbols)
prediction = np.argmax(y, axis=1)
print prediction


