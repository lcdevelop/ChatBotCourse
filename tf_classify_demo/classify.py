"""
利用tensorflow做图书分类模型训练
"""
#!/usr/bin/env python
# coding=utf8

import sys
import tensorflow as tf
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse as parse
from sample_data import InputData

samples = InputData.read_data_sets('./data/sample/samples')
config = tf.ConfigProto(device_count={'CPU':4})
sess = tf.InteractiveSession(config=config)
feature_len = samples.dim_info.x_dim
x = tf.placeholder(tf.float32, [None, feature_len])
W = tf.Variable(tf.zeros([feature_len, samples.maps.group_id_size()]))
b = tf.Variable(tf.zeros([samples.maps.group_id_size()]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, samples.maps.group_id_size()])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)),\
        reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
tf.global_variables_initializer().run()
saver = tf.train.Saver()

def train(samples, sess, x, y, y_, train_step):
    """
    利用无隐藏层的softmax实现简单的分类模型
    """

    samples.clear_word_vector()
    test_xs, test_ys = samples.test_sets()

    for i in range(10000):
        batch_xs, batch_ys = samples.next_batch(1)
        train_step.run({x: batch_xs, y_: batch_ys})

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(accuracy.eval({x: test_xs, y_: test_ys}))
        saver.save(sess, 'data/model/model')

def predict(samples, sess, x, y, y_, train_step):
    x_s = samples.generate_xs('数据科学入门')
    print(sess.run(tf.argmax(y, 1), feed_dict={x:x_s}))

class MyServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        arg_dict = parse.urlparse(self.path)
        if len(arg_dict.query) > 0 and 'q' in parse.parse_qs(arg_dict.query):
            q = parse.parse_qs(arg_dict.query)['q'][0]
            x_s = samples.generate_xs(q)
            local_group_id = sess.run(tf.argmax(y, 1), feed_dict={x:x_s})[0]
            group_id = samples.maps.real_group_id_map[str(local_group_id)]
            print("q=", q, "group_id=", group_id)
            self.wfile.write(bytes(str(group_id), "utf-8"))

def main(is_predict):
    if is_predict:
        saver.restore(sess, 'data/model/model')
        #predict(samples, sess, x, y, y_, train_step)
        myServer = HTTPServer(("0.0.0.0", 5001), MyServer)
        print("begin listen")
        myServer.serve_forever()
    else:
        train(samples, sess, x, y, y_, train_step)


if __name__ == '__main__':
    is_predict = True
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        is_predict = False
    main(is_predict)
