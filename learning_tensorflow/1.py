import tensorflow as tf

sess = tf.Session()

a = tf.placeholder("float")
b = tf.placeholder("float")
c = tf.constant(6.0)
d = tf.mul(a, b)
y = tf.mul(d, c)
print sess.run(y, feed_dict={a: 3, b: 3})

A = [[1.1,2.3],[3.4,4.1]]
Y = tf.matrix_inverse(A)
print sess.run(Y)
sess.close()
