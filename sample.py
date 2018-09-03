import tensorflow as tf

a = tf.constant("Hello World!")
b = tf.constant(13)
c = tf.constant(27)
d = tf.add(b,c)

with tf.Session() as sess:
    print(sess.run(a).decode())
    print(sess.run(d))

