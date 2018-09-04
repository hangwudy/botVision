import tensorflow as tf

# save to file

W = tf.Variable([[1, 2, 8], [4, 5, 8]], dtype=tf.float32, name='weight')

b = tf.Variable([1, 2, 8], dtype=tf.float32, name='biases')

init = tf.global_variables_initializer()

saver = tf.train.Saver()


with tf.Session() as sess:

    sess.run(init)

    save_path = saver.save(sess, '/tmp/model.ckpt')

    print("Model saved to path: %s!" % save_path)
