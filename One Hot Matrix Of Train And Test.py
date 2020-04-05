import tensorflow as tf

vocab = ['British ShortHair Cat','Persian Cat','Maine Coon Cat','Siamese Cat','Bombay Cat','Chartreux Cat']

input = tf.placeholder(dtype=tf.string, shape=[356,1])
matches = tf.stack([tf.equal(input, s) for s in vocab], axis=-1)
onehot = tf.cast(matches, tf.int8)

with tf.Session() as sess:
    out = sess.run(onehot, feed_dict={input: Y_train})
    
Y_train=out

input1 = tf.placeholder(dtype=tf.string, shape=[26,1])
matches1 = tf.stack([tf.equal(input1, s) for s in vocab], axis=-1)
onehot1 = tf.cast(matches1, tf.int8)

with tf.Session() as sess:
    out1 = sess.run(onehot1, feed_dict={input1: Y_test})
    
Y_test=out1