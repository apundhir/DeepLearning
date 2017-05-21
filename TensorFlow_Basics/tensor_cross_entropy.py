import tensorflow as tf

## x = tf.reduce_sum([1, 2, 3, 4, 5])  # 15 sum of vector
## x = tf.log(100)  # 4.60517 natural log

softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)

# TODO: Print cross entropy from session
with tf.Session() as session:
    loss = -tf.reduce_sum(tf.multiply(one_hot, tf.log(softmax)))
    result = session.run(loss, feed_dict={softmax:softmax_data, one_hot:one_hot_data})
    print(result)
