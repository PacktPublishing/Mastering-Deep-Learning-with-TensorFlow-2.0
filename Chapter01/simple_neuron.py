import tensorflow as tf
import numpy as np

print(tf.__version__)
#import tensorflow as tf
def simple_neuron(x,y):
    activity = tf.matmul(x,y)
    return tf.nn.relu(activity)

x = tf.placeholder(tf.float64, (3,3))
y = tf.placeholder(tf.float64,(3,3))

out = simple_neuron(x,y)
out = simple_neuron(y,x)

x_input = np.random.normal(0, 0.1, (3,3))
y_input = np.random.normal(0, 0.1, (3,3))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    sess.run(init)
    writer = tf.summary.FileWriter('graphs', sess.graph)
    print(sess.run(out, feed_dict={x:x_input, y:y_input}))

writer.close()
