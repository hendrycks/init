# Scroll down to the "% save_every == 0" condition and modify the code there if
# the experiment is too slow

# import MNIST data, Tensorflow, and other helpers
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")
import tensorflow as tf
import numpy as np
import sys
import os
import pickle

# training parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 128

# architecture parameters
n_hidden = 256
n_labels = 10
image_pixels = 28 * 28

try:
    nonlinearity_name = sys.argv[1]       # 'relu', 'elu', 'gelu', 'tanh'
except:
    print('Defaulted to relu since no nonlinearity specified through command line')
    nonlinearity_name = 'relu'

try:
    initialization_scheme = sys.argv[2]       # 'xavier', 'he', 'ours', 'ours_no_back'
except:
    print('Defaulted to xavier since no initialization scheme specified through command line')
    initialization_scheme = 'xavier'

try:
    keep_rate = float(sys.argv[3])          # 0-1
except:
    print('Defaulted to 1 dropout preservation probability')
    keep_rate = 1.

p = keep_rate

# E[f(z^l)^2]
energy_preserved = {'gelu': 0.425, 'relu': 0.5, 'elu': 0.645, 'tanh': 0.394}

# E[f'(z^(l+1))^2]
back_correction = {'gelu': 0.444, 'relu': 0.5, 'elu': 0.671, 'tanh': 0.216}

x = tf.placeholder(dtype=tf.float32, shape=[None, image_pixels])
y = tf.placeholder(dtype=tf.int64, shape=[None])
is_training = tf.placeholder(tf.bool)

if initialization_scheme == 'xavier':
    W = {
        '1': tf.Variable(tf.random_uniform([image_pixels, n_hidden], minval=-1, maxval=1)/tf.sqrt((image_pixels + n_hidden)/6.)),
        '2': tf.Variable(tf.random_uniform([n_hidden, n_hidden], minval=-1, maxval=1)/tf.sqrt(n_hidden/3.)),    # 2*n_hidden/6
        '3': tf.Variable(tf.random_uniform([n_hidden, n_hidden], minval=-1, maxval=1)/tf.sqrt(n_hidden/3.)),
        '4': tf.Variable(tf.random_uniform([n_hidden, n_hidden], minval=-1, maxval=1)/tf.sqrt(n_hidden/3.)),
        '5': tf.Variable(tf.random_uniform([n_hidden, n_hidden], minval=-1, maxval=1)/tf.sqrt(n_hidden/3.)),
        '6': tf.Variable(tf.random_uniform([n_hidden, n_hidden], minval=-1, maxval=1)/tf.sqrt(n_hidden/3.)),
        '7': tf.Variable(tf.random_uniform([n_hidden, n_hidden], minval=-1, maxval=1)/tf.sqrt(n_hidden/3.)),
        '8': tf.Variable(tf.random_uniform([n_hidden, n_labels], minval=-1, maxval=1)/tf.sqrt((n_hidden + n_labels)/6.))
    }

elif initialization_scheme == 'he':
    W = {
        '1': tf.Variable(tf.random_normal([image_pixels, n_hidden])/tf.sqrt(0.5*image_pixels)),
        '2': tf.Variable(tf.random_normal([n_hidden, n_hidden])/tf.sqrt(0.5*n_hidden)),
        '3': tf.Variable(tf.random_normal([n_hidden, n_hidden])/tf.sqrt(0.5*n_hidden)),
        '4': tf.Variable(tf.random_normal([n_hidden, n_hidden])/tf.sqrt(0.5*n_hidden)),
        '5': tf.Variable(tf.random_normal([n_hidden, n_hidden])/tf.sqrt(0.5*n_hidden)),
        '6': tf.Variable(tf.random_normal([n_hidden, n_hidden])/tf.sqrt(0.5*n_hidden)),
        '7': tf.Variable(tf.random_normal([n_hidden, n_hidden])/tf.sqrt(0.5*n_hidden)),
        '8': tf.Variable(tf.random_normal([n_hidden, n_labels])/tf.sqrt(0.5*n_hidden))
    }

elif initialization_scheme == 'ours':
    c_forward = energy_preserved[nonlinearity_name]
    c_backward = back_correction[nonlinearity_name]
    W = {
        '1': tf.Variable(tf.nn.l2_normalize(tf.random_normal([image_pixels, n_hidden]), 0)/tf.sqrt(1+c_backward*p)),
        '2': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden]), 0)/tf.sqrt(c_forward/p + c_backward*p)),
        '3': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden]), 0)/tf.sqrt(c_forward/p + c_backward*p)),
        '4': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden]), 0)/tf.sqrt(c_forward/p + c_backward*p)),
        '5': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden]), 0)/tf.sqrt(c_forward/p + c_backward*p)),
        '6': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden]), 0)/tf.sqrt(c_forward/p + c_backward*p)),
        '7': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden]), 0)/tf.sqrt(c_forward/p + c_backward*p)),
        '8': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_labels]), 0)/tf.sqrt(1+c_forward/p))
    }


b = {
    '1': tf.Variable(tf.zeros([n_hidden])),
    '2': tf.Variable(tf.zeros([n_hidden])),
    '3': tf.Variable(tf.zeros([n_hidden])),
    '4': tf.Variable(tf.zeros([n_hidden])),
    '5': tf.Variable(tf.zeros([n_hidden])),
    '6': tf.Variable(tf.zeros([n_hidden])),
    '7': tf.Variable(tf.zeros([n_hidden])),
    '8': tf.Variable(tf.zeros([n_labels]))
}

def feedforward(x):
    if nonlinearity_name == 'relu':
        rho = tf.nn.relu
    elif nonlinearity_name == 'elu':
        rho = tf.nn.elu
    elif nonlinearity_name == 'gelu':
        def gelu(x):
            return tf.mul(x, tf.erfc(-x / tf.sqrt(2.)) / 2.)
        rho = gelu
    elif nonlinearity_name == 'tanh':
        rho = tf.tanh
    elif nonlinearity_name == 'gelu':
        def gelu(x):
            return tf.mul(x, tf.erfc(-x/tf.sqrt(2.))/2.)
        rho = gelu
    else:
        raise NameError("Need 'relu', 'elu', 'gelu', or 'tanh' for nonlinearity_name")

    h1 = rho(tf.matmul(x, W['1']) + b['1'])
    h1 = tf.cond(is_training, lambda: tf.nn.dropout(h1, keep_rate), lambda: h1)
    h2 = rho(tf.matmul(h1, W['2']) + b['2'])
    h2 = tf.cond(is_training, lambda: tf.nn.dropout(h2, keep_rate), lambda: h2)
    h3 = rho(tf.matmul(h2, W['3']) + b['3'])
    h3 = tf.cond(is_training, lambda: tf.nn.dropout(h3, keep_rate), lambda: h3)
    h4 = rho(tf.matmul(h3, W['4']) + b['4'])
    h4 = tf.cond(is_training, lambda: tf.nn.dropout(h4, keep_rate), lambda: h4)
    h5 = rho(tf.matmul(h4, W['5']) + b['5'])
    h5 = tf.cond(is_training, lambda: tf.nn.dropout(h5, keep_rate), lambda: h5)
    h6 = rho(tf.matmul(h5, W['6']) + b['6'])
    h6 = tf.cond(is_training, lambda: tf.nn.dropout(h6, keep_rate), lambda: h6)
    h7 = rho(tf.matmul(h6, W['7']) + b['7'])
    h7 = tf.cond(is_training, lambda: tf.nn.dropout(h7, keep_rate), lambda: h7)
    return tf.matmul(h7, W['8']) + b['8']


logits = feedforward(x)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
compute_error = tf.not_equal(tf.argmax(logits, 1), y)

# store future results with previous results
if not os.path.exists("./data/"):
    os.makedirs("./data/")

if os.path.exists("./data/" + nonlinearity_name + '_' + initialization_scheme + '_' +
                               str(keep_rate) + "_history.p"):
    history = pickle.load(open("./data/" + nonlinearity_name + '_' + initialization_scheme + '_' +
                               str(keep_rate) + "_history.p", "rb"))
    key_str = str(len(history)//3 + 1)
    history["loss" + key_str] = []
    history["loss_val" + key_str] = []
    history["test" + key_str] = 1
else:
    history = {'loss1': [], 'loss_val1': [], 'test1': 1}
    key_str = '1'


with tf.Session() as sess:
    print('Beginning training')
    sess.run(tf.initialize_all_variables())

    num_batches = int(mnist.train.num_examples / batch_size)
    save_every = int(batch_size/3.1)      # save training information 3 times per epoch
    best_val_ce = 3     # -log(0.1) s

    for epoch in range(training_epochs):
        average_loss = 0
        for i in range(num_batches):
            bx, by = mnist.train.next_batch(batch_size)
            _, l = sess.run([optimizer, loss], feed_dict={x: bx, y: by, is_training: True})
            average_loss += l / num_batches

            if i % save_every == 0:
                # we feed it all forward to we can have smooth graphs
				# MODIFY THIS SECTION IF YOU WANT TO RUN THIS EXPERIMENT MORE QUICKLY
                l = sess.run(loss, feed_dict={x: mnist.train.images, y: mnist.train.labels, is_training: True})
                history['loss' + key_str].append(l)


                l_val = sess.run(loss, feed_dict={x: mnist.validation.images, y: mnist.validation.labels,
                                                     is_training: False})
                history['loss_val' + key_str].append(l_val)
                if l_val < best_val_ce:
                    t = sess.run(tf.reduce_mean(tf.to_float(compute_error)),
                                                         feed_dict={x: mnist.test.images, y: mnist.test.labels,
                                                                    is_training: False})
                    history['test' + key_str] = t
                    best_val_ce = l_val


        print('Epoch:', epoch, '|', 'average loss for epoch:', average_loss)

    print('Test error (%):', 100*history['test' + key_str])

# save history
pickle.dump(history, open("./data/" + nonlinearity_name + '_' + initialization_scheme + '_' +
                               str(keep_rate) + "_history.p", "wb"))
