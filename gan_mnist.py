import matplotlib
matplotlib.use('Agg')

import os, time, itertools, imageio, pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def generator(x):

	w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
	b_init = tf.constant_initializer(0.)

	w0 = get_variable('G_w0', [x.get_shape()[1], 256], initializer=w_init)
	b0 = get_variable('G_b0', [256], initializer=b_init)
	h0 = tf.nn.relu(tf.matmul(x, w0) + b0)

	w1 = get_variable('G_w1', [h0.get_shape()[1], 512], initializer=w_init)
	b1 = get_variable('G_b1', [512], initializer=b_init)
	h1 = tf.nn.relu(tf.matmul(h0, w1) + b1)

	w2 = get_variable('G_w2', [h1.get_shape()[1], 1024], initializer=w_init)
	b2 = get_variable('G_b2', [1024], initializer=b_init)
	h2 = tf.nn.relu(tf.matmul(g1, w2) + b2)

	w3 = get_variable('G_w3', [h2.get_shape()[1], 784], initializer=w_init)
	b3 = get_variable('G_b3', [784], initializer=b_init)
	h3 = tf.nn.relu(tf.matmul(h2, w3) + b3)

	return h3

def discriminator(x, drop_out):

	w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
	b_init = tf.constant_initializer(0.)

	w0 = tf.variable('D_w0', [x.get_shape()[1], 1024], initializer=w_init)
	b0 = tf.variable('D_b0', [1024], initializer=b_init)
	h0 = tf.nn.relu(tf.matmul(x, w0) + b0)
	h0 = tf.nn.dropout(h0, drop_out)

	w1 = tf.variable('D_w1', [h0.get_shape()[1], 512], initializer=w_init)
	b1 = tf.variable('D_b1', [512], initializer=b_init)
	h1 = tf.nn.relu(tf.matmul(h0, w1) + b1)
	h1 = tf.nn.dropout(h1, drop_out)

	w2 = tf.variable('D_w2', [h1.get_shape()[1], 256], initializer=w_init)
	b2 = tf.variable('D_b2', [256], initializer=b_init)
	h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
	h2 = tf.nn.dropout(h2, drop_out)

	w3 = tf.variable('D_w3', [h2.get_shape()[1], 1], initializer=w_init)
	b3 = tf.variable('D_b3', [1], initializer=b_init)
	h3 = tf.nn.relu(tf.matmul(h2, w3) + b3)

	return h3

batch_size = 100
lr = 0.0002
train_epoch = 100

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_set = (mnist.train.images - 0.5) / 0.5  # normalization; range: -1 ~ 1

with tf.variable_scope('G'):
	z = tf.placeholder(tf.float32, shape=(None, 100))
	G_z = generator(z)

with tf.variable_scope('D') as scope:
	drop_out = tf.placeholder(dtype=tf.float32, name='drop_out')
	x = tf.placeholder(tf.float32, shape=(None, 784))
	D_real = discriminator(x, drop_out)
	scope.reuse_variables()
	D_fake = discriminator(G_z, drop_out)


	
