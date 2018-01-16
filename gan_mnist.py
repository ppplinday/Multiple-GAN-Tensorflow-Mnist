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

fixed_z_ = np.random.normal(0, 1, (25, 100))
def show_result(num_epoch, save = False, path = 'result.png', isFix=False):
	z_ = np.random.normal(0, 1, (25, 100))

	if isFix:
		test_images = sess.run(G_z, {z: fixed_z_, drop_out: 0.0})
	else:
		test_images = sess.run(G_z, {z: z_, drop_out: 0.0})

	size_figure_grid = 5
	fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
	for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
		ax[i, j].get_xaxis().set_visible(False)
		ax[i, j].get_yaxis().set_visible(False)

	for k in range(5*5):
		i = k // 5
		j = k % 5
		ax[i, j].cla()
		ax[i, j].imshow(np.reshape(test_images[k], (28, 28)), cmap='gray')

	label = 'Epoch {0}'.format(num_epoch)
	fig.text(0.5, 0.04, label, ha='center')
	plt.savefig(path)

	plt.close()

batch_size = 100
lr = 0.0002
train_epoch = 100

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_set = (mnist.train.images - 0.5) / 0.5  # normalization; range: -1 ~ 1

with tf.variable_scope('G') as scope:
	z = tf.placeholder(tf.float32, shape=(None, 100))
	G_z = generator(z)

with tf.variable_scope('D') as scope:
	drop_out = tf.placeholder(dtype=tf.float32, name='drop_out')
	x = tf.placeholder(tf.float32, shape=(None, 784))
	D_real = discriminator(x, drop_out)
	scope.reuse_variables()
	D_fake = discriminator(G_z, drop_out)

eps = 1e-2
D_loss = tf.reduce_mean(-tf.log(D_real + eps) - tf.log(1 - D_fake + eps))
G_loss = tf.reduce_mean(- tf.log(D_fake + eps))

t_vars = tf.trainable_variables()
D_vars = [var for var in t_vars if 'D_' in var.name]
G_vars = [var for var in t_vars if 'G_' in var.name]

D_optim = tf.train.AdamOptimizer(lr).minimize(D_loss, var_list=D_vars)
G_optim = tf.train.AdamOptimizer(lr).minimize(G_loss, var_list=G_vars)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

if not os.path.isdir('MNIST_GAN_results'):
	os.mkdir('MNIST_GAN_results')
if not os.path.isdir('MNIST_GAN_results/Random_results'):
	os.mkdir('MNIST_GAN_results/Random_results')
if not os.path.isdir('MNIST_GAN_results/Fixed_results'):
	os.mkdir('MNIST_GAN_results/Fixed_results')

for epoch in range(train_epoch):
	for iter in range(train_set.shape[0] // batch_size):
		x_ = train_set[iter*batch_size:(iter+1)*batch_size]
		z_ = np.random.normal(0, 1, (batch_size, 100))
		loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, drop_out: 0.3})

		z_ = np.random.normal(0, 1, (batch_size, 100))
		loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, drop_out: 0.3})

	print('epoch: %d loss_d: %.3f, loss_g: %.3f' % (epoch + 1, D_losses, G_losses))
	p = 'MNIST_GAN_results/Random_results/MNIST_GAN_' + str(epoch + 1) + '.png'
	fixed_p = 'MNIST_GAN_results/Fixed_results/MNIST_GAN_' + str(epoch + 1) + '.png'
	show_result((epoch + 1), save=True, path=p, isFix=False)
	show_result((epoch + 1), save=True, path=fixed_p, isFix=True)

images = []
for e in range(train_epoch):
	img_name = 'MNIST_GAN_results/Fixed_results/MNIST_GAN_' + str(e + 1) + '.png'
	images.append(imageio.imread(img_name))
imageio.mimsave('MNIST_GAN_results/generation_animation.gif', images, fps=5)
	
sess.close()