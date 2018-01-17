import matplotlib
matplotlib.use('Agg')

import os, time, itertools, imageio, pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def generator(x, isTrain=True, reuse=False):
	with tf.variable_scope('generator', reuse=reuse):
		conv1 = tf.layers.conv2d_transpose(x, 1024, [4, 4], strides=(1, 1), padding='valid')
		lrelu1 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv1, training=isTrain), alpha=0.2)

		conv2 = tf.layers.conv2d_transpose(lrelu1, 512, [4, 4], strides=(2, 2), padding='same')
		lrelu2 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv2, training=isTrain), alpha=0.2)

		conv3 = tf.layers.conv2d_transpose(lrelu2, 256, [4, 4], strides=(2, 2), padding='same')
		lrelu3 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv3, training=isTrain), alpha=0.2)

		conv4 = tf.layers.conv2d_transpose(lrelu3, 128, [4, 4], strides=(2, 2), padding='same')
		lrelu4 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv4, training=isTrain), alpha=0.2)

		conv5 = tf.layers.conv2d_transpose(lrelu4, 1, [4, 4], strides=(2, 2), padding='same')
		out = tf.nn.tanh(conv5)

		return out

def discriminator(x, isTrain=True, reuse=False):
	with tf.variable_scope('discriminator', reuse=reuse):
		conv1 = tf.layers.conv2d(x, 128, [4, 4], strides=(2, 2), padding='same')
		lrelu1 = tf.nn.leaky_relu(conv1, alpha=0.2)

		conv2 = tf.layers.conv2d(lrelu1, 256, [4, 4], strides=(2, 2), padding='same')
		lrelu2 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv2, training=isTrain), alpha=0.2)

		conv3 = tf.layers.conv2d(lrelu2, 512, [4, 4], strides=(2, 2), padding='same')
		lrelu3 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv3, training=isTrain), alpha=0.2)

		conv4 = tf.layers.conv2d(lrelu3, 1024, [4, 4], strides=(2, 2), padding='same')
		lrelu4 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv4, training=isTrain), alpha=0.2)

		conv5 = tf.layers.conv2d(lrelu4, 1, [4, 4], strides=(1, 1), padding='valid')
		out = tf.nn.sigmoid(conv5)

		return out, conv5

fixed_z_ = np.random.normal(0, 1, (25, 100))
def show_result(num_epoch, show = False, save = False, path = 'result.png'):
	z_ = np.random.normal(0, 1, (25, 100))
	fixed_z_ = fixed_z_.reshape(25, 1, 1, 100)
	test_images = sess.run(G_z, {z: fixed_z_, isTrain: False})

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
train_epoch = 20

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])

x = tf.placeholder(tf.float32, shape=(None, 64, 64, 1))
z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100))
isTrain = tf.placeholder(dtype=tf.bool)

G_z = generator(z, isTrain)
D_real, D_real_logits = discriminator(x, isTrain)
D_fake, D_fake_logits = discriminator(G_z, isTrain, reuse=True)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1, 1, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 1, 1, 1])))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 1, 1, 1])))

T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
	D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
	G_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(G_loss, var_list=G_vars)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

train_set = tf.image.resize_images(mnist.train.images, [64, 64])
train_set = (train_set - 0.5) / 0.5

root = 'MNIST_DCGAN_results/'
model = 'MNIST_DCGAN_'
if not os.path.isdir(root):
	os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
	os.mkdir(root + 'Fixed_results')

np.random.seed(int(time.time()))
print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
	G_losses = []
	D_losses = []

	epoch_start_time = time.time()
	for iter in range(mnist.train.num_examples // batch_size):
		x_ = train_set[iter*batch_size:(iter+1)*batch_size]
		z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))
		x_ = np.array(sess.run([x_])).reshape(batch_size, 64, 64, 1)

		loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, isTrain: True})
		D_losses.append(loss_d_)

		z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))
		loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, x: x_, isTrain: True})
		G_losses.append(loss_g_)

	epoch_end_time = time.time()
	per_epoch_ptime = epoch_end_time - epoch_start_time
	print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
	fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
	show_result((epoch + 1), save=True, path=fixed_p)

end_time = time.time()
total_ptime = end_time - start_time
print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
print("Training finish!... save training results")

images = []
for e in range(train_epoch):
	img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
	images.append(imageio.imread(img_name))
imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)

sess.close()