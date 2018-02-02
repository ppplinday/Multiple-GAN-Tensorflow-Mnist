import os
import time
import math
import random
import config
import itertools
import scipy.misc
import PIL
from PIL import Image
import numpy as np
from glob import glob
import tensorflow as tf
from six.moves import xrange
from tensorflow.examples.tutorials.mnist import input_data

class Generator:
	def __init__(self, depths=[512, 256, 128, 64], s_size=4):
		self.depths = depths + [1]
		self.s_size = s_size
		self.reuse = False

	def __call__(self, inputs, training=False):
		inputs = tf.convert_to_tensor(inputs)
		inputs = tf.cast(inputs, tf.float32)
		with tf.variable_scope('generator', reuse=self.reuse):
			
			with tf.variable_scope('reshape'):
				outputs = tf.layers.dense(inputs, self.depths[0] * self.s_size * self.s_size, 
					kernel_initializer=tf.random_normal_initializer(stddev=0.02))
				outputs = tf.reshape(outputs, [-1, self.s_size, self.s_size, self.depths[0]])
				outputs = tf.nn.relu(tf.contrib.layers.batch_norm(outputs, 
					decay=0.9, updates_collections=None, epsilon=1e-5, center=True, scale=True, is_training=training), name='outputs')
				
			with tf.variable_scope('deconv1'):
				outputs = tf.layers.conv2d_transpose(outputs, self.depths[1], [5, 5], strides=(2, 2), padding='SAME',
					kernel_initializer=tf.random_normal_initializer(stddev=0.02))
				outputs = tf.nn.relu(tf.contrib.layers.batch_norm(outputs, 
					decay=0.9, updates_collections=None, epsilon=1e-5, center=True, scale=True, is_training=training), name='outputs')
				
			with tf.variable_scope('deconv2'):
				outputs = tf.layers.conv2d_transpose(outputs, self.depths[2], [5, 5], strides=(2, 2), padding='SAME',
					kernel_initializer=tf.random_normal_initializer(stddev=0.02))
				outputs = tf.nn.relu(tf.contrib.layers.batch_norm(outputs, 
					decay=0.9, updates_collections=None, epsilon=1e-5, center=True, scale=True, is_training=training), name='outputs')
				
			with tf.variable_scope('deconv3'):
				outputs = tf.layers.conv2d_transpose(outputs, self.depths[3], [5, 5], strides=(2, 2), padding='SAME',
					kernel_initializer=tf.random_normal_initializer(stddev=0.02))
				outputs = tf.nn.relu(tf.contrib.layers.batch_norm(outputs, 
					decay=0.9, updates_collections=None, epsilon=1e-5, center=True, scale=True, is_training=training), name='outputs')
				
			with tf.variable_scope('deconv4'):
				outputs = tf.layers.conv2d_transpose(outputs, self.depths[4], [5, 5], strides=(2, 2), padding='SAME',
					kernel_initializer=tf.random_normal_initializer(stddev=0.02))
				
			with tf.variable_scope('tanh'):
				outputs = tf.tanh(outputs, name='outputs')
			self.reuse = True
			self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')
			return outputs

class Discriminator:
	def __init__(self, depths=[64, 128, 256, 512]):
		self.depths = [1] + depths
		self.reuse = False

	def __call__(self, inputs, training=False):
		outputs = tf.convert_to_tensor(inputs)
		outputs = tf.cast(outputs, tf.float32)
		with tf.variable_scope('discriminator', reuse=self.reuse):
			
			with tf.variable_scope('conv1'):
				outputs = tf.layers.conv2d(outputs, self.depths[1], [5, 5], strides=(2, 2), padding='SAME',
					kernel_initializer=tf.random_normal_initializer(stddev=0.02))
				outputs = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(outputs, 
					decay=0.9, updates_collections=None, epsilon=1e-5, center=True, scale=True, is_training=training), name='outputs')
				
			with tf.variable_scope('conv2'):
				outputs = tf.layers.conv2d(outputs, self.depths[2], [5, 5], strides=(2, 2), padding='SAME',
					kernel_initializer=tf.random_normal_initializer(stddev=0.02))
				outputs = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(outputs, 
					decay=0.9, updates_collections=None, epsilon=1e-5, center=True, scale=True, is_training=training), name='outputs')
				
			with tf.variable_scope('conv3'):
				outputs = tf.layers.conv2d(outputs, self.depths[3], [5, 5], strides=(2, 2), padding='SAME',
					kernel_initializer=tf.random_normal_initializer(stddev=0.02))
				outputs = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(outputs, 
					decay=0.9, updates_collections=None, epsilon=1e-5, center=True, scale=True, is_training=training), name='outputs')
				
			with tf.variable_scope('conv4'):
				outputs = tf.layers.conv2d(outputs, self.depths[4], [5, 5], strides=(2, 2), padding='SAME',
					kernel_initializer=tf.random_normal_initializer(stddev=0.02))
				outputs = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(outputs, 
					decay=0.9, updates_collections=None, epsilon=1e-5, center=True, scale=True, is_training=training), name='outputs')
				
			with tf.variable_scope('classify'):
				batch_size = outputs.get_shape()[0].value
				reshape = tf.reshape(outputs, [-1, 8192])
				outputs = tf.layers.dense(reshape, 1, name='outputs')
				
		self.reuse = True
		self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d')
		return tf.nn.sigmoid(outputs), outputs

# merge picture
def merge(images, size):
	h, w = images.shape[1], images.shape[2]
	img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
	for id, image in enumerate(images):
		i = id // size[0]
		j = id % size[1]
		i = int(i)
		j = int(j)
		img[i * h: (i + 1) * h, j * w: (j + 1) * w, :] = image
	return img

# save merge picture
def save_images(images, size, image_path):
	images = images
	img = merge(images, size)
	return scipy.misc.imsave(image_path, (255*img).astype(np.uint8))

def create_file(path):
	if not os.path.exists(path):
		os.makedirs(path)

class DCGAN:
	def __init__(self, sess, dataset):
		self.sess = sess;
		self.batch_size = 100
		self.z_dim = 100
		self.g = Generator()
		self.d = Discriminator()
		#self.z = tf.random_uniform([self.batch_size, self.z_dim], minval=-1.0, maxval=1.0)

		self.sample_size = 100
		self.model_name = "DCGAN.models"
		self.data_name = dataset
		self.name = self.model_name + "_" + self.data_name + "_"

		self.checkpoint_dir = self.name + "checkpoint"
		self.sample_dir = "./" + self.name + "samples/"
		self.logs_dir = "./" + self.name + "logs"
		create_file(self.checkpoint_dir)
		create_file(self.sample_dir)
		create_file(self.logs_dir)

		self.build()

	def build(self):
		self.images = tf.placeholder(tf.float32, [None, 64, 64, 1], name='real_images')
		self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
		self.z_sum = tf.summary.histogram("z", self.z)

		self.G = self.g(self.z)
		self.D, self.D_logits = self.d(self.images)
		self.D_, self.D_logits_ = self.d(self.G)

		self.d_sum = tf.summary.histogram("d", self.D)
		self.d__sum = tf.summary.histogram("d_", self.D_)
		#self.G_sum = tf.summary.image("G", tf.reshape(self.G, [None, 28, 28]))

		self.d_loss_real = tf.reduce_mean(
			 tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits,
			 										 labels=tf.ones_like(self.D)))
		self.d_loss_fake = tf.reduce_mean(
			 tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
			 										 labels=tf.zeros_like(self.D_)))
		self.g_loss = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
			 										labels=tf.ones_like(self.D_)))
		self.d_loss = self.d_loss_real + self.d_loss_fake

		self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
		self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
		self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
		self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

		t_vars = tf.trainable_variables()
		self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
		self.g_vars = [var for var in t_vars if 'generator' in var.name]
		
		self.saver = tf.train.Saver(max_to_keep=1)

	def train(self):
		mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

		d_optim = tf.train.AdamOptimizer().minimize(self.d_loss, var_list=self.d_vars)
		g_optim = tf.train.AdamOptimizer().minimize(self.g_loss, var_list=self.g_vars)

		try:
			tf.global_variables_initializer().run()
		except:
			tf.initialize_all_variables().run()

		self.g_sum = tf.summary.merge([self.z_sum, self.d__sum,  self.d_loss_fake_sum, self.g_loss_sum])
		self.d_sum = tf.summary.merge([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
		self.writer = tf.summary.FileWriter(self.logs_dir, self.sess.graph)

		# for sample picture
		sample_z = np.random.uniform(-1, 1, size=(self.sample_size, self.z_dim))
		sample_images, _ = mnist.test.next_batch(100)
		sample_images = sample_images.reshape(-1, 28, 28, 1)
		sample_images = tf.image.resize_images(sample_images, [64, 64]).eval()
		sample_images = sample_images.reshape(-1, 64, 64, 1)
		#sample_images = (sample_images - 0.5) * 2.0
		start_time = time.time()

		# load check point
		if self.load(self.checkpoint_dir):
			print('load the checkpoint!')
		else:
			print('cannot load the checkpoint and init all the varibale')

		for id in range(12000):

			# mini_batch
			batch_images, batch_labels =  mnist.train.next_batch(self.batch_size)
			batch_images = batch_images.reshape(-1, 28, 28, 1)
			batch_images = tf.image.resize_images(batch_images, [64, 64]).eval()
			
			batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
			# before = batch_images.reshape(-1, 28, 28, 1)
			# print(before[0])
			# save_images(before, [10, 10], self.sample_dir + 'before.png')

			# update D and G
			#if id % 100 == 0:
			_, summary_str, err_d = self.sess.run([d_optim, self.d_sum, self.d_loss], 
				feed_dict={self.images: batch_images, self.z: batch_z})
			self.writer.add_summary(summary_str, id)

			_, summary_str, err_g = self.sess.run([g_optim, self.g_sum, self.g_loss],
				feed_dict={self.z: batch_z})
			self.writer.add_summary(summary_str, id)

			# err_d_fake = self.d_loss_fake.eval({self.z: batch_z})
			# err_d_real = self.d_loss_real.eval({self.images: batch_images})
			# err_g = self.g_loss.eval({self.z: batch_z})

			
			print("Epoch: [{:4d}/{:4d}] time: {:4.4f}, d_loss: {:.8f}, g_loss: {:.8f}".format(
					id, 12000, time.time() - start_time, err_d, err_g))

			if id % 1 == 0:
				samples, d_loss, g_loss = self.sess.run(
					[self.G, self.d_loss, self.g_loss],
					feed_dict={self.z: sample_z, self.images: sample_images}
				)
				samples = samples.reshape(-1, 64, 64, 1)
				samples = tf.image.resize_images(samples, [28, 28]).eval()
				samples = samples.reshape(-1, 28, 28, 1)
				# print('shape of samples = ')
				# print(samples[0])
				save_images(samples, [10, 10], self.sample_dir + 'sample_{:07d}.png'.format(id))
				print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

			if id % 100 == 0:
					self.save(self.checkpoint_dir)
					print("Save the checkpoint for the count: {:4d}".format(id))


	def save(self, checkpoint_dir):
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name))

	def load(self, checkpoint_dir):
		print('Being to load the checkpoint')

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			self.saver.restore(self.sess, ckpt.model_checkpoint_path)
			return True
		else:
			return False



