import os
import time
import math
import random
import config
import itertools
import scipy.misc
import PIL
import numpy as np
from glob import glob
import tensorflow as tf
from six.moves import xrange
from tensorflow.examples.tutorials.mnist import input_data

class Generator:
	def __init__(self, depths=[128, 784]):
		self.depths = depths
		self.reuse = False

	def __call__(self, inputs, labels):
		inputs = tf.convert_to_tensor(inputs)
		inputs = tf.cast(inputs, tf.float32)
		labels = tf.convert_to_tensor(labels)
		labels = tf.cast(labels, tf.float32)
		with tf.variable_scope('generator', reuse=self.reuse):

			outputs = tf.concat([inputs, labels], axis=1)

			with tf.variable_scope('fc1'):
				outputs = tf.layers.dense(outputs, self.depths[0], kernel_initializer=tf.random_normal_initializer(stddev=0.02), name='fc')
				outputs = tf.nn.relu(outputs, name='outputs')

			with tf.variable_scope('fc2'):
				outputs = tf.layers.dense(outputs, self.depths[1], kernel_initializer=tf.random_normal_initializer(stddev=0.02), name='fc')
				outputs = tf.nn.sigmoid(outputs, name='outputs') 	

		self.reuse = True
		return outputs

class Discriminator:
	def __init__(self, depths=[128, 1]):
		self.depths = depths
		self.reuse = False

	def __call__(self, inputs, labels):
		inputs = tf.convert_to_tensor(inputs)
		inputs = tf.cast(inputs, tf.float32)
		labels = tf.convert_to_tensor(labels)
		labels = tf.cast(labels, tf.float32)
		with tf.variable_scope('discriminator', reuse=self.reuse):

			outputs = tf.concat([inputs, labels], axis=1)

			with tf.variable_scope('fc1'):
				outputs = tf.layers.dense(outputs, self.depths[0], kernel_initializer=tf.random_normal_initializer(stddev=0.02), name='fc')
				outputs = tf.nn.relu(outputs, name='outputs')

			with tf.variable_scope('fc2'):
				outputs = tf.layers.dense(outputs, self.depths[1], kernel_initializer=tf.random_normal_initializer(stddev=0.02), name='fc')
				res = tf.nn.sigmoid(outputs, name='outputs') 	

			self.reuse = True
			return res, outputs

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
	images = images / 2. + 0.5
	img = merge(images, size)
	return scipy.misc.imsave(image_path, (255*img).astype(np.uint8))

def create_file(path):
	if not os.path.exists(path):
		os.makedirs(path)

class CGAN:
	def __init__(self, sess, dataset):
		self.sess = sess;
		self.batch_size = 100
		self.z_dim = 100
		self.class_dim = 10
		self.g = Generator()
		self.d = Discriminator()
		#self.z = tf.random_uniform([self.batch_size, self.z_dim], minval=-1.0, maxval=1.0)

		self.sample_size = 100
		self.model_name = "CGAN.models"
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
		self.images = tf.placeholder(tf.float32, [None, 784], name='real_images')
		self.labels = tf.placeholder(tf.float32, [None, 10], name='labels')
		self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
		self.z_sum = tf.summary.histogram("z", self.z)

		self.G = self.g(self.z, self.labels)
		self.D, self.D_logits = self.d(self.images, self.labels)
		self.D_, self.D_logits_ = self.d(self.G, self.labels)

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
		sample_labels = np.zeros([self.sample_size, self.class_dim])
		for i in range(10):
			sample_labels[i * 10: (i + 1) * 10, i] = 1
		sample_images, _ = mnist.test.next_batch(100)
		sample_images = (sample_images - 0.5) * 2.0
		start_time = time.time()

		# load check point
		if self.load(self.checkpoint_dir):
			print('load the checkpoint!')
		else:
			print('cannot load the checkpoint and init all the varibale')

		for id in range(1000001):

			# mini_batch
			batch_images, batch_labels =  mnist.train.next_batch(self.batch_size)
			batch_images = (batch_images - 0.5) * 2.0
			batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
			before = batch_images.reshape(-1, 28, 28, 1)
			save_images(before, [10, 10], self.sample_dir + 'before.png')

			# update D and G
			if id % 100 == 0:
				_, summary_str = self.sess.run([d_optim, self.d_sum], 
					feed_dict={self.images: batch_images, self.labels: batch_labels, self.z: batch_z})
				self.writer.add_summary(summary_str, id)

			_, summary_str = self.sess.run([g_optim, self.g_sum],
				feed_dict={self.labels: batch_labels, self.z: batch_z})
			self.writer.add_summary(summary_str, id)

			err_d_fake = self.d_loss_fake.eval({self.z: batch_z, self.labels: batch_labels})
			err_d_real = self.d_loss_real.eval({self.images: batch_images, self.labels: batch_labels})
			err_g = self.g_loss.eval({self.z: batch_z, self.labels: batch_labels})

			if id % 100 == 0:
				print("Epoch: [{:4d}/{:4d}] time: {:4.4f}, d_loss: {:.8f}, g_loss: {:.8f}".format(
						id, 1000000, time.time() - start_time, err_d_fake + err_d_real, err_g))

			if id % 10000 == 0:
				samples, d_loss, g_loss = self.sess.run(
					[self.G, self.d_loss, self.g_loss],
					feed_dict={self.z: sample_z, self.images: sample_images, self.labels: sample_labels}
				)
				samples = samples.reshape(-1, 28, 28, 1)
				print('shape of samples = ')
				print(samples.shape)
				print(samples[0])
				save_images(samples, [10, 10], self.sample_dir + 'sample_{:07d}.png'.format(id))
				print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

			if id % 50000 == 0:
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



