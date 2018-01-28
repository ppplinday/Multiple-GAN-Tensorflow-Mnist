import os
import sys
import scipy.misc
import numpy as np
from cgan import CGAN
import tensorflow as tf

if __name__ == '__main__':
	model_name = sys.argv[1]
	dataset = sys.argv[2]
	with tf.Session() as sess:
		if model_name == 'cgan':
			model = CGAN(sess, dataset)
		else:
			print("We cannot find this model")

		model.train()

		print("finish to train dcgan")
