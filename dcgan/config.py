learning_rate = 0.0002
beta1 = 0.5
epoch = 20
batch_size = 64
dataset = 'image'
checkpoint_dir = 'checkpoint'
sample_dir = 'samples'

#
import numpy as np
img = np.zeros((int(64 * 8), int(64 * 8), 3))
print(img.shape)

lowres_mask = np.zeros([64, 64, 8])
lowres_mask[0, 0, 0]=6
if lowres_mask.any():
	print('xxx')
else:
	print('yyy')