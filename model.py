'''
Create and train a neural network from
https://github.com/commaai/research/blob/master/train_steering_model.py
'''
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.utils.visualize_util import plot
from sklearn.cross_validation import train_test_split
import time
import numpy as np
from PIL import Image
import pickle
import tensorflow as tf
tf.python.control_flow_ops = tf  # bug workaround (for dropout): http://stackoverflow.com/a/40066895


# Model/Training parameters
NUM_EPOCH = 10
BATCH_SIZE = 16
H, W, CH = 160, 320, 3
LR = 1e-4
L2_REG_SCALE = 0.


def gen(data, labels, batch_size):
	"""
	Batch generator

	Arguments:
		data: List of strings containing the path of image files
		labels: List of steering angles
		batch_size: Size of the batch to generate

	Yields:
		A tuple (X_batch, y_batch), where:
			X_batch: Batch of images, a tensor of shape (batch_size, H, W, CH)
			y_batch: Batch of steering angles
	"""
	start = 0
	end = start + batch_size
	n = data.shape[0]

	while True:
		# Read image data into memory as-needed
		image_files  = data[start:end]
		images = []
		for image_file in image_files:
			# Resize image, create numpy array representation
			image = Image.open(image_file).convert('RGB')
			image = image.resize((W, H), Image.ANTIALIAS)
			image = np.asarray(image, dtype='float32')
			images.append(image)
		images = np.array(images, dtype='float32')

		X_batch = images
		y_batch = labels[start:end]
		start += batch_size
		end += batch_size
		if start >= n:
			start = 0
			end = batch_size

		yield (X_batch, y_batch)


def get_model():
	"""
	Creates the comma.ai model, and returns a reference to the model
	The comma.ai model's original source code is available at:
	https://github.com/commaai/research/blob/master/train_steering_model.py
	"""
	ch, row, col = CH, H, W  # camera format

	model = Sequential()
	model.add(Lambda(lambda x: x/127.5 - 1.,
		input_shape=(row, col, ch),
		output_shape=(row, col, ch)))
	model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode='same', W_regularizer=l2(L2_REG_SCALE)))
	model.add(ELU())
	model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode='same', W_regularizer=l2(L2_REG_SCALE)))
	model.add(ELU())
	model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode='same', W_regularizer=l2(L2_REG_SCALE)))
	model.add(Flatten())
	model.add(Dropout(.2))
	model.add(ELU())
	model.add(Dense(512, W_regularizer=l2(0.)))
	model.add(Dropout(.5))
	model.add(ELU())
	model.add(Dense(1, W_regularizer=l2(0.)))

	model.compile(optimizer=Adam(lr=LR), loss='mean_squared_error')

	return model


def train_model():
	# Load driving data
	with open('data/driving_data_prep.p', mode='rb') as f:
		driving_data = pickle.load(f)

	data, labels = driving_data['images'], driving_data['labels']
	X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.1, random_state=0)

	# Get model
	model = get_model()
	model.summary()

	# Visualize model and save it to disk
	plot(model, to_file='img/model.png', show_shapes=True, show_layer_names=False)
	print('Saved model visualization at img/model.png')

	# Instantiate generators
	train_gen = gen(X_train, y_train, BATCH_SIZE)
	val_gen = gen(X_val, y_val, BATCH_SIZE)

	train_start_time = time.time()

	# Train model
	h = model.fit_generator(generator=train_gen, samples_per_epoch=X_train.shape[0], nb_epoch=NUM_EPOCH, validation_data=val_gen, nb_val_samples=X_val.shape[0])
	history = h.history

	total_time = time.time() - train_start_time
	print('Total training time: %.2f sec (%.2f min)' % (total_time, total_time/60))

	# Save model architecture to model.json, model weights to model.h5
	json_string = model.to_json()
	with open('model.json', 'w') as f:
		f.write(json_string)
	model.save_weights('model.h5')

	# Save training history
	with open('train_hist.p', 'wb') as f:
		pickle.dump(history, f)

	print('Model saved in model.json/h5, history saved in train_hist.p')


if __name__ == '__main__':
	print('Training comma.ai model')
	train_model()
	print('DONE: Training comma.ai model')
