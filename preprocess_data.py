'''
Preprocess raw data

Assume the data is stored in data/driving_data.p
Writes to data/driving_data_prep.p, which contains a dict:

{'data': image_file_names, 'labels': adjusted_steering_angles}
'''
import pickle
import numpy as np


# Script parameters
ANGLE_OFFSET = 0.15  # angle offset for left/right cameras
OFFSET_THRESHOLD = 0.  # angle threshold to exceed before applying offset
MA_WINDOW = 3  # moving average window
BALANCE_DATA = False  # change data distribution to reduce frequency of 0-degree steering?
ZERO_PCT = 0.5  # fraction of data for 0-degree steering angle


def moving_average(a, n=3):
	'''
	Calculate moving average
	http://stackoverflow.com/a/14314054
	'''
	if n==1:
		return a
	ret = np.cumsum(a, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n


def balance_data(data_in, labels_in, zero_pct):
	"""
	Balance data such that zero_pct of data has steering angle 0
	"""
	total_data_size = data_in.shape[0]
	zero_idx = []  # all list indices where labels[idx]==0
	for i, label in enumerate(labels_in):
		if label[0] == 0.0:
			zero_idx.append(i)

	nonzero_data_size = total_data_size - len(zero_idx)
	zero_data_size = int(zero_pct * nonzero_data_size / (1 - zero_pct))

	# Randomly remove 0-data, such that zero_pct of new dataset is 0-data
	remove_idx = np.random.choice(zero_idx, total_data_size - zero_data_size - nonzero_data_size, replace=False)
	data = np.delete(data_in, remove_idx)
	labels = np.delete(labels_in, remove_idx, axis=0)

	return data, labels


print('Preprocessing data')
# Get raw driving data
with open('data/driving_data_normal.p', mode='rb') as f:
	driving_data = pickle.load(f)

images = driving_data['images']
labels = driving_data['labels']

# Calculate moving average of steering angle
center_angles = labels[::3][:,0]  # 1D tensor of all steering angles from center camera
left_angles   = labels[1::3][:,0]
right_angles  = labels[2::3][:,0]

center_angles_ma = moving_average(center_angles, n=MA_WINDOW)
left_angles_ma   = moving_average(left_angles, n=MA_WINDOW)
right_angles_ma  = moving_average(right_angles, n=MA_WINDOW)

# Throw away first few data points due to moving avg
images = images[3 * (MA_WINDOW - 1):]
labels = labels[3 * (MA_WINDOW - 1):]

# Replace original labels with moving average steering angles
labels[::3][:,0]  = center_angles_ma
labels[1::3][:,0] = left_angles_ma
labels[2::3][:,0] = right_angles_ma

# Balance the data to remove bias of driving straight
if BALANCE_DATA:
	images, labels = balance_data(images, labels, ZERO_PCT)

# For all left/right camera images
# If steering_angle >+/- ANGLE_OFFSET to steering angle
new_images = []
new_labels = []
for i, label in enumerate(labels):
	if label[1] == 0.:  # center
		new_label = label[0]
	elif abs(label[0]) >= OFFSET_THRESHOLD:  # only offset left/right camera angle if steering angle >= threshold
		if label[1] == 1.:  # left
			new_label = min(1., label[0] + ANGLE_OFFSET)
		else:  # right
			new_label = max(-1., label[0] - ANGLE_OFFSET)
	else:  # if not center camera, and steering angle < threshold, skip these data points
		continue

	new_images.append(images[i])
	new_labels.append(new_label)

images = np.array(new_images)
labels = np.array(new_labels)

driving_data['images'] = images
driving_data['labels'] = labels

with open('data/driving_data_prep_normal.p', mode='wb') as f:
	pickle.dump(driving_data, f)

print('DONE: Preprocessing data')