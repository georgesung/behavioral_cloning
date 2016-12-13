'''
Preprocess raw *recovery* data

Assume the data is stored in data/driving_data.p
Writes to data/driving_data_prep.p, which contains a dict:

{'data': image_file_names, 'labels': adjusted_steering_angles}
'''
import pickle
import numpy as np


# Script parameters
ANGLE_OFFSET = 0.5  # angle offset for left/right cameras
OFFSET_THRESHOLD = 0.  # angle threshold to exceed before applying offset
MA_WINDOW = 1  # moving average window


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


print('Preprocessing recovery data')

for case in ('left', 'right'):
	# Get raw driving data
	with open('data/driving_data_%s.p' % case, mode='rb') as f:
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

	new_images = []
	new_labels = []
	for i, label in enumerate(labels):
		# For left/right recovery, only care about left/right camera images
		# Ignore all steering angles that are negative/positive
		if case=='left' and label[1]==1. and label[0] >= 0:
			new_label = min(1., label[0] + ANGLE_OFFSET)
		elif case=='right' and label[1]==2. and label[0] <= 0:
			new_label = max(-1., label[0] - ANGLE_OFFSET)
		else:
			continue

		new_images.append(images[i])
		new_labels.append(new_label)

	images = np.array(new_images)
	labels = np.array(new_labels)

	driving_data['images'] = images
	driving_data['labels'] = labels

	with open('data/driving_data_prep_%s.p' % case, mode='wb') as f:
		pickle.dump(driving_data, f)

print('DONE: Preprocessing recovery data')