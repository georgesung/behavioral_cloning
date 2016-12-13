'''
Parse driving_log.csv to create a dict mapping:
image_file_name -> (steering_angle, 0/1/2)

0/1/2 deontes center/left/right camera image

Save this dict as pickle file
'''
import pickle
import numpy as np

cases = ['normal', 'left', 'right']

for case in cases:
	with open('driving_log_%s.csv' % case, mode='r') as f:
		lines = f.readlines()

	images = []
	labels = []

	for line in lines:
		fields = line.split(',')

		# Center image
		images.append(fields[0])
		labels.append((float(fields[3]), 0.))

		# Left image
		images.append(fields[1][1:])  # strip leading space
		labels.append((float(fields[3]), 1.))

		# Right image
		images.append(fields[2][1:])  # strip leading space
		labels.append((float(fields[3]), 2.))

	data = {'images': np.array(images), 'labels': np.array(labels)}

	# Save to pickle file
	with open('driving_data_%s.p' % case, mode='wb') as f:
		pickle.dump(data, f)
