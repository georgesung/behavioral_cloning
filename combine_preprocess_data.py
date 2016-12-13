'''
Combine preprocessed data for normal and recovery driving
'''
import numpy as np
import pickle

print('Combining preprocessed data for normal and recovery driving')

with open('data/driving_data_prep_normal.p', 'rb') as f:
	normal = pickle.load(f)

with open('data/driving_data_prep_left.p', 'rb') as f:
	left = pickle.load(f)

with open('data/driving_data_prep_right.p', 'rb') as f:
	right = pickle.load(f)

images = np.concatenate((normal['images'], left['images'], right['images']))
labels = np.concatenate((normal['labels'], left['labels'], right['labels']))

driving_data = {'images': images, 'labels': labels}
with open('data/driving_data_prep.p', mode='wb') as f:
	pickle.dump(driving_data, f)

print('DONE: Combining preprocessed data for normal and recovery driving')
