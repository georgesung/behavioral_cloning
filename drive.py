import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import tensorflow as tf
tf.python.control_flow_ops = tf  # bug workaround (for dropout): http://stackoverflow.com/a/40066895

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

# Inupt image dimensions
H, W, CH = 160, 320, 3

@sio.on('telemetry')
def telemetry(sid, data):
	# The current steering angle of the car
	steering_angle = data["steering_angle"]
	# The current throttle of the car
	throttle = data["throttle"]
	# The current speed of the car
	speed = data["speed"]
	# The current image from the center camera of the car
	imgString = data["image"]
	image = Image.open(BytesIO(base64.b64decode(imgString)))

	# Resize image, create numpy array representation
	image = image.convert('RGB')
	image = image.resize((W, H), Image.ANTIALIAS)
	image = np.asarray(image, dtype='float32')
	image = image[None, :, :, :]
	transformed_image_array = image

	# Use the model to predict steering angle
	steering_angle = float(model.predict(transformed_image_array, batch_size=1))
	
	# Throttle is a peicewise linear function of steering angle
	#throttle = max(0.1, -0.15/0.05 * abs(steering_angle) + 0.35)
	#throttle = 0.35 if abs(steering_angle) < 0.04 else 0.15 if abs(steering_angle) < 0.4 else -0.5
	if abs(steering_angle) < 0.03:
		throttle = 0.25
	elif abs(steering_angle) < 0.4:
		throttle = 0.15
	else:
		throttle = -0.5

	print('steering angle: %.4f, throttle: %.4f' % (steering_angle, throttle))
	send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
	print("connect ", sid)
	send_control(0, 0)


def send_control(steering_angle, throttle):
	sio.emit("steer", data={
	'steering_angle': steering_angle.__str__(),
	'throttle': throttle.__str__()
	}, skip_sid=True)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Remote Driving')
	parser.add_argument('model', type=str,
	help='Path to model definition json. Model weights should be on the same path.')
	args = parser.parse_args()
	with open(args.model, 'r') as jfile:
		#model = model_from_json(json.load(jfile))
		model = model_from_json(jfile.readline())

	model.compile("adam", "mse")
	weights_file = args.model.replace('json', 'h5')
	model.load_weights(weights_file)

	# wrap Flask application with engineio's middleware
	app = socketio.Middleware(sio, app)

	# deploy as an eventlet WSGI server
	eventlet.wsgi.server(eventlet.listen(('', 4567)), app)