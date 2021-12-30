"""Module to get models and preprocessing functions for pretrained models"""
import tensorflow as tf
import numpy as np
from typing import Tuple

EFFICIENTNET_VERSION = {
	'B0': {'model': tf.keras.applications.EfficientNetB0, 'img_size': 224},
	'B1': {'model': tf.keras.applications.EfficientNetB1, 'img_size': 240},
	'B2': {'model': tf.keras.applications.EfficientNetB2, 'img_size': 260},
	'B3': {'model': tf.keras.applications.EfficientNetB3, 'img_size': 300},
	'B4': {'model': tf.keras.applications.EfficientNetB4, 'img_size': 380},
	'B5': {'model': tf.keras.applications.EfficientNetB5, 'img_size': 456},
	'B6': {'model': tf.keras.applications.EfficientNetB6, 'img_size': 528},
	'B7': {'model': tf.keras.applications.EfficientNetB7, 'img_size': 600},
}


def get_efficientnet(version: str = 'B0', weights: str = 'imagenet') -> Tuple[tf.keras.Model, callable]:
	"""
	Function to get one of the EfficientNet models with pretrained weights. For more information
	on image resolution and number of parameters for each of the EfficientNet models see:
	https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
	"""
	img_size = EFFICIENTNET_VERSION[version]['img_size']
	def preprocess_image(x):
		img = np.expand_dims(x, 0)
		return tf.image.resize(x, (img_size, img_size))

	return (EFFICIENTNET_VERSION[version]['model'](weights=weights), preprocess_image)
