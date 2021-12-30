"""
Implementation of Local interpretable model-agnostic explanations.
Source: Ribeiro, Marco Tulio, Sameer Singh, and Carlos Guestrin.
"Why should I trust you?: Explaining the predictions of any classifier."
Proceedings of the 22nd ACM SIGKDD international conference on knowledge
discovery and data mining. ACM (2016).
"""
import tensorflow as tf
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import skimage
import matplotlib.pyplot as plt

EXPLAINABLE_MODELS = {
	'linear_regression': LinearRegression,
	'decision_tree_regressor': DecisionTreeRegressor
}


class LIME:

	def __init__(self, image: tf.Tensor, model, random_seed=9):
		self.image = image
		self.model = model
		self.random_seed = random_seed
		self.super_pixels, self.super_pixel_count = self.create_super_pixels()
		self.perturbation_vectors = self.generate_pertubation_vectors()

	def create_super_pixels(self, kernel_size: int = 8, max_dist: int = 1000, ratio: float = 0.2):
		super_pixels = skimage.segmentation.quickshift(self.image, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio)
		super_pixel_count = len(np.unique(super_pixels))
		return super_pixels, super_pixel_count

	def plot_super_pixel_boundary(self):
		super_pixel_boundaries = skimage.segmentation.mark_boundaries(self.image.numpy().astype(int), self.super_pixels)
		plt.imshow(super_pixel_boundaries)
		plt.title('Superpixel boundaries')

	def generate_pertubation_vectors(self, num_perturbations: int = 100):
		"""Generates a number of perturbation vectors. These are binary vectors of length
		num_super_pixels, which define if a superpixel is perturbed or not"""
		if self.random_seed is not None:
			np.random.seed(self.random_seed)
		return np.random.binomial(1, 0.5, size=(num_perturbations, self.super_pixel_count))

	def predict_perturbed_images(self):
		perturbed_images = self.create_perturbed_images()
		return self.model(perturbed_images).numpy()

	def create_perturbed_images(self):
		self.generate_pertubation_vectors()
		return np.apply_along_axis(lambda x: self._create_perturbed_image(x), 1, self.perturbation_vectors)

	def _create_perturbed_image(self, perturbation_vector):
		perturbation_mask = np.isin(self.super_pixels, np.argwhere(perturbation_vector == 1))
		return np.where(np.expand_dims(perturbation_mask, -1), self.image, 0)

	def plot_perturbed_image(self):
		if self.perturbation_vectors is None:
			self.generate_pertubation_vectors()

		if self.random_seed is not None:
			np.random.seed(self.random_seed)
		idx = np.random.randint(len(self.perturbation_vectors))
		perturbed_img = self._create_perturbed_image(self.perturbation_vectors[idx])

		plt.imshow(perturbed_img.astype(int))
		plt.title('Perturbed Image')

	def calculate_perturbation_weights(self, kernel_width: float = 0.25):
		non_perturbed_vector = np.ones((1, self.super_pixel_count))
		distances = pairwise_distances(self.perturbation_vectors, non_perturbed_vector, metric='cosine')
		return np.squeeze(np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2)))

	def _fit_explainable_model(self, predictions, weights, explainable_model_type: str = 'decision_tree_regressor'):
		if explainable_model_type not in EXPLAINABLE_MODELS.keys():
			raise ValueError(f"Please specify one of the following model_types: {EXPLAINABLE_MODELS.keys()}")

		model = EXPLAINABLE_MODELS[explainable_model_type]()
		model.fit(X=self.perturbation_vectors, y=predictions, sample_weight=weights)
		if 'regression' in explainable_model_type:
			feature_importance = model.coef_
		elif 'tree' in explainable_model_type:
			feature_importance = model.feature_importances_
		else:
			raise ValueError(f"Please specify one of the following model_types: {EXPLAINABLE_MODELS.keys()}")

		return feature_importance

	def plot_explainable_image(self, class_to_explain: int = None, num_superpixels: int = 4, explainable_model_type: str = 'decision_tree_regressor'):
		"""

		:param class_to_explain: int of imagenet id
		:param num_superpixels:
		:param explainable_model_type:
		:return:
		"""
		perturbed_image_predictions = self.predict_perturbed_images()
		weights = self.calculate_perturbation_weights()

		if class_to_explain is None:
			class_to_explain = np.argmax(self.model(np.expand_dims(self.image, 0)).numpy())

		feature_importance = self._fit_explainable_model(predictions=perturbed_image_predictions[:, class_to_explain], weights=weights, explainable_model_type=explainable_model_type)
		superpixels_to_plot = np.argsort(feature_importance)[-num_superpixels:]
		superpixel_vector = np.zeros(self.super_pixel_count)
		np.put(superpixel_vector, superpixels_to_plot,  v=1)

		perturbed_img = self._create_perturbed_image(superpixel_vector)

		plt.imshow(perturbed_img.astype(int))
		plt.title(f'LIME explanation')



