import cv2
import numpy as np
from src.utils import int_to_float


'''
Blur + downsample by 2 using OpenCV's pyrDown
'''
def pyramid_downscale(image: np.ndarray) -> np.ndarray:
	(rows, columns) = map(int, image.shape[:2])
	downscaled_image = cv2.pyrDown(image, dstsize = (columns // 2, rows // 2))

	return downscaled_image

'''
Upsample by 2 with OpenCV pyrUp.
Shape should be passed as (height, width)
'''
def pyramid_upscale(image: np.ndarray, out_shape: tuple) -> np.ndarray:
	(h, w) = out_shape[:2]
	upscaled_image = cv2.pyrUp(image, dstsize = (w, h))
	upscaled_image = upscaled_image[:h, :w, ...] # Safely crop
	return upscaled_image


def make_gaussian_pyramid(image: np.ndarray, level: int) -> list[np.ndarray]:
	if level < 1:
		raise ValueError("Level must be >= 1 (lvl 0 is the base mip)")
	
	pyramid = [None] * level
	pyramid[0] = int_to_float(image)

	for i in range(1, level):
		pyramid[i] = pyramid_downscale(pyramid[i - 1])
	
	return pyramid


def make_laplacian_pyramid(image: np.ndarray, level: int, gaussian_pyramid = None) -> list[np.ndarray]:
	if gaussian_pyramid == None:
		gaussian_pyramid = make_gaussian_pyramid(image, level)
	laplacian_pyramid = [None] * level

	# The final level is a gaussian pyramid
	for i in range(0, level - 1):
		upscaled = pyramid_upscale(gaussian_pyramid[i + 1], gaussian_pyramid[i].shape)
		laplacian_pyramid[i] = gaussian_pyramid[i] - upscaled

	laplacian_pyramid[-1] = gaussian_pyramid[-1].copy()

	return laplacian_pyramid