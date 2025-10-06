import cv2
import numpy as np

'''
Blur + downsample by 2 using OpenCV's pyrDown
'''
def pyramid_downscale(image: np.ndarray) -> np.ndarray:
	(rows, columns, _) = map(int, image.shape)
	cv2.pyrDown(image, dstsize = (columns // 2, rows // 2))

'''
Upsample by 2 with OpenCV pyrUp.
Shape should be passed as (height, width)
'''
def pyramid_upscale(image: np.ndarray, out_shape: tuple) -> np.ndarray:
	(h, w) = out_shape[:2]
	upscaled_image = cv2.pyrUp(image, dstsize = (w, h))
	upscaled_image = upscaled_image[:h, :w, ...] # Safely crop
	return upscaled_image


def make_gaussian_pyramid