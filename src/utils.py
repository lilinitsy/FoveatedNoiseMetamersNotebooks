import cv2
import numpy as np
import matplotlib.pyplot as plt

from typing import List

def int_to_float(image: np.ndarray) -> np.ndarray:
	if image.dtype == np.uint8:
		return image.astype(np.float32) / 255.0
	elif image.dtype == np.uint16:
		return image.astype(np.float32) / 65535.0
	
	# float: clip to [0, 1]
	return np.clip(image.astype(np.float32), 0.0, 1.0)

def load_image(path: str, grayscale = False, as_float = False) -> np.ndarray:
	flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
	img = cv2.imread(str(path), flag)

	if img is None:
		raise FileNotFoundError(f"Can't read: {path}")

	if as_float:
		img = img.astype(np.float32) / 255.0
		
	return img


def display_image(image: np.ndarray, title: str = "Image") -> None:
	if image.ndim == 3 and image.shape[2] == 3:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	plt.figure()
	if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
		plt.imshow(image, cmap="gray")
	else:
		plt.imshow(image)
	plt.title(title)
	plt.axis("off")
	plt.show()



# Chose pretty random params for now. Pick them based on screen later.
def pixels_per_degree(dpi: float, viewing_distance_m: float) -> float:
	deg_rad = np.deg2rad(1.0)
	size_m = 2.0 * viewing_distance_m * np.tan(deg_rad / 2.0)
	px_per_m = dpi / 25.4 * 1000.0 / 1000.0 * 25.4 * 39.37007874
	px_per_m = dpi / 0.0254
	return px_per_m * size_m

	