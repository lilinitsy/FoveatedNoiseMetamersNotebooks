import cv2
import numpy as np
from typing import List

def int_to_float(image: np.ndarray) -> np.ndarray:
	if image.dtype == np.uint8:
		return image.astype(np.float32) / 255.0
	elif image.dtype == np.uint16:
		return image.astype(np.float32) / 65535.0
	
	# float: clip to [0, 1]
	return np.clip(image.astype(np.float32), 0.0, 1.0)

