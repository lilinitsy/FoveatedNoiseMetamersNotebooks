import os
import numpy as np
import cv2
from scipy.stats import qmc

from src.gabor_synthesis import	choose_laplacian_level_from_sigma
from src.pyramids import make_gaussian_pyramid, make_laplacian_pyramid




def poisson_disk_points(height, width, radius_px, seed=0):
	r_unit = radius_px / max(height, width)
	eng = qmc.PoissonDisk(d = 2, radius = r_unit, ncandidates = 30, seed = seed)
	pts01 = eng.fill_space() 
	points = np.column_stack([pts01[:, 0] * width, pts01[:, 1] * height]).astype(np.float32)
	return points


# sigma_blob correspond to pixel width
# so with sigma_blob = 0.5, this corresponds to
# 68% of the distribution within 1 pixel (+0.5, -0.5),
# 95% within 2 totals (total width, a 2x2 window), and
# 97% within 3 width (1.5 either side = 3x3 window)

# This function creates a field of Gaussians centered around the poisson points
def blue_noise_field_from_poisson_points(height, width, r_px = 12, seed = 0, sigma_blob = 0.5):
	points = poisson_disk_points(height, width, r_px, seed)
	field = np.zeros((height, width), np.float32)
	kernel_size = int(max(7, int(6 * sigma_blob )+ 1)) | 1
	gaussian_term = cv2.getGaussianKernel(kernel_size, sigma_blob)
	gaussian_blob = (gaussian_term @ gaussian_term.T).astype(np.float32)
	half_y = half_x = kernel_size // 2

	# This splats a single gaussian at each point.
	# I wonder if multiple gaussians need to be splatted... take into account feature orientation?
	for (x, y) in points:
		(xi, yi) = int(round(x)), int(round(y)) # center
		(y0, y1) = max(0, yi - half_y), min(height, yi + half_y + 1)  # bottom to top
		(x0, x1) = max(0, xi - half_x), min(width, xi + half_x + 1) # left to right
		(by0, by1) = half_y - (yi - y0), half_y + (y1 - yi) - 1
		(bx0, bx1) = half_x - (xi - x0), half_x + (x1 - xi) - 1
		field[y0:y1, x0:x1] += gaussian_blob[by0:by1 + 1, bx0:bx1 + 1]
	field -= field.mean()
	field /= (field.std() + 1e-6)
	
	field_norm = (field - field.min()) / (field.max() - field.min())


	return field_norm


def blue_noise_field_from_poisson_points2(height, width, r_px = 12, seed = 0, sigma_blob = 0.5):
	points = poisson_disk_points(height, width, r_px, seed)
	field = np.zeros((height, width), np.float32)

	kappa = 1.6
	sigma1 = float(sigma_blob)
	sigma2 = max(1e-6, kappa * sigma1)
	kernel_size = (int(max(7, int(6 * sigma2) + 1)) | 1)
	ga1 = cv2.getGaussianKernel(kernel_size, sigma1).astype(np.float32)
	ga2 = cv2.getGaussianKernel(kernel_size, sigma2).astype(np.float32)
	G1 = (ga1 @ ga1.T)
	G2 = (ga2 @ ga2.T)
	alpha = float(G1.sum()) / (float(G2.sum()) + 1e-12)
	K = G1 - alpha * G2
	K -= K.mean()
	K /= (np.sqrt((K * K).sum()) + 1e-12)
	half_y = half_x = kernel_size // 2

	for (x, y) in points:
		(xi, yi) = int(round(x)), int(round(y))
		(y0, y1) = max(0, yi - half_y), min(height, yi + half_y + 1)
		(x0, x1) = max(0, xi - half_x), min(width, xi + half_x + 1)
		(by0, by1) = half_y - (yi - y0), half_y + (y1 - yi) - 1
		(bx0, bx1) = half_x - (xi - x0), half_x + (x1 - xi) - 1
		Kc = K[by0:by1 + 1, bx0:bx1 + 1]
		Kc = Kc - Kc.mean()
		Kc /= (np.sqrt((Kc * Kc).sum()) + 1e-12)
		field[y0:y1, x0:x1] += Kc

	field -= field.mean()
	field /= (field.std() + 1e-6)

	return field






def bandlimited_blue_noise(height, width, sigma_map, levels = 5, r_px = 12, seed = 0, sigma_blob = 0.5):
	field = blue_noise_field_from_poisson_points2(height, width, r_px = r_px, seed = seed, sigma_blob = sigma_blob)

	gaussian_pyramid = make_gaussian_pyramid(field, levels)
	laplacian_pyramid = make_laplacian_pyramid(field, levels, gaussian_pyramid = gaussian_pyramid)

	# Resize all Laplacian bands to full resolution
	full = []
	for l in range(levels - 1):
		full.append(cv2.resize(laplacian_pyramid[l].astype(np.float32), (width, height), interpolation = cv2.INTER_CUBIC))
	full = np.stack(full, axis = 0)

	l_a = choose_laplacian_level_from_sigma(sigma_map)
	l0 = np.clip(np.floor(l_a).astype(np.int32), 0, levels - 2)
	l1 = np.clip(l0 + 1, 0, levels - 2)
	t = (l_a - l0).astype(np.float32)

	yy = np.arange(height)[:, None]
	xx = np.arange(width)[None, :]
	N0 = full[l0, yy, xx]
	N1 = full[l1, yy, xx]
	result = (1.0 - t) * N0 + t * N1

	result -= result.mean()
	result /= (result.std() + 1e-6)

	return result


def radial_sigma_map(height, width, gaze_xy, px_per_deg, blur_rate_arcmin_per_degree, fovea_radius_deg = 5.0):
	y, x = np.mgrid[0:height, 0:width]
	dx = (x - gaze_xy[0]) / px_per_deg
	dy = (y - gaze_xy[1]) / px_per_deg
	ecc_deg = np.sqrt(dx * dx + dy * dy)
	ramp = np.clip(ecc_deg - fovea_radius_deg, 0.0, None)
	sigma_deg = (blur_rate_arcmin_per_degree / 60.0) * ramp
	return sigma_deg * px_per_deg

def smoothstep(x, lo, hi):
	t = np.clip((x - lo) / max(1e-8, (hi - lo)), 0.0, 1.0)
	return t * t * (3.0 - 2.0 * t)





