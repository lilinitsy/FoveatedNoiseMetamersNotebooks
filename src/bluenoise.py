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

	cv2.imwrite("poisson_points.png", (np.clip(points, 0, 1) * 255).astype(np.uint8))
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


def blue_noise_field_from_poisson_points2(height, width, r_px = 12, seed = 0, sigma_blob = 0.5, theta_map = None, coherency_map = None, rho_min = 1.0, rho_max = 3.0):
	points = poisson_disk_points(height, width, r_px, seed)
	field = np.zeros((height, width), np.float32)
	kappa = 1.6
	for (x, y) in points:
		(xi, yi) = int(round(x)), int(round(y))
		if theta_map is None:
			theta = 0.0
			coh = 0.0
		else:
			yi_clamp = int(np.clip(yi, 0, height - 1))
			xi_clamp = int(np.clip(xi, 0, width - 1))
			theta = float(theta_map[yi_clamp, xi_clamp])
			coh = float(coherency_map[yi_clamp, xi_clamp])
		rho = rho_min + (rho_max - rho_min) * coh
		sigma_minor = float(sigma_blob)
		sigma_major = rho * sigma_minor
		sigma1_u = sigma_major
		sigma1_v = sigma_minor
		sigma2_u = kappa * sigma1_u
		sigma2_v = kappa * sigma1_v
		s_max = max(sigma2_u, sigma2_v)
		kernel_size = int(max(7, int(6 * s_max) + 1)) | 1
		half = kernel_size // 2
		yy, xx = np.mgrid[-half:half+1, -half:half+1].astype(np.float32)
		ct = np.cos(-theta)
		st = np.sin(-theta)
		u =  ct * xx - st * yy
		v =  st * xx + ct * yy
		G1 = np.exp(-0.5 * ((u * u) / (sigma1_u * sigma1_u + 1e-12) + (v * v) / (sigma1_v * sigma1_v + 1e-12))).astype(np.float32)
		G2 = np.exp(-0.5 * ((u * u) / (sigma2_u * sigma2_u + 1e-12) + (v * v) / (sigma2_v * sigma2_v + 1e-12))).astype(np.float32)
		alpha = float(G1.sum()) / (float(G2.sum()) + 1e-12)
		K = G1 - alpha * G2
		K -= K.mean()
		K /= (np.sqrt((K * K).sum()) + 1e-12)
		(y0, y1) = max(0, yi - half), min(height, yi + half + 1)
		(x0, x1) = max(0, xi - half), min(width, xi + half + 1)
		(by0, by1) = half - (yi - y0), half + (y1 - yi) - 1
		(bx0, bx1) = half - (xi - x0), half + (x1 - xi) - 1
		Kc = K[by0:by1 + 1, bx0:bx1 + 1]
		Kc = Kc - Kc.mean()
		Kc /= (np.sqrt((Kc * Kc).sum()) + 1e-12)
		field[y0:y1, x0:x1] += Kc
	field -= field.mean()
	field /= (field.std() + 1e-6)
	return field





def compute_orientation_field(img, sigma_smooth = 1.0, eps = 1e-6):
	Y  = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
	Ix = cv2.Sobel(Y, cv2.CV_32F, 1, 0, ksize = 3)
	Iy = cv2.Sobel(Y, cv2.CV_32F, 0, 1, ksize = 3)
	Ixx = Ix * Ix
	Iyy = Iy * Iy
	Ixy = Ix * Iy
	if sigma_smooth > 0:
		Ixx = cv2.GaussianBlur(Ixx, (0, 0), sigmaX = sigma_smooth, sigmaY = sigma_smooth, borderType = cv2.BORDER_REPLICATE)
		Iyy = cv2.GaussianBlur(Iyy, (0, 0), sigmaX = sigma_smooth, sigmaY = sigma_smooth, borderType = cv2.BORDER_REPLICATE)
		Ixy = cv2.GaussianBlur(Ixy, (0, 0), sigmaX = sigma_smooth, sigmaY = sigma_smooth, borderType = cv2.BORDER_REPLICATE)
	trace = Ixx + Iyy
	diff  = Ixx - Iyy
	theta = 0.5 * np.arctan2(2.0 * Ixy, diff + eps) + np.pi * 0.5
	l1 = 0.5 * (trace + np.sqrt(diff * diff + 4.0 * Ixy * Ixy))
	l2 = 0.5 * (trace - np.sqrt(diff * diff + 4.0 * Ixy * Ixy))
	coherency = (l1 - l2) / (trace + eps)
	coherency = np.clip(coherency, 0.0, 1.0).astype(np.float32)
	return theta.astype(np.float32), coherency



def bandlimited_blue_noise(height, width, sigma_map, levels = 5, r_px = 12, seed = 0, sigma_blob = 0.5, theta_map = None, coherency_map = None, rho_min = 1.0, rho_max = 3.0):
	field = blue_noise_field_from_poisson_points2(height, width, r_px = r_px, seed = seed, sigma_blob = sigma_blob, theta_map = theta_map, coherency_map = coherency_map, rho_min = rho_min, rho_max = rho_max)
	
	gaussian_pyramid = make_gaussian_pyramid(field, levels)
	laplacian_pyramid = make_laplacian_pyramid(field, levels, gaussian_pyramid = gaussian_pyramid)
	
	# resize the laplacians to rullres
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





