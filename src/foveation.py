import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.pyramids import make_gaussian_pyramid, make_laplacian_pyramid
from src.utils import load_image, pixels_per_degree

from src.gabor_synthesis import (
	freq_bounds_cpp,
	estimate_orientation,
	choose_laplacian_level_from_sigma,
	amplitude_from_laplacian,
	synthesize_gabor_noise,
	contrast_enhance
)

def smoothstep(x, lo, hi):
	t = np.clip((x - lo) / max(1e-8, (hi - lo)), 0.0, 1.0)
	return t * t * (3 - 2 * t)


def radial_sigma_map(height, width, gaze_xy, pixels_per_degree, blur_rate_arcmin_per_degree, fovea_radius_deg = 5.0):
	y, x = np.mgrid[0:height, 0:width]
	dx = (x - gaze_xy[0]) / pixels_per_degree
	dy = (y - gaze_xy[1]) / pixels_per_degree
	ecc_deg = np.sqrt(dx*dx + dy*dy)

	# zero in fovea, linear outside
	ramp = np.clip(ecc_deg - fovea_radius_deg, 0, None)
	sigma_deg = (blur_rate_arcmin_per_degree / 60.0) * ramp
	sigma_pix = sigma_deg * pixels_per_degree
	return sigma_pix


def foveate_image(image, sigma_map):
	sigmas = [0.0, 1.0, 2.0, 4.0, 8.0]
	blurred = [image]
	for s in sigmas[1:]:
		k = max(3, int(6*s+1) | 1)
		blurred.append(cv2.GaussianBlur(image, (k, k), s, borderType=cv2.BORDER_REFLECT_101))

	s = np.clip(sigma_map, sigmas[0], sigmas[-1])
	idx0 = np.searchsorted(sigmas, s, side='right') - 1
	idx0 = np.clip(idx0, 0, len(sigmas)-2)
	idx1 = idx0 + 1
	s0 = np.asarray(sigmas)[idx0]
	s1 = np.asarray(sigmas)[idx1]
	t = ((s - s0) / np.maximum(1e-8, (s1 - s0))).astype(np.float32)

	stack = np.stack(blurred, axis=0)              # [K,H,W,C]
	H, W = s.shape
	yy = np.arange(H)[:, None]
	xx = np.arange(W)[None, :]
	I0 = stack[idx0, yy, xx]                       # [H,W,C]
	I1 = stack[idx1, yy, xx]
	out = (1.0 - t)[..., None] * I0 + t[..., None] * I1
	return out.astype(image.dtype)



def generate_foveated_image(
    path,
    levels: int = 5,
    gaze_xy: tuple | None = None,
    dpi: float = 110.0,
    viewing_distance_m: float = 0.6,
    blur_rate_arcmin_per_degree: float = 0.34,
    fe: float = 0.2,
    s_k: float = 20.0,
    cells: int = 32,
    impulses_per_cell: int = 12,
    seed: int = 10
):
	image = load_image(path, grayscale=False, as_float=True)   # BGR in [0,1]
	H, W = image.shape[:2]
	if gaze_xy is None:
		gaze_xy = (W // 2, H // 2)

	# per-pixel blur sigma map (pixels)
	ppd = pixels_per_degree(dpi, viewing_distance_m)
	sigma_map = radial_sigma_map(H, W, gaze_xy, ppd, blur_rate_arcmin_per_degree)

	# optional preview of just the variable blur
	foveated_image = foveate_image(image, sigma_map)

	# pyramids
	gaussian_pyramid = make_gaussian_pyramid(image, levels)
	laplacian_pyramid = make_laplacian_pyramid(image, levels, gaussian_pyramid = gaussian_pyramid)

	# frequency bounds & orientation
	F_L, F_H = freq_bounds_cpp(sigma_map)  # F_H is scalar 0.5 inside if you kept that default
	theta = estimate_orientation(image)

	# amplitude from Laplacian bands
	l_a = choose_laplacian_level_from_sigma(sigma_map)
	amp = amplitude_from_laplacian(laplacian_pyramid, l_a, s_k=s_k)

	w = smoothstep(sigma_map, 0.5, 2.0)   # 0 in fovea, 1 in periphery
	amp *= w

	noise = synthesize_gabor_noise(H, W, F_L, 0.5, amp, theta,
								cells=cells, impulses_per_cell=impulses_per_cell, seed=seed)

	# Weighted contrast enhancement (from Patney)
	blur = cv2.GaussianBlur(image, (0, 0), 2.0, borderType=cv2.BORDER_REFLECT_101)
	ce   = np.clip(image + (fe * w)[..., None] * (image - blur), 0.0, 1.0)

	Y  = 0.2126 * ce[..., 2] + 0.7152 * ce[..., 1] + 0.0722 * ce[..., 0]
	Yn = np.clip(Y + noise, 0.0, 1.0)

	eps = 1e-6
	scale = ((Yn + eps) / (Y + eps))[..., None]
	final_enhanced_image = np.clip(ce * scale, 0.0, 1.0)

	return final_enhanced_image, foveated_image, sigma_map


def display_foveated_results(final_enhanced_image, foveated_image, sigma_map):
	def to_rgb(img):
		return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if (img.ndim == 3 and img.shape[2] == 3) else img

	fig, axes = plt.subplots(1, 3, figsize=(16, 5))

	axes[0].imshow(to_rgb(foveated_image))
	axes[0].set_title("Foveated (blur only)")
	axes[0].axis("off")

	im = axes[1].imshow(sigma_map, cmap="magma")
	axes[1].set_title("Sigma map (pixels)")
	axes[1].axis("off")
	fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

	axes[2].imshow(to_rgb(final_enhanced_image))
	axes[2].set_title("Final output (final_enhanced_image)")
	axes[2].axis("off")

	plt.tight_layout()
	plt.show()
