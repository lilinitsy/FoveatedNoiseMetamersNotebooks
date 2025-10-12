import numpy as np
import cv2


'''
Bounds in cycles per pixel, lower (3 * 1 / 2 * pi * sigma) and upper (Nyquist) cutoffs.
'''
def freq_bounds_cpp(sigma_pix, fl_tbl_min = 1 / 512, fh_limit = 0.5):
	with np.errstate(divide='ignore'):
		sigma_f = 1.0 / (2.0*np.pi*np.maximum(1e-6, sigma_pix))
	F_L = 3.0 * sigma_f
	F_H = fh_limit
	F_L = np.clip(F_L, fl_tbl_min, F_H*0.999)
	return F_L, F_H


'''
Use a Gaussian level and then Sobel for orientation.
'''
def estimate_orientation(image_rgb):

	H, W = image_rgb.shape[:2]
	small = cv2.resize(image_rgb, (W//8, H//8), interpolation=cv2.INTER_AREA)
	gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
	gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
	gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
	ang = np.arctan2(gy, gx)
	ang_full = cv2.resize(ang, (W, H), interpolation=cv2.INTER_CUBIC)
	return ang_full


def choose_laplacian_level_from_sigma(sigma_pix, a = 0.25):
    with np.errstate(divide = 'ignore'):
        sigma_f = 1.0 / (2.0*np.pi*np.maximum(1e-6, sigma_pix))
    f_c = np.sqrt(-np.log(a)) / (np.pi*np.maximum(1e-6, sigma_pix))
    l_a = -np.log2(np.maximum(1e-6, f_c)) - 0.5
    return l_a


def amplitude_from_laplacian(L_pyr, l_a_map, s_k=20.0):
	height, width = l_a_map.shape
	L = len(L_pyr) - 1  # last is Gaussian

	# precompute |L_l| at full-res
	full = []
	for l in range(L):
		Li = L_pyr[l]
		if Li.ndim == 3 and Li.shape[2] == 3:
			Li = cv2.cvtColor(Li, cv2.COLOR_BGR2GRAY)
		full.append(np.abs(cv2.resize(Li, (width, height), interpolation=cv2.INTER_CUBIC)).astype(np.float32))
	full = np.stack(full, axis=0)  # [L,H,W]

	l0 = np.clip(np.floor(l_a_map).astype(np.int32), 0, L-1)
	l1 = np.clip(l0 + 1, 0, L-1)
	t = (l_a_map - l0).astype(np.float32)

	yy = np.arange(height)[:, None]
	xx = np.arange(width)[None, :]
	A0 = full[l0, yy, xx]
	A1 = full[l1, yy, xx]
	amp = (1.0 - t) * A0 + t * A1
	return s_k * amp


def L_pyr_list_to_gray(level_image, H, W, level_idx):
	if level_image.ndim == 3 and level_image.shape[2] == 3:
		level_image = cv2.cvtColor(level_image, cv2.COLOR_BGR2GRAY)
	return cv2.resize(level_image, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32)



def gabor_kernel(freq_cpp, theta, sigma_pix_env=8.0, gamma=1.0):
	f = max(1e-4, float(freq_cpp))
	lam = 1.0 / f
	ksize = int(max(7, int(6*sigma_pix_env)|1))
	return cv2.getGaborKernel((ksize, ksize), sigma_pix_env, float(theta), lam, gamma, psi=0, ktype=cv2.CV_32F)

def synthesize_gabor_noise(height, width, F_L, F_H, amp, theta, cells=32, impulses_per_cell=12, seed=1337):
	rng = np.random.default_rng(seed)
	noise = np.zeros((height, width), np.float32)
	cell_h, cell_w = height // cells, width // cells
	for cy in range(cells):
		for cx in range(cells):
			y0, x0 = cy * cell_h, cx * cell_w
			y1, x1 = min(height, y0 + cell_h), min(width, x0 + cell_w)
			for _ in range(impulses_per_cell):
				y = rng.integers(y0, y1)
				x = rng.integers(x0, x1)
				# Sample a frequency in [F_L, F_H] at that location (log-normal would be closer to paper)
				fl, fh = F_L[y, x], F_H if np.isscalar(F_H) else F_H[y, x]
				fcpp = float(np.exp(rng.normal(np.log(np.sqrt(fl * fh)), 0.25)))  # crude log-normal
				th = float(theta[y, x])
				k = gabor_kernel(fcpp, th)
				kh, kw = k.shape
				oy, ox = kh // 2, kw // 2
				yb0, yb1 = max(0, y - oy), min(height, y + oy + 1)
				xb0, xb1 = max(0, x - ox), min(width, x + ox + 1)
				ky0, ky1 = oy - (y - yb0), oy + (yb1 - y) - 1
				kx0, kx1 = ox - (x - xb0), ox + (xb1 - x) - 1
				noise[yb0:yb1, xb0:xb1] += amp[y, x] * k[ky0:ky1 + 1, kx0:kx1 + 1]
	return noise


def contrast_enhance(image, fe = 0.2):
	blur = cv2.GaussianBlur(image, (0,0), 2.0, borderType = cv2.BORDER_REFLECT_101)
	return np.clip(image + fe * (image - blur), 0.0, 1.0).astype(image.dtype)
