import numpy as np
import cv2
import matplotlib.pyplot as plt


# Input either image or image_path
import numpy as np
import cv2
import matplotlib.pyplot as plt


def compute_power_spectrum(image: np.ndarray = None, image_path: str = None):
	if image is None:
		# read as grayscale to avoid having to convert
		image_grayscale = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	else:
		image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Compute 2D FFT
	fft = np.fft.fft2(image_grayscale)
	fft_shifted = np.fft.fftshift(fft)

	# Compute power spectrum
	power_spectrum = np.abs(fft_shifted) ** 2

	# Compute radial average for radial power spectrum
	(h, w) = power_spectrum.shape
	(center_y, center_x) = (h // 2, w // 2)
	(y, x) = np.ogrid[:h, :w]
	r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2).astype(int)

	# Compute radial average
	max_radius = int(np.sqrt(center_x ** 2 + center_y ** 2))
	radial_profile = np.zeros(max_radius)

	for radius in range(max_radius):
		mask = (r == radius)
		if np.sum(mask) > 0:
			radial_profile[radius] = np.mean(power_spectrum[mask])

	return (power_spectrum, radial_profile)


def plot_power_spectrum(image: np.ndarray = None, image_path: str = None, title: str = None):
	(power_spectrum, radial_profile) = compute_power_spectrum(image, image_path)

	(h, w) = power_spectrum.shape
	(center_y, center_x) = (h // 2, w // 2)
	max_radius = int(np.sqrt(center_x ** 2 + center_y ** 2))

	# Log scale for better visualization
	power_spectrum_log = np.log10(power_spectrum + 1)

	# Plot radial power spectrum with normalized frequency
	# Normalize by Nyquist frequency
	frequencies_normalized = np.arange(max_radius) / max_radius

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

	im1 = ax1.imshow(power_spectrum_log, cmap='gray', origin='lower')
	ax1.set_title(f'2D Power Spectral Density\n{title}')
	ax1.set_xlabel('')
	ax1.set_ylabel('')
	ax1.set_xticks([])
	ax1.set_yticks([])

	# Plot radial power spectrum with normalized frequency
	# Normalize by Nyquist frequency
	frequencies_normalized = np.arange(max_radius) / max_radius

	ax2.loglog(frequencies_normalized[1:], radial_profile[1:])  # Skip DC component
	ax2.set_title(f'Radial Power Spectrum\n{title}')
	ax2.set_xlabel('Radial Frequency')
	ax2.set_ylabel('Power')
	ax2.grid(True, alpha=0.3, which='both')

	plt.tight_layout()
	plt.show()

	return (power_spectrum, radial_profile)




def plot_image_with_power_spectrum(image: np.ndarray = None, image_path: str = None, title: str = None):
	(power_spectrum, radial_profile) = compute_power_spectrum(image, image_path)

	if image is None:
		# OpenCV loads BGR by default
		image = cv2.imread(image_path, cv2.IMREAD_COLOR)

	# for matplotlib display (expects RGB)
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	(h, w) = power_spectrum.shape
	(center_y, center_x) = (h // 2, w // 2)
	max_radius = int(np.sqrt(center_x ** 2 + center_y ** 2))

	# Log scale for better visualization
	power_spectrum_log = np.log10(power_spectrum + 1.0)

	# frequencies normalized to Nyquist for the 1D curve
	# match length of radial_profile to avoid shape mismatches
	n_bins = len(radial_profile)
	frequencies_normalized = np.arange(n_bins, dtype=np.float32) / float(max(1, n_bins - 1))

	fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 5))

	ax0.imshow(image_rgb)
	ax0.set_title(f'Image {title}')
	ax0.set_axis_off()

	im1 = ax1.imshow(power_spectrum_log, cmap='gray', origin='lower')
	ax1.set_title(f'2D Power Spectral Density\n{title}')
	ax1.set_xlabel('')
	ax1.set_ylabel('')
	ax1.set_xticks([])
	ax1.set_yticks([])

	ax2.loglog(frequencies_normalized[1:], radial_profile[1:])  # Skip DC component
	ax2.set_title(f'Radial Power Spectrum\n{title}')
	ax2.set_xlabel('Radial Frequency')
	ax2.set_ylabel('Power')
	ax2.grid(True, alpha=0.3, which='both')

	plt.tight_layout()
	plt.show()

	return (power_spectrum, radial_profile)


def plot_three_radial_profiles(radial_profile_a: np.ndarray, radial_profile_b: np.ndarray, radial_profile_c: np.ndarray,
								title_a: str, title_b: str, title_c: str, main_title: str = None):
	n_a = len(radial_profile_a)
	n_b = len(radial_profile_b)
	n_c = len(radial_profile_c)

	freq_a = np.arange(n_a, dtype=np.float32) / float(max(1, n_a - 1))
	freq_b = np.arange(n_b, dtype=np.float32) / float(max(1, n_b - 1))
	freq_c = np.arange(n_c, dtype=np.float32) / float(max(1, n_c - 1))

	plt.figure(figsize=(15, 5))
	plt.loglog(freq_a[1:], radial_profile_a[1:], label = title_a)
	plt.loglog(freq_b[1:], radial_profile_b[1:], label = title_b)
	plt.loglog(freq_c[1:], radial_profile_c[1:], label = title_c)

	plt.xlabel('Radial Frequency (normalized to Nyquist)')
	plt.ylabel('Power')
	if main_title is not None:
		plt.title(main_title)
	plt.grid(True, which='both', ls=':', alpha=0.3)
	plt.legend()
	plt.tight_layout()