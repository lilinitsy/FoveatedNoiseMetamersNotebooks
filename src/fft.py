import numpy as np
import cv2
import matplotlib.pyplot as plt


# Input either image or image_path
import numpy as np
import cv2
import matplotlib.pyplot as plt

def plot_power_spectrum(image: np.ndarray = None, image_path: str = None, title: str = "None"):
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

	# Log scale for better visualization
	power_spectrum_log = np.log10(power_spectrum + 1)

	# Compute radial average for radial power spectrum
	(h, w) = power_spectrum.shape
	center_y, center_x = h // 2, w // 2
	(y, x) = np.ogrid[:h, :w]
	r = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)

	# Compute radial average
	max_radius = int(np.sqrt(center_x**2 + center_y**2))
	radial_profile = np.zeros(max_radius)

	for radius in range(max_radius):
		mask = (r == radius)
		if np.sum(mask) > 0:
			radial_profile[radius] = np.mean(power_spectrum[mask])

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

	return power_spectrum, radial_profile



