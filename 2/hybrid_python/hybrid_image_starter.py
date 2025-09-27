import matplotlib.pyplot as plt
import numpy as np
from align_image_code import align_images
from scipy.signal import convolve2d
import cv2

# First load images

# high sf
im1 = plt.imread('./hybrid_python/DerekPicture.jpg')/255.
im1_gray = cv2.imread('./hybrid_python/DerekPicture.jpg', cv2.IMREAD_GRAYSCALE)

# low sf
im2 = plt.imread('./hybrid_python/nutmeg.jpg')/255
im2_gray = cv2.imread('./hybrid_python/nutmeg.jpg', cv2.IMREAD_GRAYSCALE)


#Log Magnitudes
plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(im1_gray)))), cmap='gray')
plt.title("Log magnitude im1")
plt.tight_layout()
plt.show()
plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(im2_gray)))), cmap='gray')
plt.title("Log magnitude im2")
plt.tight_layout()
plt.show()

#DEBUG
# im1 = cv2.imread('./hybrid_python/DerekPicture.jpg', cv2.IMREAD_GRAYSCALE)
# im2 = cv2.imread('./hybrid_python/nutmeg.jpg', cv2.IMREAD_GRAYSCALE)

# Next align images (this code is provided, but may be improved)
im1_aligned, im2_aligned = align_images(im1, im2)

## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies
hybrid = np.zeros_like(im1_aligned)
sigma1 = 10.0
sigma2 = 3.0

#Define Gaussian and Laplacian Filters
lowpass = np.outer(cv2.getGaussianKernel(27, sigma1), cv2.getGaussianKernel(27, sigma1).T)
highpass = np.outer(cv2.getGaussianKernel(27, sigma2), cv2.getGaussianKernel(27, sigma2).T)


for c in range(3):
    #Convolve and add
    #Low Frequencies
    gaussian_image = convolve2d(im2_aligned[:,:,c], lowpass, mode='same', boundary='fill', fillvalue=0)
    
    #High Frequencies
    lfreq = convolve2d(im1_aligned[:,:,c], highpass, mode='same', boundary='fill', fillvalue=0)
    laplacian_image = im1_aligned[:,:,c] - lfreq

    hybrid[:,:,c] = gaussian_image + laplacian_image


# gaussian_image = convolve2d(im2_aligned, lowpass, mode='same', boundary='fill', fillvalue=0)

# #High Frequencies
# lfreq = convolve2d(im1_aligned, highpass, mode='same', boundary='fill', fillvalue=0)
# laplacian_image = im1_aligned - lfreq

# hybrid = gaussian_image + laplacian_image
hybrid_gray = np.mean(hybrid, axis=2)

plt.imshow(hybrid)
plt.title(f"Hybrid image")
plt.tight_layout()
plt.show()

plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(hybrid_gray)))), cmap='gray')
plt.title("Log magnitude hybrid image")
plt.tight_layout()
plt.show()

