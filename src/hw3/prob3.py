import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener


def add_gauss_prime(image: np.ndarray, mean = 0, var = 0.05):

    image_norm = image.astype('float') / 255.0
    sigma: float =  var ** 0.5

    noise: np.ndarray = np.random.normal(mean, sigma, image.shape)

    noise_image: np.ndarray = image_norm + noise
    noise_image = np.clip(noise_image, 0, 1)

    noise_image = (noise_image * 255).astype('uint8')

    return noise_image

def main():
    image = cv2.imread('./img/im1.tif')

    if image is None:
        raise 'image is None'

    f = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gauss_prime_image = add_gauss_prime(f)

    avg_filter_img = cv2.blur(gauss_prime_image, (5, 5))

    accumulator = np.zeros_like(f, dtype=np.float64)

    for _ in range(100):
        noise_sample = add_gauss_prime(f)
        accumulator += noise_sample
    
    img_average = (accumulator / 100).astype('uint8')

    img_wiener: np.ndarray = wiener(gauss_prime_image.astype('float64'), mysize=(5, 5))
    img_wiener = img_wiener.astype('uint8')

    plt.figure(figsize=(16, 6))

    plt.subplot(1, 4, 1)
    plt.imshow(f, cmap='gray')

    plt.subplot(1, 4, 2)
    plt.imshow(avg_filter_img, cmap='gray')

    plt.subplot(1, 4, 3)
    plt.imshow(img_average, cmap='gray')

    plt.subplot(1, 4, 4)
    plt.imshow(img_wiener, cmap='gray')

    plt.tight_layout()
    plt.show()
main()