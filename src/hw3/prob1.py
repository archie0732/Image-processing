import cv2
import numpy as np
import matplotlib.pyplot as plt

# 資工三A 411211480 許育祁
# imgae: im1.tif


def hw_three_prob_one(image: np.ndarray, percent):

    row, col = image.shape

    noise_image = image.copy()

    total_pixel = row * col
    num_noise = int(total_pixel * percent)
    num_salt = num_noise // 2
    num_pepper = num_noise // 2

    salt_position = [ np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noise_image[tuple(salt_position)] = 255 

    pepper_position = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noise_image[tuple(pepper_position)] = 0

    return noise_image



def main():
    image = cv2.imread('./img/im1.tif')

    if image is None:
        raise 'image is None'
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    f = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    fn = hw_three_prob_one(f, 0.1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.imshow(f, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.imshow(fn, cmap='gray')

    plt.tight_layout()
    plt.show()

    cv2.imwrite('./img/fn.tif', fn)
    cv2.imwrite('img/f.tif', f)

main()


