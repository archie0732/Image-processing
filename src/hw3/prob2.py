import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():

    fn = cv2.imread('./img/fn.tif', cv2.IMREAD_GRAYSCALE)

    if fn is None:
        raise 'fn is None'
    

    kernel_size = 5
    img_avg = cv2.blur(fn, (kernel_size, kernel_size))
    img_mid = cv2.medianBlur(fn, kernel_size)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(fn, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.imshow(img_avg, cmap='gray')

    plt.subplot(1, 3 ,3)
    plt.imshow(img_mid, cmap='gray')

    plt.tight_layout()
    plt.show()

main()