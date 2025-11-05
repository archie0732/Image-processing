import cv2

im = cv2.imread('image.jpg')

r = cv2.selectROI("Image", im)

(x, y, w, h) = r
print(f"Selected ROI - x:{x}, y:{y}, width:{w}, height:{h}")

