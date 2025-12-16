import cv2
import numpy as np
import os

# 1. 讀取圖片 (讀取剛剛轉好的灰階圖)
input_filename = './img/img1_gray.jpg'
if not os.path.exists(input_filename):
    # 如果找不到 jpg，嘗試讀取原圖
    img = cv2.imread('圖片.jpg', cv2.IMREAD_GRAYSCALE)
else:
    img = cv2.imread(input_filename, cv2.IMREAD_GRAYSCALE)

# ==========================================
# A. 中位數濾波 (Median Filter)
# ==========================================
# ksize=5 表示使用 5x5 的區域來找中位數
img_median = cv2.medianBlur(img, 5)

# ==========================================
# B. 最小值濾波 (Minimum Filter)
# ==========================================
# 在 OpenCV 中，最小值濾波等同於 "侵蝕 (Erosion)"
# 我們定義一個 3x3 的核 (Kernel)
kernel = np.ones((3, 3), np.uint8) 
img_min = cv2.erode(img, kernel, iterations=1)

# ==========================================
# 存檔
# ==========================================
cv2.imwrite('result_median.jpg', img_median)
cv2.imwrite('result_min.jpg', img_min)

print("處理完成！")
print("中位數濾波結果已存為: result_median.jpg")
print("最小值濾波結果已存為: result_min.jpg")