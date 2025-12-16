import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener
import os

# 1. 準備輸入圖片 (使用 Wiener 濾波後的結果，如果沒有就現場重算)
save_dir = './img_results'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 讀取原圖並重做一次 Wiener 作為基底
img_path = './img/img1.tif'
if not os.path.exists(img_path) and os.path.exists('圖片.jpg'):
    img_path = '圖片.jpg'

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img_wiener = wiener(img.astype(np.float64), (5, 5))
img_wiener = np.clip(img_wiener, 0, 255).astype(np.uint8)

# ==========================================
# 方法 A: 增強對比度 (CLAHE)
# ==========================================
# clipLimit: 對比限制閾值，越大對比越強 (通常 2.0-4.0)
# tileGridSize: 切塊大小 (通常 8x8)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_clahe = clahe.apply(img_wiener)
cv2.imwrite(f'{save_dir}/advanced_1_clahe.jpg', img_clahe)

# ==========================================
# 方法 B: 銳利化 (Unsharp Masking)
# ==========================================
# 原理: 原始圖 + (原始圖 - 模糊圖) * amount
gaussian = cv2.GaussianBlur(img_wiener, (9, 9), 10.0)
img_sharp = cv2.addWeighted(img_wiener, 1.5, gaussian, -0.5, 0, img_wiener)
cv2.imwrite(f'{save_dir}/advanced_2_sharpened.jpg', img_sharp)

# ==========================================
# 方法 C: 非局部均值去噪 (Non-local Means) - 慢但強大
# ==========================================
# h: 決定過濾強度，越大去噪越強但細節越少 (通常 10-30)
# templateWindowSize: 尋找相似像素的窗口大小 (7)
# searchWindowSize: 搜索範圍 (21)
img_nlm = cv2.fastNlMeansDenoising(img, None, h=30, templateWindowSize=7, searchWindowSize=21)
cv2.imwrite(f'{save_dir}/advanced_3_nlm.jpg', img_nlm)

# ==========================================
# 顯示比較結果
# ==========================================
plt.figure(figsize=(12, 10))

# 1. Wiener (原本的最佳)
plt.subplot(2, 2, 1)
plt.imshow(img_wiener, cmap='gray')
plt.title('Base: Wiener Filter')
plt.axis('off')

# 2. CLAHE (增強對比)
plt.subplot(2, 2, 2)
plt.imshow(img_clahe, cmap='gray')
plt.title('A. CLAHE (Contrast)')
plt.axis('off')

# 3. Sharpened (銳利化)
plt.subplot(2, 2, 3)
plt.imshow(img_sharp, cmap='gray')
plt.title('B. Sharpening')
plt.axis('off')

# 4. Non-local Means (更強的去噪)
plt.subplot(2, 2, 4)
plt.imshow(img_nlm, cmap='gray')
plt.title('C. Non-local Means (Denoise)')
plt.axis('off')

plt.tight_layout()
plt.show()

print("進階處理完成！圖片已儲存至 img_results 資料夾。")