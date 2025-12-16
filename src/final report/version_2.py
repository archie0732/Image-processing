import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener
import os  # 新增: 用來處理資料夾路徑

# ==========================================
# 0. 前置設定
# ==========================================

# 確保 ./img 資料夾存在，如果不存在就建立
save_dir = './img'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 設定圖片路徑
img_path = './img/img1.tif' 
output_path = './img/result_img.tif' # 設定輸出的檔名

# 1. 讀取影像並轉為灰階
try:
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"找不到圖片: {img_path}")
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
except Exception as e:
    print(f"錯誤: {str(e)}")
    # 若讀不到圖則產生隨機圖 (避免程式崩潰)
    img_gray = np.random.randint(0, 256, (256, 256), dtype=np.uint8)

# ==========================================
# Part 1: 各種濾波去噪 (作業要求 1)
# ==========================================

# A. 平均濾波 (Average Filter)
img_mean_filter = cv2.blur(img_gray, (5, 5))

# B. 中位數濾波 (Median Filter)
img_median_filter = cv2.medianBlur(img_gray, 5)

# C. 影像平均法 (Image Averaging)
h, w = img_gray.shape
accumulated_img = np.zeros((h, w), dtype=np.float64)
num_images = 100
np.random.seed(42) 
for i in range(num_images):
    noise = np.random.normal(0, 25, (h, w))
    noisy_img = img_gray.astype(np.float64) + noise
    accumulated_img += noisy_img
img_averaging = np.clip(accumulated_img / num_images, 0, 255).astype(np.uint8)

# D. Wiener 濾波 (視為最佳還原結果)
img_wiener = wiener(img_gray.astype(np.float64), (5, 5))
img_wiener = np.clip(img_wiener, 0, 255).astype(np.uint8)

# 設定 "最佳還原影像" 為 Wiener 濾波結果
best_restored_img = img_wiener.copy()

# ==========================================
# Part 2: 邊緣偵測 (作業要求 2)
# ==========================================
# 使用 Canny 演算法找出最佳還原影像的邊緣
img_edges = cv2.Canny(best_restored_img, 50, 150)

# ==========================================
# Part 3: 4灰階與最佳混色 (作業要求 3)
# ==========================================

# A. 單純 4 灰階量化
def quantize_4_levels(image):
    return (np.round((image / 255.0) * 3) / 3 * 255).astype(np.uint8)

img_quantized_simple = quantize_4_levels(best_restored_img)

# B. Floyd-Steinberg Dithering (最佳混色效果)
def floyd_steinberg_dither(image, levels=4):
    h, w = image.shape
    dithered = image.astype(np.float64)
    max_val = 255.0
    step = max_val / (levels - 1)

    for y in range(h):
        for x in range(w):
            old_pixel = dithered[y, x]
            new_pixel = round(old_pixel / step) * step
            dithered[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            
            if x + 1 < w:
                dithered[y, x + 1] += quant_error * 7 / 16
            if x - 1 >= 0 and y + 1 < h:
                dithered[y + 1, x - 1] += quant_error * 3 / 16
            if y + 1 < h:
                dithered[y + 1, x] += quant_error * 5 / 16
            if x + 1 < w and y + 1 < h:
                dithered[y + 1, x + 1] += quant_error * 1 / 16
                
    return np.clip(dithered, 0, 255).astype(np.uint8)

img_dithered = floyd_steinberg_dither(best_restored_img, levels=4)

# ==========================================
# 儲存與顯示結果
# ==========================================

# [新增功能] 將結果存檔
cv2.imwrite(output_path, img_dithered)
print(f"成功儲存檔案至: {output_path}")

# 顯示比較圖
plt.figure(figsize=(15, 12))

# --- 第一列：去噪比較 ---
plt.subplot(3, 3, 1)
plt.imshow(img_gray, cmap='gray')
plt.title('1. Original Image')
plt.axis('off')

plt.subplot(3, 3, 2)
plt.imshow(img_median_filter, cmap='gray')
plt.title('1. Median Filter')
plt.axis('off')

plt.subplot(3, 3, 3)
plt.imshow(img_wiener, cmap='gray')
plt.title('1. Wiener Filter (Best Restored)')
plt.axis('off')

# --- 第二列：邊緣偵測 ---
plt.subplot(3, 2, 3)
plt.imshow(best_restored_img, cmap='gray')
plt.title('2. Before: Restored Image')
plt.axis('off')

plt.subplot(3, 2, 4)
plt.imshow(img_edges, cmap='gray')
plt.title('2. After: Edge Detection (Canny)')
plt.axis('off')

# --- 第三列：混色比較 ---
plt.subplot(3, 2, 5)
plt.imshow(img_quantized_simple, cmap='gray', vmin=0, vmax=255)
plt.title('3. 4-Level Simple Quantization')
plt.axis('off')

plt.subplot(3, 2, 6)
plt.imshow(img_dithered, cmap='gray', vmin=0, vmax=255)
plt.title('3. 4-Level Dithered (Final Output)')
plt.axis('off')

plt.tight_layout()
plt.show()