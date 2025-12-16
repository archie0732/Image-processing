import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener
import os

# ==========================================
# 0. 前置設定
# ==========================================
# 建立輸出資料夾
save_dir = './img_results'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 設定輸入圖片路徑
input_path = './img/img1.tif' 
# 如果找不到 tif，嘗試找 jpg (防呆機制)
if not os.path.exists(input_path):
    if os.path.exists('圖片.jpg'):
        input_path = '圖片.jpg'
    elif os.path.exists('img1_gray.jpg'):
        input_path = 'img1_gray.jpg'

print(f"正在讀取圖片: {input_path}")

# 1. 讀取並轉灰階
img_bgr = cv2.imread(input_path)
if img_bgr is None:
    print("錯誤: 找不到圖片，請確認路徑。")
    exit()
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
cv2.imwrite(f'{save_dir}/0_original.jpg', img_gray)

# ==========================================
# 2. 開始執行各種濾波
# ==========================================

# --- A. 平均濾波 (Average Filter) ---
# 讓鄰域內的像素取平均值，均勻模糊
img_mean = cv2.blur(img_gray, (5, 5))
cv2.imwrite(f'{save_dir}/1_average_filter.jpg', img_mean)

# --- B. 高斯濾波 (Gaussian Filter) ---
# 根據距離給權重，比平均濾波保留更多輪廓
img_gaussian = cv2.GaussianBlur(img_gray, (5, 5), 0)
cv2.imwrite(f'{save_dir}/2_gaussian_filter.jpg', img_gaussian)

# --- C. 中位數濾波 (Median Filter) ---
# 取中間值，去除極端亮/暗點 (椒鹽雜訊殺手)
img_median = cv2.medianBlur(img_gray, 5)
cv2.imwrite(f'{save_dir}/3_median_filter.jpg', img_median)

# --- D. 最小值濾波 (Minimum Filter / Erosion) ---
# 取鄰域最暗點 (去除亮雜訊，字體變粗)
kernel = np.ones((3, 3), np.uint8) # 3x3 效果較剛好
img_min = cv2.erode(img_gray, kernel, iterations=1)
cv2.imwrite(f'{save_dir}/4_min_filter.jpg', img_min)

# --- E. 最大值濾波 (Maximum Filter / Dilation) ---
# 取鄰域最亮點 (去除暗雜訊，字體變細)
img_max = cv2.dilate(img_gray, kernel, iterations=1)
cv2.imwrite(f'{save_dir}/5_max_filter.jpg', img_max)

# --- F. Wiener 濾波 (Wiener Filter) ---
# 自適應濾波，保留高頻細節
img_wiener = wiener(img_gray.astype(np.float64), (5, 5))
img_wiener = np.clip(img_wiener, 0, 255).astype(np.uint8)
cv2.imwrite(f'{save_dir}/6_wiener_filter.jpg', img_wiener)

# --- G. 影像平均法 (Image Averaging) ---
# 模擬疊加 100 張高斯雜訊圖後取平均
h, w = img_gray.shape
accumulated_img = np.zeros((h, w), dtype=np.float64)
num_images = 100
np.random.seed(42)
for i in range(num_images):
    noise = np.random.normal(0, 25, (h, w))
    noisy_img = img_gray.astype(np.float64) + noise
    accumulated_img += noisy_img
img_averaging = np.clip(accumulated_img / num_images, 0, 255).astype(np.uint8)
cv2.imwrite(f'{save_dir}/7_image_averaging.jpg', img_averaging)

print(f"所有圖片已儲存至 {save_dir} 資料夾！")

# ==========================================
# 3. 繪製比較總表
# ==========================================
plt.figure(figsize=(16, 8))

titles = ['Original', 'Average (Mean)', 'Gaussian', 'Median', 
          'Min (Erosion)', 'Max (Dilation)', 'Wiener', 'Img Averaging']
images = [img_gray, img_mean, img_gaussian, img_median, 
          img_min, img_max, img_wiener, img_averaging]

for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.savefig(f'{save_dir}/all_filters_comparison.png')
plt.show()