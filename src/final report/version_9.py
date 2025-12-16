import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener
import os

# ==========================================
# 1. 準備影像資料
# ==========================================
save_dir = './img_results'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 讀取原圖
img_path = './img/img1.tif'
if not os.path.exists(img_path) and os.path.exists('圖片.jpg'):
    img_path = '圖片.jpg'
    
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# 若讀不到圖則中斷
if img is None:
    print("錯誤: 找不到圖片")
    exit()

# --- 重現剛剛的四種處理結果 ---

# 1. Wiener (基礎)
img_wiener = wiener(img.astype(np.float64), (5, 5))
img_wiener = np.clip(img_wiener, 0, 255).astype(np.uint8)

# 2. CLAHE (對比增強)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_clahe = clahe.apply(img_wiener)

# 3. Sharpening (銳利化)
gaussian = cv2.GaussianBlur(img_wiener, (9, 9), 10.0)
img_sharp = cv2.addWeighted(img_wiener, 1.5, gaussian, -0.5, 0, img_wiener)

# 4. Non-local Means (強力去噪)
# h 參數設為 30 以獲得明顯的去噪效果
img_nlm = cv2.fastNlMeansDenoising(img, None, h=30, templateWindowSize=7, searchWindowSize=21)

# ==========================================
# 2. 執行 Canny 邊緣偵測
# ==========================================
# 閾值設定: 低閾值 50, 高閾值 150 (常用設定)
threshold_low = 50
threshold_high = 150

edge_wiener = cv2.Canny(img_wiener, threshold_low, threshold_high)
edge_clahe = cv2.Canny(img_clahe, threshold_low, threshold_high)
edge_sharp = cv2.Canny(img_sharp, threshold_low, threshold_high)
edge_nlm = cv2.Canny(img_nlm, threshold_low, threshold_high)

# 儲存結果
cv2.imwrite(f'{save_dir}/edge_1_wiener.jpg', edge_wiener)
cv2.imwrite(f'{save_dir}/edge_2_clahe.jpg', edge_clahe)
cv2.imwrite(f'{save_dir}/edge_3_sharp.jpg', edge_sharp)
cv2.imwrite(f'{save_dir}/edge_4_nlm.jpg', edge_nlm)

# ==========================================
# 3. 顯示比較圖
# ==========================================
plt.figure(figsize=(12, 12))

# 1. Wiener Edges
plt.subplot(2, 2, 1)
plt.imshow(edge_wiener, cmap='gray')
plt.title('Edges: Wiener Filter')
plt.axis('off')

# 2. CLAHE Edges
plt.subplot(2, 2, 2)
plt.imshow(edge_clahe, cmap='gray')
plt.title('Edges: CLAHE (High Contrast)')
plt.axis('off')

# 3. Sharpening Edges
plt.subplot(2, 2, 3)
plt.imshow(edge_sharp, cmap='gray')
plt.title('Edges: Sharpened')
plt.axis('off')

# 4. NLM Edges
plt.subplot(2, 2, 4)
plt.imshow(edge_nlm, cmap='gray')
plt.title('Edges: Non-local Means')
plt.axis('off')

plt.tight_layout()
plt.savefig(f'{save_dir}/edges_comparison.png')
plt.show()

print(f"邊緣偵測完成！圖片已儲存至 {save_dir}")