import cv2
import numpy as np
from scipy.signal import wiener

# 1. 讀取原始圖片
img_path = 'img/img1.tif'
img_bgr = cv2.imread(img_path)

# 2. 轉為灰階 (這是第一張要下載的圖)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
cv2.imwrite('./img/original_gray.tif', img_gray)

# 3. 去除雜訊 (使用 5x5 的 Wiener 濾波器)
# 轉為 float64 運算以避免溢位，運算完再轉回 uint8
img_denoised = wiener(img_gray.astype(np.float64), (5, 5))
img_denoised = np.clip(img_denoised, 0, 255).astype(np.uint8)

# 4. 儲存去噪後的圖片 (這是第二張要下載的圖)
cv2.imwrite('./img/denoised_result.tif', img_denoised)

print("處理完成，已儲存 original_gray.tif 與 denoised_result.tif")

