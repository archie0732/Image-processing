import cv2
import numpy as np
import os

# 建立輸出資料夾
output_dir = 'final_report_result'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. 讀取影像 (學號末碼 0, 3 使用 img1)
img_path = './img/img1.tif' 
# 若檔名不同請自行修改，例如 './img/img1.tif'
if not os.path.exists(img_path) and os.path.exists('圖片.jpg'):
    img_path = '圖片.jpg'

print(f"正在讀取影像: {img_path}")
img = cv2.imread(img_path)

if img is None:
    print("錯誤: 找不到圖片")
    exit()

# 儲存原始影像供報告對照
cv2.imwrite(f'{output_dir}/1_original.jpg', img)

# ========================================================
# Task 1: 影像還原 (Restoration) - 佔 60%
# 策略: 使用針對彩色影像的 Non-local Means 去除條紋與噪點
# ========================================================
print("執行 Task 1: 影像還原...")

# 參數說明: 
# h=15 (亮度去噪強度, 數值越大去越乾淨但越糊), hColor=15 (色彩去噪強度)
# templateWindowSize=7, searchWindowSize=21 (標準參數)
img_restored = cv2.fastNlMeansDenoisingColored(img, None, 15, 15, 7, 21)

# [加分優化] 銳利化 (Unsharp Masking)
# 原理: 原始圖 + (原始圖 - 模糊圖) * 強度
gaussian = cv2.GaussianBlur(img_restored, (9, 9), 10.0)
img_restored = cv2.addWeighted(img_restored, 1.5, gaussian, -0.5, 0)

cv2.imwrite(f'{output_dir}/2_restored_best.jpg', img_restored)
print("-> 已儲存最佳還原影像: 2_restored_best.jpg")

# ========================================================
# Task 2: 邊緣偵測 (Edge Detection) - 佔 20%
# 策略: 使用還原後的影像轉灰階 -> Canny
# ========================================================
print("執行 Task 2: 邊緣偵測...")

# 轉灰階
img_gray = cv2.cvtColor(img_restored, cv2.COLOR_BGR2GRAY)

# Canny 邊緣偵測
# threshold1=50, threshold2=150 為常用閾值
edges = cv2.Canny(img_gray, 50, 150)

cv2.imwrite(f'{output_dir}/3_edges.jpg', edges)
print("-> 已儲存邊緣偵測影像: 3_edges.jpg")

# ========================================================
# Task 3: 4灰階與混色 (Dithering) - 佔 20%
# 策略: 實作 Floyd-Steinberg 誤差擴散法
# ========================================================
print("執行 Task 3: 4灰階混色...")

# 定義 4 灰階量化函數
def quantize_simple(image, levels=4):
    # 將 0-255 對應到 4 個值 (0, 85, 170, 255)
    return (np.round((image / 255.0) * (levels - 1)) / (levels - 1) * 255).astype(np.uint8)

# 產生單純 4 灰階 (無混色)
img_4_level = quantize_simple(img_gray)
cv2.imwrite(f'{output_dir}/4_gray_4level.jpg', img_4_level)

# 定義 Floyd-Steinberg Dithering 演算法
def floyd_steinberg_dither(image, levels=4):
    h, w = image.shape
    # 轉 float 進行誤差計算
    dithered = image.astype(np.float64)
    max_val = 255.0
    step = max_val / (levels - 1)

    for y in range(h):
        for x in range(w):
            old_pixel = dithered[y, x]
            # 找到最近的階層顏色
            new_pixel = round(old_pixel / step) * step
            dithered[y, x] = new_pixel
            
            # 計算誤差
            quant_error = old_pixel - new_pixel
            
            # 擴散誤差到周圍 (右, 左下, 下, 右下)
            if x + 1 < w:
                dithered[y, x + 1] += quant_error * 7 / 16
            if x - 1 >= 0 and y + 1 < h:
                dithered[y + 1, x - 1] += quant_error * 3 / 16
            if y + 1 < h:
                dithered[y + 1, x] += quant_error * 5 / 16
            if x + 1 < w and y + 1 < h:
                dithered[y + 1, x + 1] += quant_error * 1 / 16
                
    return np.clip(dithered, 0, 255).astype(np.uint8)

# 產生最佳混色影像
img_dithered = floyd_steinberg_dither(img_gray, levels=4)
cv2.imwrite(f'{output_dir}/5_dithered_best.jpg', img_dithered)
print("-> 已儲存最佳混色影像: 5_dithered_best.jpg")

print(f"\n全部完成！請檢查 '{output_dir}' 資料夾中的圖片。")