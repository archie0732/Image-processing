import cv2
import numpy as np
import os

# ==========================================
# 0. 準備工作
# ==========================================
output_dir = 'final_report_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

img_path = '圖片.jpg'
if not os.path.exists(img_path) and os.path.exists('./img/img1.tif'):
    img_path = './img/img1.tif'

print(f"正在讀取: {img_path}")
img = cv2.imread(img_path)
if img is None:
    print("錯誤：找不到圖片")
    exit()

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 儲存圖片 1: 原始影像
cv2.imwrite(f'{output_dir}/1_original.jpg', img_gray)
print("已儲存: 原始影像")

# ==========================================
# Task 1: 影像還原 (NLM 去噪)
# ==========================================
print("正在執行 Task 1: NLM 去噪...")
# h=25 強度設高一點以去除橫紋，這張圖會比較柔和乾淨
img_restored = cv2.fastNlMeansDenoising(img_gray, None, h=25, templateWindowSize=7, searchWindowSize=21)
# 儲存圖片 2: 最佳還原影像 (柔和乾淨版 - 推薦用於後續步驟)
cv2.imwrite(f'{output_dir}/2_restored_best.jpg', img_restored)

# === [新增] 補遺：額外產生一張銳利一點的版本供比較 ===
print("正在生成補遺：銳利化版本...")
# 原理：反銳化遮罩 (Unsharp Masking)
# 將 NLM 乾淨的圖進行高斯模糊
gaussian_blur = cv2.GaussianBlur(img_restored, (9, 9), 10.0)
# 利用公式加強邊緣：原圖 * 1.5 + 模糊圖 * (-0.5)
img_sharpened = cv2.addWeighted(img_restored, 1.5, gaussian_blur, -0.5, 0)
# 儲存補遺圖片
cv2.imwrite(f'{output_dir}/2_restored_sharpened_supplement.jpg', img_sharpened)
print("已儲存: 補遺銳利化影像 (2_restored_sharpened_supplement.jpg)")
# ========================================================


# ==========================================
# Task 2: 邊緣偵測 (使用最乾淨的 NLM 版本)
# ==========================================
print("正在執行 Task 2: 邊緣偵測...")
# 注意：這裡我們還是用 img_restored (柔和版)，因為雜訊少，抓出來的邊緣最漂亮
img_edges = cv2.Canny(img_restored, 50, 150)
cv2.imwrite(f'{output_dir}/3_edges.jpg', img_edges)

# ==========================================
# Task 3: 4灰階與最佳混色
# ==========================================
print("正在執行 Task 3: 4灰階混色...")

# A. 產生 "4灰階影像" (不混色)
def quantize_simple(image, levels=4):
    return (np.round((image / 255.0) * (levels - 1)) / (levels - 1) * 255).astype(np.uint8)
img_4_gray = quantize_simple(img_restored)
cv2.imwrite(f'{output_dir}/4_quantized_simple.jpg', img_4_gray)

# B. 產生 "最佳混色影像" (Floyd-Steinberg Dithering)
def floyd_steinberg(image, levels=4):
    h, w = image.shape
    dithered = image.astype(np.float64)
    max_val = 255.0
    step = max_val / (levels - 1)
    for y in range(h):
        for x in range(w):
            old = dithered[y, x]
            new = round(old / step) * step
            dithered[y, x] = new
            error = old - new
            if x + 1 < w: dithered[y, x + 1] += error * 0.4375
            if x - 1 >= 0 and y + 1 < h: dithered[y + 1, x - 1] += error * 0.1875
            if y + 1 < h: dithered[y + 1, x] += error * 0.3125
            if x + 1 < w and y + 1 < h: dithered[y + 1, x + 1] += error * 0.0625
    return np.clip(dithered, 0, 255).astype(np.uint8)

img_dithered = floyd_steinberg(img_restored)
cv2.imwrite(f'{output_dir}/5_dithered_best.jpg', img_dithered)

print("----------------------------------")
print("所有圖片生成完畢！請檢查 final_report_images 資料夾")
print("您可以比較 '2_restored_best.jpg' (柔和) 與 '2_restored_sharpened_supplement.jpg' (銳利) 決定報告要用哪張。")