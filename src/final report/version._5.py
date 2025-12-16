import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 參數設定區 (根據您的圖片調整這裡)
# ==========================================
# 讀取圖片 (請確保圖片在同一目錄下)
# 根據評分標準，請確認這是您學號對應的正確圖片
img_path = './img/img1.tif' 

# [重要] 模糊核的長度 (Motion Blur Length)
# 這需要您手動測試。如果是程式生成的模糊，通常是整數。
# 請嘗試修改這個值：例如 20, 30, 40, 50, 60... 直到圖片變清晰
LEN = 40  

# 模糊角度 (Angle)
# 從圖片看是水平移動，所以角度設為 0
THETA = 0 

# 維納濾波的雜訊訊號比 (1/SNR)
# 數值越小，銳化越強但雜訊越多；數值越大，影像越平滑
SNR_INV = 0.01 

# ==========================================
# 第一部分：影像還原 (Wiener Filter) [佔分 60%]
# ==========================================
def wiener_deblur(img, length, angle, snr_inv):
    # 1. 建立點擴散函數 (PSF) - 模擬水平移動
    psf = np.zeros((img.shape[0], img.shape[1]))
    # 在中心畫一條線
    center = (img.shape[0] // 2, img.shape[1] // 2)
    cv2.ellipse(psf, center, (0, length // 2), angle, 0, 360, 1, -1)
    # 正規化 PSF，確保總和為 1
    psf /= psf.sum()

    # 2. 轉換到頻率域 (FFT)
    img_fft = np.fft.fft2(img)
    psf_fft = np.fft.fft2(psf)
    
    # 3. 維納濾波公式
    # G = H* / (|H|^2 + K)
    # 其中 H 是 PSF, K 是 SNR倒數
    psf_fft_conj = np.conj(psf_fft)
    wiener_factor = psf_fft_conj / (np.abs(psf_fft)**2 + snr_inv)
    
    result_fft = img_fft * wiener_factor
    
    # 4. 轉回空間域 (IFFT)
    result = np.fft.ifft2(result_fft)
    result = np.abs(np.fft.fftshift(result))
    
    # 調整數值範圍回 0-255
    return np.uint8(np.clip(result, 0, 255))

# 讀取灰階影像
original_blur = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if original_blur is None:
    print(f"錯誤：找不到檔案 {img_path}")
else:
    # 執行還原
    restored_img = wiener_deblur(original_blur, LEN, THETA, SNR_INV)

    # ==========================================
    # 第二部分：邊緣檢測 (Canny Edge) [佔分 20%]
    # ==========================================
    # 使用還原後的影像進行檢測
    # 門檻值 (50, 150) 可以微調以獲得最佳邊緣
    edges = cv2.Canny(restored_img, 50, 150)

    # ==========================================
    # 第三部分：4灰階與混色 (Dithering) [佔分 20%]
    # ==========================================
    
    # 3.1 製作單純的 4 灰階影像 (Quantization)
    # 將 0-255 分成 4 個區間: 0, 85, 170, 255
    def quantize_4_levels(value):
        if value < 42.5: return 0
        elif value < 127.5: return 85
        elif value < 212.5: return 170
        else: return 255

    # 向量化函數以加速處理
    v_quantize = np.vectorize(quantize_4_levels)
    simple_4_gray = v_quantize(restored_img).astype(np.uint8)

    # 3.2 製作最佳混色影像 (Floyd-Steinberg Error Diffusion)
    dithered_img = restored_img.astype(float).copy()
    h, w = dithered_img.shape

    for y in range(h):
        for x in range(w):
            old_pixel = dithered_img[y, x]
            new_pixel = quantize_4_levels(old_pixel) # 找最近的 4 階色
            dithered_img[y, x] = new_pixel
            
            quant_error = old_pixel - new_pixel
            
            # 將誤差擴散給周圍像素
            if x + 1 < w:
                dithered_img[y, x + 1] += quant_error * 7 / 16
            if x - 1 > 0 and y + 1 < h:
                dithered_img[y + 1, x - 1] += quant_error * 3 / 16
            if y + 1 < h:
                dithered_img[y + 1, x] += quant_error * 5 / 16
            if x + 1 < w and y + 1 < h:
                dithered_img[y + 1, x + 1] += quant_error * 1 / 16

    dithered_img = np.clip(dithered_img, 0, 255).astype(np.uint8)

    # ==========================================
    # 顯示與儲存結果 (用於報告列印)
    # ==========================================
    plt.figure(figsize=(12, 10))

    # 1. 原始 vs 還原
    plt.subplot(2, 3, 1)
    plt.title("Original Blur Image")
    plt.imshow(original_blur, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title(f"Restored (Len={LEN})") # 顯示使用的參數
    plt.imshow(restored_img, cmap='gray')
    plt.axis('off')

    # 2. 邊緣檢測
    plt.subplot(2, 3, 3)
    plt.title("Edge Detection (Canny)")
    plt.imshow(edges, cmap='gray')
    plt.axis('off')

    # 3. 4灰階 vs 混色
    plt.subplot(2, 3, 5)
    plt.title("Simple 4-Level Gray")
    plt.imshow(simple_4_gray, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title("Dithered 4-Level (Best Mix)")
    plt.imshow(dithered_img, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    
    print("處理完成。請記得截圖程式碼與結果圖放入報告。")