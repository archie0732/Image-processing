import cv2
import numpy as np

def restore_high_quality(input_path, output_path):
    print(f"正在讀取: {input_path} ...")
    
    # 1. 強制灰階讀取
    img = cv2.imread(input_path, 0)
    
    if img is None:
        print("讀取失敗")
        return

    # --- 步驟 A: 智能去噪 (Non-Local Means) ---
    # 這是比高斯模糊、中值濾波更高階的方法
    # h: 決定過濾強度。設 10-15 可以去掉顆粒但保留線條。之前的方法像用砂紙磨，這個像用鑷子夾。
    denoised = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # --- 步驟 B: 增強對比度 (CLAHE) ---
    # 原圖灰濛濛的，這個步驟會讓黑的更黑，白的更白，讓輪廓跳出來
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(denoised)

    # --- 步驟 C: 銳化 (Unsharp Masking) ---
    # 用高斯模糊做遮罩來反向銳化，比直接用銳化矩陣更自然，不會有怪怪的白邊
    gaussian = cv2.GaussianBlur(contrast_enhanced, (0, 0), 3.0)
    sharpened = cv2.addWeighted(contrast_enhanced, 1.5, gaussian, -0.5, 0)

    # --- 儲存 ---
    tiff_params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]
    cv2.imwrite(output_path, sharpened, tiff_params)
    
    print("處理完成。請比較 output_v3.tif 與原圖。")
    
    # 顯示對比
    cv2.imshow('Original', img)
    cv2.imshow('V3 Result (Sharper)', sharpened)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 執行
restore_high_quality('./img/img1.tif', './img/output_v3.tif')