import cv2
import os

# ==========================================
# 設定區
# ==========================================
# 輸入的圖片路徑 (請確認您的檔案名稱)
input_filename = './img/img1.tif'  
# 輸出的圖片路徑 (自動存成 jpg)
output_filename = './img/img1_gray.jpg' 

# ==========================================
# 主程式
# ==========================================

# 1. 檢查檔案是否存在
if not os.path.exists(input_filename):
    print(f"找不到檔案: {input_filename}")
    # 如果找不到 tif，試試看有沒有 jpg (防止報錯用)
    if os.path.exists('圖片.jpg'):
        input_filename = '圖片.jpg'
        print(f"改讀取: {input_filename}")
    else:
        print("請確認圖片路徑是否正確！")
        exit()

# 2. 讀取圖片
img = cv2.imread(input_filename)

# 3. 轉為灰階
# 注意: OpenCV 讀取進來是 BGR 格式，所以要用 BGR2GRAY
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 4. 存檔 (存成 .jpg)
# jpg 是一種有損壓縮格式，但適合放在報告中減小檔案體積
cv2.imwrite(output_filename, img_gray)

print(f"成功！灰階圖片已儲存至: {output_filename}")