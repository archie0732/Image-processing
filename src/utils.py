import numpy as np


def quantize_image(image, levels):
    """
    將影像量化到指定的灰階級數 (levels)。

    量化步驟:
    1. 計算每個量化級別的寬度 (step = 256 / levels)。
    2. 將原始像素值除以 step，四捨五入後乘上 step，得到量化值。
    """
    # OpenCV 指令集：numpy.uint8()
    # 灰階級數 (levels) 必須大於等於 2
    if levels < 2:
        return image

    # 計算量化步長 (每個灰階級別的寬度)
    step = 256 / levels

    # 核心量化公式：[pixel / step] * step
    # 這是將 0-255 的值映射到 0 到 (levels-1) 個級別，再將其映射回 0-255 範圍
    # NumPy 運算 (高效能)
    quantized_image = np.floor(image / step) * step

    # 確保資料型態為 uint8 (0-255)
    return quantized_image.astype(np.uint8)


def floyd_steinberg_dithering(image):
    # 確保影像為浮點數，以便處理誤差
    dithered_img = image.astype(np.float32).copy()

    H, W = dithered_img.shape

    for y in range(H):
        for x in range(W):
            # 取得原始像素值
            old_pixel = dithered_img[y, x]

            # 將像素量化到最近的 0 或 255
            # 這是二元 dithering 的量化目標
            new_pixel = 255 * (old_pixel > 127)
            dithered_img[y, x] = new_pixel

            # 計算量化誤差
            quant_error = old_pixel - new_pixel

            # 擴散誤差到鄰近像素 (Floyd-Steinberg 係數)
            # 係數: [7/16] right, [3/16] down-left, [5/16] down, [1/16] down-right

            if x + 1 < W:
                dithered_img[y, x + 1] += quant_error * (7 / 16)
            if y + 1 < H:
                if x - 1 >= 0:
                    dithered_img[y + 1, x - 1] += quant_error * (3 / 16)
                dithered_img[y + 1, x] += quant_error * (5 / 16)
                if x + 1 < W:
                    dithered_img[y + 1, x + 1] += quant_error * (1 / 16)

    # 確保輸出為 uint8 格式
    return dithered_img.clip(0, 255).astype(np.uint8)
