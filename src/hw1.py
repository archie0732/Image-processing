import cv2 as c
import os
from yachalk import chalk
from utils import floyd_steinberg_dithering as floyd
from utils import quantize_image as quantize
import matplotlib.pyplot as plt

# print("OpenCV Version: ", c.__version__)

e_tif = "./img/engineer.tif"
c_tif = "./img/cameraman.tif"


def q1():
    print(chalk.cyan("Question 1"))
    try:
        gray = c.imread(os.path.join(e_tif), c.IMREAD_GRAYSCALE)
        if gray is None:
            raise IOError("img engineer 404!")

        print((f"ans1 (h * w): "), chalk.green(f"{gray.shape}"))

        jpg = "./img/engineer.jpg"
        c.imwrite(os.path.join(jpg), gray)
        print((f"ans2 (jpg size): "), chalk.green(f"{os.path.getsize(jpg)} bytes"))

        png = "./img/engineer.png"
        c.imwrite(os.path.join(png), gray)

        print((f"ans3 (png size) "), chalk.green(f"{os.path.getsize(png)} bytes"))

        return {
            "ans1": gray.shape,
            "ans2": os.path.getsize(jpg),
            "ans3": os.path.getsize(png),
        }
    except IOError as e:
        print(chalk.red(e))


def q2():
    print(chalk.cyan("Question 2"))
    try:
        gray = c.imread(os.path.join(c_tif), c.IMREAD_GRAYSCALE)

        if gray is None:
            raise IOError("img cathedral 404!")
        _, b_img = c.threshold(gray, 127, 255, c.THRESH_BINARY)
        b_img_path = "./img/cathedral_binary.png"
        c.imwrite(b_img_path, b_img)

        print("ans1 (bin size): ", chalk.green(f"{os.path.getsize(b_img_path)} bytes"))

        color_img = c.imread(os.path.join(c_tif), c.IMREAD_COLOR)
        color_img_path = "./img/cathedral_color.png"
        c.imwrite(color_img_path, color_img)
        print(
            "ans2 (color size): ",
            chalk.green(f"{os.path.getsize(color_img_path)} bytes"),
        )

        return {
            "ans1": os.path.getsize(b_img_path),
            "ans2": os.path.getsize(color_img_path),
        }
    except IOError as e:
        print(chalk.red(e))


def q3():
    print(chalk.cyan("Question 3"))
    print(chalk.yellow("Show images in a window..."))
    print(chalk.red("Close the window(Ctrl + C) to end the program..."))
    try:
        gray = c.imread(os.path.join(c_tif), c.IMREAD_GRAYSCALE)
        if gray is None:
            raise IOError("img cathedral 404!")
        quantized_2_dithered = floyd(gray)
        quantized_2_level = quantize(gray, 2)
        quantized_4_level = quantize(gray, 4)

        # 列印是顯示在終端機上??
        # 是存檔案嗎??
        # 還是直接顯示在視窗上??
        # 這邊我選擇直接顯示在視窗上
        # 這樣比較直觀
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(quantized_2_level, cmap="gray")
        plt.title("picture 1: 2 gray [10%]", fontsize=14)
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.imshow(quantized_4_level, cmap="gray")
        plt.title("picture 2: 4 gray [10%]", fontsize=14)
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.imshow(quantized_2_dithered, cmap="gray")
        plt.title("picture 3: 2 (Floyd-Steinberg) [20%]", fontsize=14)
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    except IOError as e:
        print(chalk.red(e))


q1()
print("===============================")
q2()
print("===============================")
q3()
