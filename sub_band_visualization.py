import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt


def visualize_wavelet_transform(gray_frame):
    """对输入灰度图像进行二级小波变换，并显示生成的8个子带"""
    # 一级小波变换
    coeffs2 = pywt.dwt2(gray_frame, 'haar')
    LL, (LH, HL, HH) = coeffs2

    # 二级小波变换（对一级小波变换中的低频子带LL进行二次变换）
    coeffs3 = pywt.dwt2(LL, 'haar')
    LL2, (LH2, HL2, HH2) = coeffs3

    # 创建图像网格显示
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()

    # 显示 8 个子带图像
    images = [LL, LH, HL, HH, LL2, LH2, HL2, HH2]
    titles = ["LL", "LH", "HL", "HH", "LL2", "LH2", "HL2", "HH2"]

    for i in range(8):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(titles[i])
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# 读取测试图像
image_path = "sub_band_test.jpg"  # 替换为你的指静脉图像路径
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is not None:
    # 预处理：高斯滤波
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # 进行小波变换并可视化
    visualize_wavelet_transform(image)
else:
    print("无法加载图像，请检查文件路径！")
