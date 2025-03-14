import os
import numpy as np
import cv2
from skimage import exposure
from skimage.metrics import peak_signal_noise_ratio as psnr


def linear_adjust(y_img, ref_img):
    """通过线性回归调整图像以最小化MSE"""
    y_float = y_img.astype(np.float64)
    ref_float = ref_img.astype(np.float64)

    mu_y = np.mean(y_float)
    mu_ref = np.mean(ref_float)

    # 计算协方差和方差
    covariance = np.cov(y_float.flatten(), ref_float.flatten())[0, 1]
    var_y = np.var(y_float)

    if var_y == 0:
        return y_img  # 避免除以零

    a = covariance / var_y
    b = mu_ref - a * mu_y

    adjusted = a * y_float + b
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return adjusted


def histogram_match(y_img, ref_img):
    """直方图匹配"""
    matched = exposure.match_histograms(y_img, ref_img, channel_axis=None)
    return matched.astype(np.uint8)


def process_images(input_dir, ref_dir, output_dir):
    """处理目录中的所有图像"""
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.bmp'):
            y_path = os.path.join(input_dir, filename)
            ref_path = os.path.join(ref_dir, filename)

            # 读取图像
            y_img = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
            ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)

            if y_img is None or ref_img is None:
                print(f"错误：无法读取 {filename} 或参考图像")
                continue

            if y_img.shape != ref_img.shape:
                print(f"警告：{filename} 与参考图像尺寸不符，跳过")
                continue

            # 应用两种增强方法
            linear_enhanced = linear_adjust(y_img, ref_img)
            hist_matched = histogram_match(y_img, ref_img)

            # 计算PSNR
            psnr_orig = psnr(ref_img, y_img, data_range=255)
            psnr_linear = psnr(ref_img, linear_enhanced, data_range=255)
            psnr_hist = psnr(ref_img, hist_matched, data_range=255)

            # 选择最佳结果
            if psnr_linear >= psnr_hist:
                best_img = linear_enhanced
                best_psnr = psnr_linear
                method = 'linear'
            else:
                best_img = hist_matched
                best_psnr = psnr_hist
                method = 'histogram'

            # 保存图像
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, best_img)

            print(f"图像: {filename}\n原PSNR: {psnr_orig:.2f} dB | {method} PSNR: {best_psnr:.2f} dB\n")


# 使用示例
input_dir = 'path/to/your/input_images'
ref_dir = 'path/to/your/reference_images'
output_dir = 'path/to/your/output_folder'

process_images(input_dir, ref_dir, output_dir)