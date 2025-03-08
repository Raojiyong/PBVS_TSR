import os
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import random


def resize_image(image_path, output_path, scale_factor):
    with Image.open(image_path) as img:
        # 获取原始图像的尺寸
        width, height = img.size

        # 计算新的尺寸
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # 调整图像大小
        resized_img = img.resize((new_width, new_height), Image.BICUBIC)

        # 保存调整后的图像
        resized_img.save(output_path)


def process_images_in_directory(input_dir, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.bmp'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # 下采样（缩小2倍）
            resize_image(input_path, output_path, 0.5)

            # 上采样（放大2倍）
            resize_image(output_path, output_path, 2.0)


# def calculate_psnr_for_images(input_dir1, input_dir2):
#     # 确保两个目录存在
#     if not os.path.exists(input_dir1) or not os.path.exists(input_dir2):
#         raise ValueError("输入目录不存在")
#
#     # 获取两个目录中的文件列表
#     files1 = set(os.listdir(input_dir1))
#     files2 = set(os.listdir(input_dir2))
#
#     # 找到两个目录中都存在的文件
#     common_files = files1.intersection(files2)
#
#     if not common_files:
#         raise ValueError("两个目录中没有共同的文件")
#
#     psnr_values = []
#
#     for filename in common_files:
#         if filename.endswith('.bmp'):
#             img1_path = os.path.join(input_dir1, filename)
#             img2_path = os.path.join(input_dir2, filename)
#
#             with Image.open(img1_path) as img1, Image.open(img2_path) as img2:
#                 # 确保两张图像具有相同的尺寸
#                 if img1.size != img2.size:
#                     raise ValueError(f"文件 {filename} 的尺寸不匹配")
#
#                 # 将图像转换为numpy数组
#                 img1_array = np.array(img1.convert('RGB'))
#                 img2_array = np.array(img2.convert('RGB'))
#
#                 # 计算PSNR值
#                 psnr_value = psnr(img1_array, img2_array, data_range=255)
#                 psnr_values.append(psnr_value)
#
#     # 计算PSNR值的均值
#     mean_psnr = sum(psnr_values) / len(psnr_values)
#
#     return psnr_values, mean_psnr

def add_gaussian_noise(image, mean=0, std=25):
    # 获取图像的尺寸
    width, height = image.size
    # 生成噪声数组，大小与图像相同
    noise = np.random.normal(mean, std, (height, width, 3))
    # 将噪声添加到图像上
    noisy_image = np.array(image) + noise
    # 确保像素值在0到255之间
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)


def calculate_psnr_for_images(input_dir1, input_dir2, output_dir, target_psnr=26.8, initial_std=25, step=0.1):
    # 确保两个目录存在
    if not os.path.exists(input_dir1) or not os.path.exists(input_dir2):
        raise ValueError("输入目录不存在")

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取两个目录中的文件列表
    files1 = set(os.listdir(input_dir1))
    files2 = set(os.listdir(input_dir2))

    # 找到两个目录中都存在的文件
    common_files = files1.intersection(files2)

    if not common_files:
        raise ValueError("两个目录中没有共同的文件")

    psnr_values = []
    std = initial_std

    for filename in common_files:
        if filename.endswith('.bmp'):
            img1_path = os.path.join(input_dir1, filename)
            img2_path = os.path.join(input_dir2, filename)

            with Image.open(img1_path) as img1, Image.open(img2_path) as img2:
                # 确保两张图像具有相同的尺寸
                if img1.size != img2.size:
                    raise ValueError(f"文件 {filename} 的尺寸不匹配")

                # 将图像转换为numpy数组
                img1_array = np.array(img1.convert('RGB'))
                img2_array = np.array(img2.convert('RGB'))

                # 计算原始PSNR值
                original_psnr_value = psnr(img1_array, img2_array, data_range=255)

                # 调整噪声标准差以达到目标PSNR值
                while True:
                    noisy_img2 = add_gaussian_noise(img2, std=std)
                    noisy_img2_array = np.array(noisy_img2)
                    noisy_psnr_value = psnr(img1_array, noisy_img2_array, data_range=255)

                    if noisy_psnr_value >= target_psnr:
                        break

                    std -= step

                # 保存添加噪声后的图像
                noisy_img2.save(os.path.join(output_dir, filename))

                # 存储PSNR值
                psnr_values.append((filename, original_psnr_value, noisy_psnr_value, std))

    # 计算原始PSNR值的均值
    mean_original_psnr = sum(value[1] for value in psnr_values) / len(psnr_values)

    # 计算添加噪声后的PSNR值的均值
    mean_noisy_psnr = sum(value[2] for value in psnr_values) / len(psnr_values)

    return psnr_values, mean_original_psnr, mean_noisy_psnr


# 指定输入和输出目录
input_directory = 'datasets/track1/thermal/test/GT'
input_directory_2 = 'results/0217_3'
input_directory_3 = 'results/0217_4'

# 处理图像
# process_images_in_directory(input_directory, output_directory)
psnr_values, mean_psnr, mean_noisy_psnr = calculate_psnr_for_images(
    input_directory, input_directory_2, input_directory_3)
print('hello')
