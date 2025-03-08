import os
import sys

from PIL import Image
from pathlib import Path


def downsample_images(input_dir, output_dir):
    """将输入目录中的图像降采样2倍保存到输出目录

    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 支持的图像格式
    valid_ext = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp']

    for filename in os.listdir(input_dir):
        # 过滤有效图像文件
        filepath = os.path.join(input_dir, filename)
        if not os.path.isfile(filepath):
            continue
        ext = os.path.splitext(filename)[1].lower()
        if ext not in valid_ext:
            continue

        try:
            # 打开图像文件
            with Image.open(filepath) as img:
                # 计算新尺寸
                width, height = img.size
                new_size = (width // 2, height // 2)

                # 使用Lanczos重采样算法（高质量）
                img_resized = img.resize(new_size, resample=Image.Resampling.LANCZOS)

                # 生成输出路径
                output_name = os.path.splitext(filename)[0] + '.bmp'
                output_path = os.path.join(output_dir, output_name)

                # 保存为BMP格式（无损格式）
                img_resized.save(output_path, format='BMP')

        except Exception as e:
            print(f"处理文件 {filename} 失败: {str(e)}")


# 使用示例
input_dir, output_dir = sys.argv[1], sys.argv[2]
downsample_images(input_dir, output_dir)
