import os
import random
import cv2
import numpy as np
import math


def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames


def crop_random_quarter(image):
    height, width = image.shape[:2]
    quarter_height, quarter_width = height // 2, width // 2
    x = random.choice([0, quarter_width])
    y = random.choice([0, quarter_height])
    return image[y:y + quarter_height, x:x + quarter_width]


def crop_random_specific_quarter(lr_img, gt_img, start_x, start_y, quarter_h, quarter_w):
    lr_cropped = lr_img[start_y:start_y + quarter_h, start_x:start_x + quarter_w]

    gt_start_x = start_x * 8
    gt_start_y = start_y * 8
    gt_quarter_h = quarter_h * 8
    gt_quarter_w = quarter_w * 8

    gt_cropped = gt_img[gt_start_y:gt_start_y + gt_quarter_h, gt_start_x:gt_start_x + gt_quarter_w]

    return lr_cropped, gt_cropped


def combine_images(images):
    combined_image = np.vstack([
        np.hstack([images[0], images[1]]),
        np.hstack([images[2], images[3]])
    ])
    return combined_image


def process_images(lr_folder, gt_folder, lr_mix_folder, gt_mix_folder):
    lr_images, lr_filenames = load_images_from_folder(lr_folder)
    gt_images, gt_filenames = load_images_from_folder(gt_folder)

    if len(lr_images) != len(gt_images):
        raise ValueError("LR and GT directories must have the same number of images.")

    num_images = len(lr_images)
    processed_count = 0
    processed_combinations = set()

    # number of merged images
    total_combinations = 5e4  # math.comb(num_images, 4)

    while processed_count < total_combinations:
        selected_indices = tuple(sorted(random.sample(range(num_images), 4)))
        # selected_indices = tuple(random.sample(range(num_images), 4))
        if selected_indices in processed_combinations:
            continue

        lr_batch = [lr_images[i] for i in selected_indices]
        gt_batch = [gt_images[i] for i in selected_indices]

        h, w = lr_batch[0].shape[:2]
        quarter_h, quarter_w = h // 2, w // 2
        start_x = random.randint(0, w - quarter_w)
        start_y = random.randint(0, h - quarter_h)
        cropped_pairs = [crop_random_specific_quarter(lr_img, gt_img, start_x, start_y, quarter_h, quarter_w)
                         for lr_img, gt_img in zip(lr_batch, gt_batch)]
        lr_cropped = [pair[0] for pair in cropped_pairs]
        gt_cropped = [pair[1] for pair in cropped_pairs]

        lr_combined = combine_images(lr_cropped)
        gt_combined = combine_images(gt_cropped)

        lr_output_path = os.path.join(lr_mix_folder, f"mixed_{processed_count}.bmp")
        gt_output_path = os.path.join(gt_mix_folder, f"mixed_{processed_count}.bmp")

        cv2.imwrite(lr_output_path, lr_combined)
        cv2.imwrite(gt_output_path, gt_combined)

        processed_combinations.add(selected_indices)

        processed_count += 1


# train+val path
lr_folder = 'datasets/track1/thermal/train/LR_x8'
gt_folder = 'datasets/track1/thermal/train/GT'
lr_mix_folder = 'datasets/track1/thermal/train/LR_x8_mix_pretrain'
gt_mix_folder = 'datasets/track1/thermal/train/GT_mix_pretrain'

os.makedirs(lr_mix_folder, exist_ok=True)
os.makedirs(gt_mix_folder, exist_ok=True)

# process images
process_images(lr_folder, gt_folder, lr_mix_folder, gt_mix_folder)
