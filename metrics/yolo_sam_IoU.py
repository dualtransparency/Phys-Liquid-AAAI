import os
import numpy as np
from PIL import Image
import json


def calculate_iou(mask1, mask2):
    """
    计算两个mask之间的IoU
    :param mask1: Ground Truth mask (numpy array)
    :param mask2: Predicted mask (numpy array)
    :return: IoU值
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def process_ground_truth(gt_image):
    """
    处理Ground Truth图像，将非背景部分作为前景
    :param gt_image: Ground Truth PIL图像
    :return: 前景mask (numpy array)
    """
    # Convert image to RGBA if it has an alpha channel, otherwise RGB
    if gt_image.mode == 'RGBA':
        gt_image = gt_image.convert('RGBA')
        # Separate the alpha channel
        r, g, b, a = gt_image.split()
        gt_mask = np.array(a) > 0  # Alpha通道大于0的部分为前景
    else:
        gt_image = gt_image.convert('RGB')
        # Convert RGB to numpy array
        gt_array = np.array(gt_image)
        # Assume background is (32, 32, 32), everything else is foreground
        gt_mask = np.logical_not(np.all(gt_array == [32, 32, 32], axis=-1))

    return gt_mask


def process_predicted_mask(pred_image):
    """
    处理预测mask，将白色部分作为前景
    :param pred_image: 预测mask PIL图像
    :return: 前景mask (numpy array)
    """
    pred_image = pred_image.convert('L')  # Convert to grayscale
    pred_mask = np.array(pred_image) > 128  # 将白色部分（大于128）作为前景
    return pred_mask


def process_sample(sample_dir, sam_data_dir):
    """
    处理单个样本文件夹，计算每个图像的IoU，并返回每个文件的IoU和平均IoU
    :param sample_dir: 样本文件夹路径 (Ground Truth)
    :param sam_data_dir: sam_data文件夹路径 (Predicted masks)
    :return: 每个文件的IoU字典，平均IoU
    """
    liquid_dir = os.path.join(sample_dir, 'liquid')
    masks_dir = os.path.join(sam_data_dir, 'masks')

    iou_results = {}
    total_iou = 0.0
    count = 0

    for gt_filename in os.listdir(liquid_dir):
        if gt_filename.endswith('.png'):
            # Ground Truth mask path
            gt_path = os.path.join(liquid_dir, gt_filename)

            # Predicted mask path (matching file with score)
            pred_filename = gt_filename.replace('.png', '') + '_*.png'
            pred_path = None
            for file in os.listdir(masks_dir):
                if file.startswith(gt_filename.replace('.png', '')):
                    pred_path = os.path.join(masks_dir, file)
                    break

            if pred_path is None:
                print(f"Predicted mask not found for {gt_filename}")
                continue

            # Load Ground Truth and Predicted masks
            gt_image = Image.open(gt_path)
            pred_image = Image.open(pred_path)

            # Process Ground Truth and Predicted masks
            gt_mask = process_ground_truth(gt_image)  # Ground Truth前景mask
            pred_mask = process_predicted_mask(pred_image)  # 预测前景mask

            # Calculate IoU
            iou = calculate_iou(gt_mask, pred_mask)
            iou_results[gt_filename] = iou
            total_iou += iou
            count += 1

    average_iou = total_iou / count if count > 0 else 0.0
    return iou_results, average_iou


def process_all_samples(root_dir, sam_data_dir, output_json_path):
    """
    处理所有样本，计算每个样本的IoU并保存到JSON文件
    :param root_dir: 根目录，包含所有样本文件夹
    :param sam_data_dir: sam_data根目录，包含预测的masks
    :param output_json_path: 保存结果的JSON文件路径
    """
    all_results = {}
    total_iou = 0.0
    total_count = 0

    for sample_folder in os.listdir(root_dir):
        sample_dir = os.path.join(root_dir, sample_folder)
        sam_sample_dir = os.path.join(sam_data_dir, sample_folder)

        if os.path.isdir(sample_dir) and os.path.isdir(sam_sample_dir):
            print(f"Processing sample: {sample_folder}")
            iou_results, average_iou = process_sample(sample_dir, sam_sample_dir)

            # 保存每个样本的IoU结果
            all_results[sample_folder] = {
                'file_iou': iou_results,
                'average_iou': average_iou
            }

            total_iou += average_iou
            total_count += 1

    # 计算所有样本的平均IoU
    overall_average_iou = total_iou / total_count if total_count > 0 else 0.0
    all_results['overall_average_iou'] = overall_average_iou

    # 保存结果到JSON文件
    with open(output_json_path, 'w') as json_file:
        json.dump(all_results, json_file, indent=4)

    print(f"Results saved to {output_json_path}")


if __name__ == "__main__":
    # 根目录路径
    root_dir = '/workspace/data4'
    sam_data_dir = os.path.join(root_dir, 'sam_data')

    # 输出JSON文件路径
    output_json_path = os.path.join(sam_data_dir, 'iou_results.json')

    # 处理所有样本并保存结果
    process_all_samples(root_dir, sam_data_dir, output_json_path)
