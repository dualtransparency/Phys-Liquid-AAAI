import json
import os
import numpy as np
from PIL import Image
import torch
import cv2
from omegaconf import OmegaConf
from tqdm import tqdm
from model import CRM
from pipelines import TwoStagePipeline
from inference import generate3d
import shutil


def filter_largest_region(alpha_channel):
    """
    只保留面积最大的连通区域，去除所有其他区域。
    使用OpenCV的连通区域分析来过滤掉小的噪声。
    """
    # 获取所有连通区域及其统计信息
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(alpha_channel, connectivity=8)

    # 如果没有找到连通区域，或者只有背景区域，直接返回原图
    if num_labels <= 1:
        return alpha_channel

    # 获取每个连通区域的面积（stats[:, cv2.CC_STAT_AREA]）
    # stats 是一个二维数组，包含每个连通区域的统计信息，第四列是面积
    areas = stats[1:, cv2.CC_STAT_AREA]  # 忽略第 0 个标签（背景）

    # 找到面积最大的连通区域的索引
    max_area_idx = np.argmax(areas) + 1  # 因为 areas 是从第 1 个标签开始的，所以要加 1

    # 创建一个新的空白图像，只保留最大连通区域
    filtered_image = np.zeros_like(alpha_channel, dtype=np.uint8)
    filtered_image[labels == max_area_idx] = 255  # 将最大区域的像素设为 255（白色）

    return filtered_image


def crop_and_resize_image(image, target_size=(512, 512), object_ratio=0.9, alpha_threshold=128):
    """
    裁剪并调整图像大小，目标尺寸为 target_size，物体占比为 object_ratio，保留透明通道。
    通过过滤小区域来避免裁剪时被噪声影响。
    """
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    img_array = np.array(image)
    alpha_channel = img_array[:, :, 3]

    # 只保留Alpha值大于阈值的区域
    alpha_channel = np.where(alpha_channel > alpha_threshold, 255, 0).astype(np.uint8)

    # 只保留最大的连通区域
    largest_alpha = filter_largest_region(alpha_channel)

    # 找到非透明区域（即非零区域）
    non_transparent = np.where(largest_alpha > 0)

    if len(non_transparent[0]) == 0 or len(non_transparent[1]) == 0:
        # 如果没有找到非透明的像素，直接返回原图的缩放版本
        return image.resize(target_size, Image.Resampling.LANCZOS)

    # 获取几何对象的边界框
    top, bottom = np.min(non_transparent[0]), np.max(non_transparent[0])
    left, right = np.min(non_transparent[1]), np.max(non_transparent[1])

    # 裁剪图像
    cropped_img = image.crop((left, top, right, bottom))

    # 计算目标宽度和高度
    target_width = int(target_size[0] * object_ratio)
    target_height = int(target_size[1] * object_ratio)

    # 计算缩放比例
    width, height = cropped_img.size
    scale_w = target_width / width
    scale_h = target_height / height
    scale = min(scale_w, scale_h)

    # 使用较小的缩放比例
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_img = cropped_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 创建新的透明背景图像 (RGBA)
    new_img = Image.new("RGBA", target_size, (255, 255, 255, 0))  # 透明背景
    # 计算放置位置
    x_pos = (target_size[0] - new_width) // 2
    y_pos = (target_size[1] - new_height) // 2
    # 粘贴缩放后的图像到新图像上
    new_img.paste(resized_img, (x_pos, y_pos), resized_img)  # 使用透明通道粘贴

    # 转换为RGB格式，丢弃Alpha通道
    return new_img.convert("RGB")



def load_multiview_images(image_folder, camera_views, sample_name, target_size=(256, 256)):
    """
    加载多视角图像并调整大小，返回处理后的RGB图像列表。
    """
    images = []
    for view in camera_views:
        image_file = os.path.join(image_folder, f"{sample_name}_{view}.png")
        if os.path.exists(image_file):
            with Image.open(image_file) as img:
                processed_img = crop_and_resize_image(img, target_size=target_size)
                images.append(processed_img)
        else:
            raise FileNotFoundError(f"Image {image_file} not found in {image_folder}")

    return images


def process_multiview_images(multiview_images, output_dir, pipeline, model, scale, step):
    os.makedirs(output_dir, exist_ok=True)

    # Step 2: Use the multiview images directly as the input for stage2_sample
    print("Generating 3D model from multiview images...")
    # multiview_images[5]实际上是000视角，为正视角
    stage2_images = pipeline.stage2_sample(multiview_images[5], multiview_images, scale=scale, step=step)

    np_imgs = np.concatenate([np.array(img) for img in multiview_images], axis=1)
    multiview_image_path = os.path.join(output_dir, "multiview_images.png")
    Image.fromarray(np_imgs).save(multiview_image_path)
    print(f"Saved multiview images to {multiview_image_path}")

    np_xyzs = np.concatenate([np.array(img) for img in stage2_images], axis=1)
    xyz_image_path = os.path.join(output_dir, "xyz_images.png")
    Image.fromarray(np_xyzs).save(xyz_image_path)
    print(f"Saved XYZ images to {xyz_image_path}")

    # Step 3: Generate 3D model using generate3d
    glb_path, obj_path = generate3d(model, np_imgs, np_xyzs, "cuda")

    # Copy the output files to the output directory
    shutil.copy(glb_path, os.path.join(output_dir, "output3d.glb"))
    shutil.copy(obj_path, os.path.join(output_dir, "output3d.obj"))
    print(f"Saved 3D model to {output_dir}")


def process_dataset(root_dir, output_root, camera_views, pipeline, model, scale, step):
    """
    递归处理数据集中的每个样本。
    """
    for dataset_name in os.listdir(root_dir):
        if not dataset_name.startswith("F00"):
            continue

        dataset_path = os.path.join(root_dir, dataset_name, "liquid")
        if os.path.isdir(dataset_path):
            sample_names = set()
            for file_name in os.listdir(dataset_path):
                if file_name.endswith(".png"):
                    sample_name = "_".join(file_name.split("_")[:2])
                    sample_names.add(sample_name)

            for sample_name in tqdm(sample_names, desc=f"Processing dataset: {dataset_name}"):
                sample_output_dir = os.path.join(output_root, dataset_name, sample_name)

                # 定义期望的输出文件
                multiview_image_path = os.path.join(sample_output_dir, "multiview_images.png")
                xyz_image_path = os.path.join(sample_output_dir, "xyz_images.png")
                glb_path = os.path.join(sample_output_dir, "output3d.glb")
                obj_path = os.path.join(sample_output_dir, "output3d.obj")

                # 如果所有文件都存在，跳过处理
                if all(os.path.exists(path) for path in [multiview_image_path, xyz_image_path, glb_path, obj_path]):
                    print(f"Sample {sample_name} already processed, skipping...")
                    continue

                print(f"Processing sample: {sample_name} in dataset: {dataset_name}")

                try:
                    multiview_images = load_multiview_images(dataset_path, camera_views, sample_name)
                except FileNotFoundError as e:
                    print(f"Error: {e}")
                    continue

                # 调用处理函数
                process_multiview_images(multiview_images, sample_output_dir, pipeline, model, scale, step)



# if __name__ == "__main__":
#     import argparse
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--root_dir",
#         type=str,
#         default="/workspace/data2/",
#         help="Root directory containing the dataset folders",
#     )
#     parser.add_argument(
#         "--outdir",
#         type=str,
#         default="/workspace/data2/multiview_data/",
#         help="Root directory to save the output files",
#     )
#     parser.add_argument(
#         "--scale",
#         type=float,
#         default=5.0,
#     )
#     parser.add_argument(
#         "--step",
#         type=int,
#         default=50,
#     )
#     args = parser.parse_args()
#
#     # Define the camera views order (as provided)
#     camera_views = ["001", "002", "003", "004", "005", "000"]
#
#     # Load CRM model and configurations
#     print("Loading model and configurations...")
#     crm_path = "/workspace/model/CRM/CRM.pth"
#     specs = json.load(open("configs/specs_objaverse_total.json"))
#     model = CRM(specs).to("cuda")
#     model.load_state_dict(torch.load(crm_path, map_location="cuda"), strict=False)
#
#     # Load stage configurations
#     stage1_config = OmegaConf.load("configs/nf7_v3_SNR_rd_size_stroke.yaml").config
#     stage2_config = OmegaConf.load("configs/stage2-v2-snr.yaml").config
#     stage2_sampler_config = stage2_config.sampler
#     stage1_sampler_config = stage1_config.sampler
#     stage1_model_config = stage1_config.models
#     stage2_model_config = stage2_config.models
#
#     # Load model checkpoints
#     pixel_path = "/workspace/model/CRM/pixel-diffusion.pth"
#     xyz_path = "/workspace/model/CRM/ccm-diffusion.pth"
#     stage1_model_config.resume = pixel_path
#     stage2_model_config.resume = xyz_path
#
#     # Initialize pipeline
#     pipeline = TwoStagePipeline(
#         stage1_model_config,
#         stage2_model_config,
#         stage1_sampler_config,
#         stage2_sampler_config,
#         device="cuda"
#     )
#
#     # Process the dataset recursively
#     process_dataset(args.root_dir, args.outdir, camera_views, pipeline, model, args.scale, args.step)
#
#     print("3D model generation complete.")
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dirs",
        type=str,
        nargs='+',
        default=["/workspace/data2/", "/workspace/data3/", "/workspace/data4/"],
        help="List of root directories containing the dataset folders",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--step",
        type=int,
        default=100,
    )
    args = parser.parse_args()

    # Define the camera views order (as provided)
    camera_views = ["001", "002", "003", "004", "005", "000"]

    # Load CRM model and configurations
    print("Loading model and configurations...")
    crm_path = "/workspace/model/CRM/CRM.pth"
    specs = json.load(open("configs/specs_objaverse_total.json"))
    model = CRM(specs).to("cuda")
    model.load_state_dict(torch.load(crm_path, map_location="cuda"), strict=False)

    # Load stage configurations
    stage1_config = OmegaConf.load("configs/nf7_v3_SNR_rd_size_stroke.yaml").config
    stage2_config = OmegaConf.load("configs/stage2-v2-snr.yaml").config
    stage2_sampler_config = stage2_config.sampler
    stage1_sampler_config = stage1_config.sampler
    stage1_model_config = stage1_config.models
    stage2_model_config = stage2_config.models

    # Load model checkpoints
    pixel_path = "/workspace/model/CRM/pixel-diffusion.pth"
    xyz_path = "/workspace/model/CRM/ccm-diffusion.pth"
    stage1_model_config.resume = pixel_path
    stage2_model_config.resume = xyz_path

    # Initialize pipeline
    pipeline = TwoStagePipeline(
        stage1_model_config,
        stage2_model_config,
        stage1_sampler_config,
        stage2_sampler_config,
        device="cuda"
    )

    # Loop through each root directory
    for root_dir in args.root_dirs:
        output_dir = os.path.join(root_dir, "multiview_data")
        print(f"Processing dataset in {root_dir} with output to {output_dir}")
        process_dataset(root_dir, output_dir, camera_views, pipeline, model, args.scale, args.step)

    print("3D model generation complete.")
