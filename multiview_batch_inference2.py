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
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(alpha_channel, connectivity=8)

    if num_labels <= 1:
        return alpha_channel

    areas = stats[1:, cv2.CC_STAT_AREA]
    max_area_idx = np.argmax(areas) + 1

    filtered_image = np.zeros_like(alpha_channel, dtype=np.uint8)
    filtered_image[labels == max_area_idx] = 255

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
    alpha_channel = np.where(alpha_channel > alpha_threshold, 255, 0).astype(np.uint8)
    largest_alpha = filter_largest_region(alpha_channel)

    non_transparent = np.where(largest_alpha > 0)

    if len(non_transparent[0]) == 0 or len(non_transparent[1]) == 0:
        return image.resize(target_size, Image.Resampling.LANCZOS)

    top, bottom = np.min(non_transparent[0]), np.max(non_transparent[0])
    left, right = np.min(non_transparent[1]), np.max(non_transparent[1])
    cropped_img = image.crop((left, top, right, bottom))

    target_width = int(target_size[0] * object_ratio)
    target_height = int(target_size[1] * object_ratio)

    width, height = cropped_img.size
    scale = min(target_width / width, target_height / height)

    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_img = cropped_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 创建新的灰色背景图像 (RGBA)，这里的背景颜色是灰色 (128, 128, 128, 0)
    new_img = Image.new("RGBA", target_size, (127, 127, 127, 0))
    x_pos = (target_size[0] - new_width) // 2
    y_pos = (target_size[1] - new_height) // 2
    new_img.paste(resized_img, (x_pos, y_pos), resized_img)

    return new_img


def load_multiview_images(image_folder, camera_views, sample_name, target_size=(256, 256)):
    """
    加载多视角图像并调整大小，返回处理后的RGB图像列表。
    """
    images = []
    for view in camera_views:
        image_file = os.path.join(image_folder, f"{sample_name}_{view}.png")
        if os.path.exists(image_file):
            with Image.open(image_file) as img:
                # 裁剪并调整大小
                processed_img = crop_and_resize_image(img, target_size=target_size)
                # 将图像从 RGBA 转换为 RGB
                processed_img = processed_img.convert("RGB")
                images.append(processed_img)
        else:
            raise FileNotFoundError(f"Image {image_file} not found in {image_folder}")

    return images


def process_multiview_images(multiview_images, output_dir, pipeline, model, scale, step):
    os.makedirs(output_dir, exist_ok=True)

    print("Generating 3D model from multiview images...")
    stage2_images = pipeline.stage2_sample(multiview_images[5], multiview_images, scale=scale, step=step)

    np_imgs = np.concatenate([np.array(img) for img in multiview_images], axis=1)
    multiview_image_path = os.path.join(output_dir, "multiview_images.png")
    Image.fromarray(np_imgs).save(multiview_image_path)
    print(f"Saved multiview images to {multiview_image_path}")

    np_xyzs = np.concatenate([np.array(img) for img in stage2_images], axis=1)
    xyz_image_path = os.path.join(output_dir, "xyz_images.png")
    Image.fromarray(np_xyzs).save(xyz_image_path)
    print(f"Saved XYZ images to {xyz_image_path}")

    glb_path, obj_path = generate3d(model, np_imgs, np_xyzs, "cuda")

    shutil.copy(obj_path, os.path.join(output_dir, "output3d.obj"))
    print(f"Saved 3D model to {output_dir}")


def process_dataset(root_dir, output_root, camera_views, pipeline, model, scale, step, pbar):
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

            for sample_name in sample_names:
                sample_output_dir = os.path.join(output_root, dataset_name, sample_name)

                # 定义期望的输出文件
                multiview_image_path = os.path.join(sample_output_dir, "multiview_images.png")
                xyz_image_path = os.path.join(sample_output_dir, "xyz_images.png")
                obj_path = os.path.join(sample_output_dir, "output3d.obj")

                # 如果所有文件都存在，跳过处理
                if all(os.path.exists(path) for path in [multiview_image_path, xyz_image_path, obj_path]):
                    print(f"Sample {sample_name} already processed, skipping...")
                    pbar.update(1)
                    continue

                print(f"Processing sample: {sample_name} in dataset: {dataset_name}")

                try:
                    multiview_images = load_multiview_images(dataset_path, camera_views, sample_name)
                except FileNotFoundError as e:
                    print(f"Error: {e}")
                    pbar.update(1)
                    continue

                # 确保输出目录存在
                if not os.path.exists(sample_output_dir):
                    try:
                        os.makedirs(sample_output_dir, exist_ok=True)
                    except OSError as e:
                        print(f"Failed to create directory {sample_output_dir}: {e}")
                        pbar.update(1)
                        continue

                # 再次检查目录是否存在，防止文件系统延迟
                if not os.path.exists(sample_output_dir):
                    print(f"Directory {sample_output_dir} still does not exist after creation attempt.")
                    pbar.update(1)
                    continue

                # 保存预处理后的图像
                preprocessed_image_path = os.path.join(sample_output_dir, "preprocessed_image.png")
                try:
                    multiview_images[5].save(preprocessed_image_path)
                    print(f"Saved preprocessed image to {preprocessed_image_path}")
                except FileNotFoundError as e:
                    print(f"Error saving preprocessed image: {e}")
                    pbar.update(1)
                    continue

                # 调用处理函数
                process_multiview_images(multiview_images, sample_output_dir, pipeline, model, scale, step)

                pbar.update(1)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--step",
        type=int,
        default=50,
    )
    args = parser.parse_args()

    directories = [
        ("/workspace/data1/", "/workspace/crm_multiview_new/data1"),
        ("/workspace/data2/", "/workspace/crm_multiview_new/data2"),
        ("/workspace/data3/", "/workspace/crm_multiview_new/data3"),
        ("/workspace/data4/", "/workspace/crm_multiview_new/data4"),
    ]

    crm_path = "/workspace/model/CRM/CRM.pth"
    specs = json.load(open("configs/specs_objaverse_total.json"))
    model = CRM(specs).to("cuda")
    model.load_state_dict(torch.load(crm_path, map_location="cuda"), strict=False)

    stage1_config = OmegaConf.load("configs/nf7_v3_SNR_rd_size_stroke.yaml").config
    stage2_config = OmegaConf.load("configs/stage2-v2-snr.yaml").config
    stage1_sampler_config = stage1_config.sampler
    stage2_sampler_config = stage2_config.sampler
    stage1_model_config = stage1_config.models
    stage2_model_config = stage2_config.models

    pipeline = TwoStagePipeline(
        stage1_model_config,
        stage2_model_config,
        stage1_sampler_config,
        stage2_sampler_config,
        device="cuda"
    )

    total_samples = 0
    for input_dir, _ in directories:
        total_samples += len([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])

    with tqdm(total=total_samples, desc="Processing Samples") as pbar:
        for input_dir, output_dir in directories:
            os.makedirs(output_dir, exist_ok=True)
            process_dataset(input_dir, output_dir, ["001", "002", "003", "004", "005", "000"], pipeline, model, args.scale, args.step, pbar)

    print("3D model generation complete.")
