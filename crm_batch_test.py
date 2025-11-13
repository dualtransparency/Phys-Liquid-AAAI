import torch
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
from inference import generate3d
import json
import argparse
import shutil
from model import CRM
import os
from pipelines import TwoStagePipeline
from tqdm import tqdm
import random


def process_image(image_path, output_dir, pipeline, model, scale, step):
    img = Image.open(image_path)
    os.makedirs(output_dir, exist_ok=True)
    rt_dict = pipeline(img, scale=scale, step=step)
    stage1_images = rt_dict["stage1_images"]
    stage2_images = rt_dict["stage2_images"]
    np_imgs = np.concatenate(stage1_images, 1)
    np_xyzs = np.concatenate(stage2_images, 1)
    Image.fromarray(np_imgs).save(os.path.join(output_dir, "pixel_images.png"))
    Image.fromarray(np_xyzs).save(os.path.join(output_dir, "xyz_images.png"))
    glb_path, obj_path = generate3d(model, np_imgs, np_xyzs, "cuda:0")
    shutil.copy(glb_path, os.path.join(output_dir, "output3d.glb"))
    shutil.copy(obj_path, os.path.join(output_dir, "output3d.obj"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samdir",
        type=str,
        default="/workspace/data/sam_data/",
        help="Directory containing cropped images from sam_data",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="/workspace/data/crm_data/",
        help="Directory to save the CRM output files",
    )
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

    # Ensure output directory exists
    os.makedirs(args.outdir, exist_ok=True)

    # Load CRM model and configurations once
    print("Loading model and configurations...")
    crm_path = "/workspace/model/CRM/CRM.pth"
    specs = json.load(open("configs/specs_objaverse_total.json"))
    model = CRM(specs).to("cuda:0")
    model.load_state_dict(torch.load(crm_path, map_location="cuda:0"), strict=False)

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

    # Initialize pipeline once
    pipeline = TwoStagePipeline(
        stage1_model_config,
        stage2_model_config,
        stage1_sampler_config,
        stage2_sampler_config,
        device="cuda:0"
    )

    sample_folders = os.listdir(args.samdir)  # Assuming sample_folders is obtained from a directory listing
    random.shuffle(sample_folders)

    for sample_folder in tqdm(sample_folders, desc="Processing folders"):
        sample_path = os.path.join(args.samdir, sample_folder, "cropped_images")

        # Ensure the cropped_images folder exists
        if not os.path.exists(sample_path):
            print(f"Skipping {sample_folder}, no cropped_images folder found.")
            continue

        # Get list of image files and shuffle them for random traversal
        image_files = [f for f in os.listdir(sample_path) if f.endswith(".png")]
        random.shuffle(image_files)

        # Use tqdm to show progress for each image in the folder
        for filename in tqdm(image_files, desc=f"Processing images in {sample_folder}", leave=False):
            input_image_path = os.path.join(sample_path, filename)
            sample_output_dir = os.path.join(args.outdir, sample_folder, os.path.splitext(filename)[0])

            # 检查是否已经存在结果文件，避免重复处理
            output_glb = os.path.join(sample_output_dir, "output3d.glb")
            output_obj = os.path.join(sample_output_dir, "output3d.obj")

            if os.path.exists(output_glb) and os.path.exists(output_obj):
                print(f"Skipping {filename} in {sample_folder}, results already exist.")
                continue

            print(f"Processing {filename} in {sample_folder}...")
            process_image(input_image_path, sample_output_dir, pipeline, model, args.scale, args.step)

    print("Processing complete.")
