import torch
from libs.base_utils import do_resize_content
from imagedream.ldm.util import (
    instantiate_from_config,
    get_obj_from_str,
)
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
from inference import generate3d
import json
import argparse
import shutil
from model import CRM
import PIL
import rembg
import os
from pipelines import TwoStagePipeline
from tqdm import tqdm  # Import tqdm for progress bar

rembg_session = rembg.new_session()


def expand_to_square(image, bg_color=(0, 0, 0, 0)):
    # expand image to 1:1
    width, height = image.size
    if width == height:
        return image
    new_size = (max(width, height), max(width, height))
    new_image = Image.new("RGBA", new_size, bg_color)
    paste_position = ((new_size[0] - width) // 2, (new_size[1] - height) // 2)
    new_image.paste(image, paste_position)
    return new_image


def remove_background(
        image: PIL.Image.Image,
        rembg_session=None,
        force: bool = False,
        **rembg_kwargs,
) -> PIL.Image.Image:
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        # Skip removing background if image already has an alpha channel
        print("Alpha channel not empty, skipping background removal.")
        background = Image.new("RGBA", image.size, (0, 0, 0, 0))
        image = Image.alpha_composite(background, image)
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
    return image


def do_resize_content(original_image: Image, scale_rate):
    # Resize image content while retaining the original image size
    if scale_rate != 1:
        new_size = tuple(int(dim * scale_rate) for dim in original_image.size)
        resized_image = original_image.resize(new_size)
        padded_image = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
        paste_position = (
            (original_image.width - resized_image.width) // 2,
            (original_image.height - resized_image.height) // 2
        )
        padded_image.paste(resized_image, paste_position)
        return padded_image
    else:
        return original_image


def add_background(image, bg_color=(255, 255, 255)):
    # Add background color to an RGBA image using the alpha channel as a mask
    background = Image.new("RGBA", image.size, bg_color)
    return Image.alpha_composite(background, image)


def preprocess_image(image, background_choice, foreground_ratio, background_color):
    """
    Preprocess the input image (PIL image in RGBA) and return an RGB image.
    """
    print(background_choice)
    if background_choice == "Alpha as mask":
        background = Image.new("RGBA", image.size, (0, 0, 0, 0))
        image = Image.alpha_composite(background, image)
    else:
        image = remove_background(image, rembg_session, force=True)
    image = do_resize_content(image, foreground_ratio)
    image = expand_to_square(image)
    image = add_background(image, background_color)
    return image.convert("RGB")


def process_sample(sample_dir, output_dir, args, model, pipeline):
    # Note: we expect sample_dir to be the sub-folder like F0002PL8S3R5_0066_000
    # Check if preprocessed image already exists
    preprocessed_image_path = os.path.join(sample_dir, "preprocessed_image.png")
    if os.path.exists(preprocessed_image_path):
        print(f"Preprocessed image for {sample_dir} already exists, loading...")
        img = Image.open(preprocessed_image_path)
    else:
        print(f"Preprocessed image for {sample_dir} does not exist, skipping...")
        return  # Skip this sample if preprocessed image is missing

    # Check if pixel_images_new.png exists, otherwise use pixel_images.png
    pixel_images_new_path = os.path.join(sample_dir, "pixel_images_new.png")
    pixel_images_path = os.path.join(sample_dir, "pixel_images.png")
    xyz_images_path = os.path.join(sample_dir, "xyz_images.png")

    # Force regeneration of xyz_images.png and .obj if pixel_images_new.png exists
    if os.path.exists(pixel_images_new_path):
        print(f"Loading pixel_images_new.png for {sample_dir} and regenerating xyz_images.png and .obj...")
        np_imgs = np.array(Image.open(pixel_images_new_path))
        force_regenerate = True  # Force regeneration if pixel_images_new.png exists
    elif os.path.exists(pixel_images_path):
        print(f"Loading pixel_images.png for {sample_dir}...")
        np_imgs = np.array(Image.open(pixel_images_path))
        force_regenerate = False  # Only regenerate if xyz_images.png or .obj is missing
    else:
        print(f"Neither pixel_images_new.png nor pixel_images.png found for {sample_dir}, skipping...")
        return  # Skip this sample if neither image is available

    # Check if xyz_images.png exists, or if we need to regenerate it
    if not os.path.exists(xyz_images_path) or force_regenerate:
        print(f"Generating xyz_images.png for {sample_dir}...")
        rt_dict = pipeline(img, scale=args.scale, step=args.step)
        stage2_images = rt_dict["stage2_images"]
        np_xyzs = np.concatenate(stage2_images, 1)
        Image.fromarray(np_xyzs).save(xyz_images_path)

    # Generate 3D models and save only the .obj file
    obj_file_path = os.path.join(sample_dir, "output3d.obj")
    if not os.path.exists(obj_file_path) or force_regenerate:
        print(f"Generating 3D model for {sample_dir}...")
        glb_path, obj_path = generate3d(model, np_imgs, np_xyzs, "cuda")
        shutil.copy(obj_path, obj_file_path)
    else:
        print(f"3D model already exists for {sample_dir}, skipping generation.")


if __name__ == "__main__":
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
    parser.add_argument(
        "--bg_choice",
        type=str,
        default="Auto Remove background",
        help="[Auto Remove background] or [Alpha as mask]",
    )
    args = parser.parse_args()

    # Define the root directory
    root_dir = "/workspace/crm_oneview_multiview3"

    # Load model
    crm_path = "/workspace/model/CRM/CRM.pth"
    specs = json.load(open("configs/specs_objaverse_total.json"))
    model = CRM(specs).to("cuda")
    model.load_state_dict(torch.load(crm_path, map_location="cuda"), strict=False)

    # Load pipeline configurations
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
    )

    # Calculate the total number of samples across all directories
    total_samples = 0
    for sample_folder in os.listdir(root_dir):
        sample_folder_path = os.path.join(root_dir, sample_folder)
        if os.path.isdir(sample_folder_path):
            # Count the subfolders like F0002PL8S3R5_0066_000
            total_samples += len([d for d in os.listdir(sample_folder_path) if os.path.isdir(os.path.join(sample_folder_path, d))])

    # Loop over each sample folder in the root directory with a global progress bar
    with tqdm(total=total_samples, desc="Processing Samples") as pbar:
        for sample_folder in os.listdir(root_dir):
            sample_folder_path = os.path.join(root_dir, sample_folder)
            if os.path.isdir(sample_folder_path):
                # Process each subfolder like F0002PL8S3R5_0066_000
                for subfolder in os.listdir(sample_folder_path):
                    subfolder_path = os.path.join(sample_folder_path, subfolder)
                    if os.path.isdir(subfolder_path):
                        process_sample(subfolder_path, sample_folder_path, args, model, pipeline)
                        pbar.update(1)  # Update the progress bar after processing each sample
