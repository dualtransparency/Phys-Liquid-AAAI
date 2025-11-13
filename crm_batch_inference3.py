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
    cropped_images_dir = os.path.join(sample_dir, "cropped_images")

    if not os.path.exists(cropped_images_dir):
        print(f"Warning: {cropped_images_dir} does not exist, skipping...")
        return

    # Process each image in the cropped_images directory
    for image_name in os.listdir(cropped_images_dir):
        if not image_name.endswith(".png"):
            continue

        view_angle = os.path.splitext(image_name)[0].split('_')[-1]
        # Skip the image if the view angle is "002" or "005"
        if view_angle in ["002", "005"]:
            print(f"Skipping {image_name} due to unwanted view angle ({view_angle})")
            continue

        image_path = os.path.join(cropped_images_dir, image_name)
        img = Image.open(image_path)

        # Create output directory for this image
        image_output_dir = os.path.join(output_dir, os.path.splitext(image_name)[0])
        os.makedirs(image_output_dir, exist_ok=True)

        # Check if the final .obj file already exists
        obj_file_path = os.path.join(image_output_dir, "output3d.obj")
        if os.path.exists(obj_file_path):
            print(f"Final result for {image_name} already exists, skipping...")
            continue  # Skip this image as the final result already exists

        # Check if preprocessed image already exists
        preprocessed_image_path = os.path.join(image_output_dir, "preprocessed_image.png")
        if os.path.exists(preprocessed_image_path):
            print(f"Preprocessed image for {image_name} already exists, loading...")
            img = Image.open(preprocessed_image_path)
        else:
            # Preprocess the image if not already done
            img = preprocess_image(img, args.bg_choice, 1.0, (127, 127, 127))
            img.save(preprocessed_image_path)

        # Check if stage1 and stage2 images already exist
        pixel_images_path = os.path.join(image_output_dir, "pixel_images.png")
        xyz_images_path = os.path.join(image_output_dir, "xyz_images.png")

        if os.path.exists(pixel_images_path) and os.path.exists(xyz_images_path):
            print(f"Stage1 and Stage2 images for {image_name} already exist, loading...")
            np_imgs = np.array(Image.open(pixel_images_path))
            np_xyzs = np.array(Image.open(xyz_images_path))
        else:
            # Run the pipeline if stage1 and stage2 images do not exist
            rt_dict = pipeline(img, scale=args.scale, step=args.step)
            stage1_images = rt_dict["stage1_images"]
            stage2_images = rt_dict["stage2_images"]

            # Save stage1 and stage2 images
            np_imgs = np.concatenate(stage1_images, 1)
            np_xyzs = np.concatenate(stage2_images, 1)
            Image.fromarray(np_imgs).save(pixel_images_path)
            Image.fromarray(np_xyzs).save(xyz_images_path)

        # Generate 3D models and save only the .obj file
        glb_path, obj_path = generate3d(model, np_imgs, np_xyzs, "cuda")
        shutil.copy(obj_path, obj_file_path)
        # Do not save the .glb file


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

    # Define the input-output directory pairs
    directories = [
        ("/workspace/data1/sam_data/", "/workspace/crm_inference_new/data1_trained/"),
        # ("/workspace/data2/sam_data/", "/workspace/crm_inference_new/data2_original"),
        # ("/workspace/data3/sam_data/", "/workspace/crm_inference_new/data3_trained"),
        # ("/workspace/data4/sam_data/", "/workspace/crm_inference_new/data4_trained"),
    ]

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
    for input_dir, _ in directories:
        total_samples += len([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])

    # Loop over each input-output directory pair with a global progress bar
    with tqdm(total=total_samples, desc="Processing Samples") as pbar:
        for input_dir, output_dir in directories:
            # Process each sample folder in the input directory
            for sample_folder in os.listdir(input_dir):
                sample_dir = os.path.join(input_dir, sample_folder)
                if os.path.isdir(sample_dir):
                    output_sample_dir = os.path.join(output_dir, sample_folder)
                    os.makedirs(output_sample_dir, exist_ok=True)
                    process_sample(sample_dir, output_sample_dir, args, model, pipeline)
                    pbar.update(1)  # Update the progress bar after processing each sample
