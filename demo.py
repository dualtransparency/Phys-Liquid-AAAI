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

# Initialize rembg session globally, but it will only be used if necessary
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
        use_rembg: bool = True,
        **rembg_kwargs,
) -> PIL.Image.Image:
    if not use_rembg:
        # If rembg is disabled, return the original image
        print("rembg is disabled, skipping background removal.")
        return image

    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        # explain why current do not rm bg
        print("alpha channel not empty, skip remove background, using alpha channel as mask")
        background = Image.new("RGBA", image.size, (0, 0, 0, 0))
        image = Image.alpha_composite(background, image)
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
    return image


def do_resize_content(original_image: Image, scale_rate):
    # resize image content while retaining the original image size
    if scale_rate != 1:
        # Calculate the new size after rescaling
        new_size = tuple(int(dim * scale_rate) for dim in original_image.size)
        # Resize the image while maintaining the aspect ratio
        resized_image = original_image.resize(new_size)
        # Create a new image with the original size and black background
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
    # given an RGBA image, alpha channel is used as mask to add background color
    background = Image.new("RGBA", image.size, bg_color)
    return Image.alpha_composite(background, image)


def preprocess_image(image, background_choice, foreground_ratio, backgroud_color, use_rembg):
    """
    input image is a pil image in RGBA, return RGB image
    """
    print(background_choice)
    if background_choice == "Alpha as mask":
        background = Image.new("RGBA", image.size, (0, 0, 0, 0))
        image = Image.alpha_composite(background, image)
    else:
        image = remove_background(image, rembg_session, force=True, use_rembg=use_rembg)
    image = do_resize_content(image, foreground_ratio)
    image = expand_to_square(image)
    image = add_background(image, backgroud_color)
    return image.convert("RGB")


def process_image(image_path, output_dir, pipeline, model, background_choice, scale, step, use_rembg):
    # Load image
    img = Image.open(image_path)

    # Preprocess image
    img = preprocess_image(img, background_choice, 1.0, (127, 127, 127), use_rembg)

    # Create output directory for this image
    os.makedirs(output_dir, exist_ok=True)

    # Save preprocessed image
    img.save(os.path.join(output_dir, "preprocessed_image.png"))

    # Run pipeline (no need to reload model/configurations)
    rt_dict = pipeline(img, scale=scale, step=step)
    stage1_images = rt_dict["stage1_images"]
    stage2_images = rt_dict["stage2_images"]

    # Save stage1 and stage2 images
    np_imgs = np.concatenate(stage1_images, 1)
    np_xyzs = np.concatenate(stage2_images, 1)
    Image.fromarray(np_imgs).save(os.path.join(output_dir, "pixel_images.png"))
    Image.fromarray(np_xyzs).save(os.path.join(output_dir, "xyz_images.png"))

    # Generate 3D model
    glb_path, obj_path = generate3d(model, np_imgs, np_xyzs, "cuda")

    # Copy the generated 3D files to output directory
    shutil.copy(glb_path, os.path.join(output_dir, "output3d.glb"))
    shutil.copy(obj_path, os.path.join(output_dir, "output3d.obj"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputdir",
        type=str,
        required=True,
        help="Directory containing input PNG images",
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
    parser.add_argument(
        "--bg_choice",
        type=str,
        default="Auto Remove background",
        help="[Auto Remove background] or [Alpha as mask]",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="out/",
        help="Directory to save the output files",
    )
    parser.add_argument(
        "--use_rembg",
        action="store_true",
        help="Enable rembg to remove background from images",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.outdir, exist_ok=True)

    # Load CRM model and configurations once
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
    # pixel_path = "/workspace/CRM/train_logs/nf7_v3_SNR_rd_size_stroke_train-default-2024-11-05T09-49-50/ckpts/unet-1008.pth"
    xyz_path = "/workspace/model/CRM/ccm-diffusion.pth"
    stage1_model_config.resume = pixel_path
    stage2_model_config.resume = xyz_path

    # Initialize pipeline once
    pipeline = TwoStagePipeline(
        stage1_model_config,
        stage2_model_config,
        stage1_sampler_config,
        stage2_sampler_config,
    )

    # Process all PNG files in the input directory
    for filename in os.listdir(args.inputdir):
        if filename.endswith(".png"):
            input_image_path = os.path.join(args.inputdir, filename)
            output_image_dir = os.path.join(args.outdir, os.path.splitext(filename)[0])

            print(f"Processing {filename}...")
            process_image(input_image_path, output_image_dir, pipeline, model, args.bg_choice, args.scale, args.step,
                          args.use_rembg)

    print("Processing complete.")
